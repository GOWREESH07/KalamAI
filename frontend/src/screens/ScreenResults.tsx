import React, { useState, useEffect, useRef } from 'react'
import { api, Metrics, AreaStat, StepStatus, StudyAreaType, ZonalStatsResponse } from '../api'
import BasemapSwitcher from '../components/BasemapSwitcher'
import { BASEMAP_SOURCES, getBasemapLayers, updateBasemapVisibility, BasemapType } from '../utils/basemaps'

interface Props {
  pipelineStatus: StepStatus[]
  onStatusChange: () => void
  studyArea: StudyAreaType
  setStudyArea?: (sa: StudyAreaType) => void
}

const CLASS_COLORS: Record<string, string> = {
  Paddy: '#8DB33F',
  Banana: '#2E6B3F',
  'Other Veg': '#BFCB8F',
  'Built-up': '#B4554B',
  Water: '#3A7CA5',
  'Fallow/Bare': '#CBA670',
}

function DebugHistogramPanel({ studyArea = 'pilot' }: { studyArea?: StudyAreaType }) {
  const [histData, setHistData] = useState<Record<string, any> | null>(null)
  const [featureKey, setFeatureKey] = useState<string>('ndvi_std')
  const [loading, setLoading] = useState(false)

  useEffect(() => {
    setLoading(true)
    api.getDebugHistograms(studyArea)
      .then(setHistData)
      .catch(() => {})
      .finally(() => setLoading(false))
  }, [studyArea])

  if (loading) return <div style={{ fontSize: 11, color: 'var(--text-secondary)' }}>Loading feature histograms…</div>
  if (!histData) return null

  const featLabels: Record<string, string> = {
    ndvi_std: 'NDVI Std Dev',
    ndvi_months_gt_05: 'NDVI > 0.5 (months)',
    vh_min: 'VH Min (dB)',
    vh_mean: 'VH Mean (dB)',
  }

  return (
    <div style={{ marginTop: 8 }}>
      <div style={{ display: 'flex', gap: 4, marginBottom: 8, flexWrap: 'wrap' }}>
        {Object.entries(featLabels).map(([key, label]) => (
          <button
            key={key}
            onClick={() => setFeatureKey(key)}
            className={`btn btn-sm ${featureKey === key ? 'btn-primary' : 'btn-secondary'}`}
            style={{ fontSize: 10, padding: '2px 6px' }}
          >
            {label}
          </button>
        ))}
      </div>

      <div style={{ display: 'flex', flexDirection: 'column', gap: 6 }}>
        {Object.entries(histData).map(([cname, cval]: [string, any]) => {
          const counts: number[] = cval[featureKey]?.counts || []
          const maxVal = Math.max(...counts, 1)
          const col = CLASS_COLORS[cname] || '#888'

          return (
            <div key={cname} style={{ background: 'var(--bg)', padding: '5px 8px', borderRadius: 4 }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 10, marginBottom: 3 }}>
                <span style={{ fontWeight: 600, color: col }}>{cname}</span>
                <span className="mono" style={{ color: 'var(--text-secondary)' }}>n={cval.count.toLocaleString()}</span>
              </div>
              <div style={{ display: 'flex', alignItems: 'flex-end', gap: 1, height: 26 }}>
                {counts.map((cnt, bIdx) => (
                  <div
                    key={bIdx}
                    style={{
                      flex: 1,
                      background: col,
                      height: `${(cnt / maxVal) * 100}%`,
                      minHeight: cnt > 0 ? 2 : 0,
                      opacity: 0.85,
                      borderRadius: '1px 1px 0 0',
                    }}
                    title={`Bin ${bIdx}: ${cnt} px`}
                  />
                ))}
              </div>
            </div>
          )
        })}
      </div>
    </div>
  )
}

export default function ScreenResults({ pipelineStatus, studyArea }: Props) {
  const mapRef = useRef<HTMLDivElement>(null)
  const mapInstance = useRef<any>(null)
  const [basemap, setBasemap] = useState<BasemapType>('esri-world')
  const [metrics, setMetrics] = useState<Metrics | null>(null)
  const [stats, setStats] = useState<AreaStat[]>([])
  const [zonalStats, setZonalStats] = useState<ZonalStatsResponse | null>(null)
  const [opacity, setOpacity] = useState(0.85)
  const [viewMode, setViewMode] = useState<'parcel' | 'pixel'>('pixel')
  const [selectedParcel, setSelectedParcel] = useState<any>(null)
  const [coords, setCoords] = useState({ lon: 77.45, lat: 8.70 })

  const AOI_PILOT = { lon_min: 77.47, lat_min: 8.680, lon_max: 77.53, lat_max: 8.725 }
  const TALUKS_BBOX = { lon_min: 77.1697, lat_min: 8.5339, lon_max: 77.6619, lat_max: 8.8820 }

  useEffect(() => {
    api.getMetrics(studyArea).then(setMetrics).catch(() => {})
    api.getStats(studyArea).then(setStats).catch(() => {})
    if (studyArea === 'taluks') {
      api.getZonalStats().then(setZonalStats).catch(() => {})
    }
  }, [pipelineStatus, studyArea])

  useEffect(() => {
    if (!mapRef.current) return
    import('maplibre-gl').then(async ({ default: maplibregl }) => {
      if (mapInstance.current) {
        mapInstance.current.remove()
        mapInstance.current = null
      }

      const isTaluks = studyArea === 'taluks'
      const bbox = isTaluks ? TALUKS_BBOX : AOI_PILOT
      const center: [number, number] = [(bbox.lon_min + bbox.lon_max) / 2, (bbox.lat_min + bbox.lat_max) / 2]

      const map = new maplibregl.Map({
        container: mapRef.current!,
        maxZoom: 19,
        style: {
          version: 8,
          sources: BASEMAP_SOURCES,
          layers: getBasemapLayers(basemap),
        },
        center,
        zoom: isTaluks ? 9.5 : 12,
      })

      map.on('load', async () => {
        // Taluk boundaries & excluded forest if in taluks mode
        if (isTaluks) {
          try {
            const taluksGeo = await api.getTalukBoundaries()
            map.addSource('taluks-boundary', { type: 'geojson', data: taluksGeo })
            map.addLayer({
              id: 'taluks-line',
              type: 'line',
              source: 'taluks-boundary',
              paint: { 'line-color': '#1F3A5F', 'line-width': 2 },
            })

            const maskGeo = await api.getTalukMask()
            map.addSource('excluded-mask', { type: 'geojson', data: maskGeo })
            map.addLayer({
              id: 'excluded-mask-fill',
              type: 'fill',
              source: 'excluded-mask',
              paint: { 'fill-color': '#475569', 'fill-opacity': 0.4 },
            })
          } catch (e) {
            console.error(e)
          }
        }

        // Classified raster overlay
        map.addSource('classified-raster', {
          type: 'image',
          url: api.getLayerUrl('classified', undefined, studyArea),
          coordinates: [
            [bbox.lon_min, bbox.lat_max],
            [bbox.lon_max, bbox.lat_max],
            [bbox.lon_max, bbox.lat_min],
            [bbox.lon_min, bbox.lat_min],
          ],
        })
        map.addLayer({
          id: 'classified-raster-layer',
          type: 'raster',
          source: 'classified-raster',
          paint: { 'raster-opacity': opacity },
          layout: { visibility: viewMode === 'pixel' ? 'visible' : 'none' },
        })

        // Parcels geojson
        map.addSource('parcels', {
          type: 'geojson',
          data: api.getParcelsGeojson(studyArea),
        })

        map.addLayer({
          id: 'parcels-fill',
          type: 'fill',
          source: 'parcels',
          layout: { visibility: viewMode === 'parcel' ? 'visible' : 'none' },
          paint: {
            'fill-color': [
              'match',
              ['get', 'class_id'],
              1, '#8DB33F',
              2, '#2E6B3F',
              3, '#BFCB8F',
              4, '#B4554B',
              5, '#3A7CA5',
              6, '#CBA670',
              '#CCCCCC'
            ],
            'fill-opacity': opacity,
          },
        })

        map.addLayer({
          id: 'parcels-line',
          type: 'line',
          source: 'parcels',
          layout: { visibility: viewMode === 'parcel' ? 'visible' : 'none' },
          paint: { 'line-color': 'rgba(0,0,0,0.3)', 'line-width': 0.8 },
        })

        map.fitBounds([
          [bbox.lon_min, bbox.lat_min],
          [bbox.lon_max, bbox.lat_max],
        ], { padding: 40 })

        map.on('click', 'parcels-fill', (e) => {
          if (!e.features || e.features.length === 0) return
          setSelectedParcel(e.features[0].properties)
        })

        mapInstance.current = map
      })

      map.on('mousemove', (e) => {
        setCoords({ lon: parseFloat(e.lngLat.lng.toFixed(5)), lat: parseFloat(e.lngLat.lat.toFixed(5)) })
      })
    })

    return () => {
      if (mapInstance.current) {
        mapInstance.current.remove()
        mapInstance.current = null
      }
    }
  }, [studyArea])

  useEffect(() => {
    updateBasemapVisibility(mapInstance.current, basemap)
  }, [basemap])

  useEffect(() => {
    if (!mapInstance.current) return
    const map = mapInstance.current
    if (map.getLayer('classified-raster-layer')) {
      map.setLayoutProperty('classified-raster-layer', 'visibility', viewMode === 'pixel' ? 'visible' : 'none')
      map.setPaintProperty('classified-raster-layer', 'raster-opacity', opacity)
    }
    if (map.getLayer('parcels-fill')) {
      map.setLayoutProperty('parcels-fill', 'visibility', viewMode === 'parcel' ? 'visible' : 'none')
      map.setLayoutProperty('parcels-line', 'visibility', viewMode === 'parcel' ? 'visible' : 'none')
      map.setPaintProperty('parcels-fill', 'fill-opacity', opacity)
    }
  }, [viewMode, opacity])

  return (
    <div style={{ display: 'flex', height: '100%', overflow: 'hidden' }}>
      <div className="map-wrap" ref={mapRef}>
        <BasemapSwitcher basemap={basemap} onChange={setBasemap} />

        <div className="coord-readout">
          {coords.lon.toFixed(5)}°E &nbsp; {coords.lat.toFixed(5)}°N
        </div>

        {/* Floating map controls */}
        <div className="map-controls">
          <div className="map-control-group">
            <span style={{ fontSize: 10, color: 'var(--text-secondary)' }}>View:</span>
            <button
              className={`btn btn-sm ${viewMode === 'pixel' ? 'btn-primary' : 'btn-secondary'}`}
              onClick={() => setViewMode('pixel')}
            >
              Classified Raster
            </button>
            <button
              className={`btn btn-sm ${viewMode === 'parcel' ? 'btn-primary' : 'btn-secondary'}`}
              onClick={() => setViewMode('parcel')}
            >
              Parcels
            </button>
          </div>
          <div className="map-control-group">
            <span style={{ fontSize: 10, color: 'var(--text-secondary)' }}>Opacity:</span>
            <input
              type="range"
              min="0"
              max="1"
              step="0.05"
              value={opacity}
              onChange={e => setOpacity(parseFloat(e.target.value))}
              style={{ width: 80 }}
            />
            <span className="mono" style={{ fontSize: 10 }}>{(opacity * 100).toFixed(0)}%</span>
          </div>
        </div>

        {/* Selected parcel card */}
        {selectedParcel && (
          <div className="selected-parcel-card">
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 4 }}>
              <span style={{ fontWeight: 600, fontSize: 12 }}>Parcel #{selectedParcel.parcel_id}</span>
              <button
                onClick={() => setSelectedParcel(null)}
                style={{ background: 'none', border: 'none', color: 'var(--text-secondary)', cursor: 'pointer', fontSize: 14 }}
              >
                ×
              </button>
            </div>
            <div style={{ display: 'flex', alignItems: 'center', gap: 6, marginBottom: 4 }}>
              <span
                style={{
                  width: 10,
                  height: 10,
                  borderRadius: 2,
                  background: CLASS_COLORS[selectedParcel.class_name] || '#888',
                }}
              />
              <span style={{ fontWeight: 600, fontSize: 13 }}>{selectedParcel.class_name}</span>
            </div>
            <div style={{ fontSize: 11, color: 'var(--text-secondary)' }}>
              Confidence: <span className="mono">{(selectedParcel.confidence * 100).toFixed(1)}%</span> &nbsp;|&nbsp;
              Area: <span className="mono">{selectedParcel.area_ha?.toFixed(2)} ha</span>
            </div>
          </div>
        )}
      </div>

      <div className="right-panel">
        <div className="panel-section">
          <div className="page-title" style={{ fontSize: 16 }}>Classification Results</div>
          <div className="page-subtitle">
            {studyArea === 'taluks' ? 'Ambasamudram & Cheranmahadevi (Full Extent)' : 'Pilot Study Area (33.5 km²)'}
          </div>
        </div>

        <div className="panel-scroll">
          {/* Taluk Zonal Statistics if in taluks mode */}
          {studyArea === 'taluks' && zonalStats && (
            <div style={{ marginBottom: 16 }}>
              <div className="section-label">Zonal Statistics: Ambasamudram vs Cheranmahadevi</div>
              <table className="data-table" style={{ fontSize: 11, marginBottom: 8 }}>
                <thead>
                  <tr style={{ color: 'var(--text-secondary)' }}>
                    <th>Class</th>
                    <th style={{ textAlign: 'right' }}>Ambasamudram</th>
                    <th style={{ textAlign: 'right' }}>Cheranmahadevi</th>
                    <th style={{ textAlign: 'right' }}>Total</th>
                  </tr>
                </thead>
                <tbody>
                  {zonalStats.classes.map(c => (
                    <tr key={c.name}>
                      <td style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
                        <span style={{ width: 8, height: 8, borderRadius: 2, background: c.color }} />
                        <span>{c.name}</span>
                      </td>
                      <td className="num mono">{c.ambasamudram_ha.toLocaleString()} ha ({c.ambasamudram_pct.toFixed(1)}%)</td>
                      <td className="num mono">{c.cheranmahadevi_ha.toLocaleString()} ha ({c.cheranmahadevi_pct.toFixed(1)}%)</td>
                      <td className="num mono" style={{ fontWeight: 600 }}>{c.total_ha.toLocaleString()} ha</td>
                    </tr>
                  ))}
                </tbody>
              </table>
              <div style={{ fontSize: 10, color: '#7B93B2', lineHeight: 1.4, padding: '6px 8px', background: '#0F1E36', borderRadius: 4 }}>
                * Coincides strictly with analysed farmed plains (548.7 km²). Excluded Western Ghats forest: 604.9 km² (Ambasamudram) & 29.8 km² (Cheranmahadevi).
              </div>
              <div className="sep" />
            </div>
          )}

          {/* Area stats breakdown */}
          <div className="section-label">Area by Class</div>
          <table className="data-table" style={{ marginBottom: 16 }}>
            <thead>
              <tr>
                <th>Class</th>
                <th className="num">Area (ha)</th>
                <th className="num">Area (km²)</th>
              </tr>
            </thead>
            <tbody>
              {stats.map(s => (
                <tr key={s.class}>
                  <td style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
                    <span
                      style={{
                        width: 8,
                        height: 8,
                        borderRadius: 2,
                        background: CLASS_COLORS[s.class] || '#888',
                      }}
                    />
                    {s.class}
                  </td>
                  <td className="num mono">{s.area_ha.toLocaleString(undefined, { maximumFractionDigits: 1 })}</td>
                  <td className="num mono">{s.area_km2.toFixed(2)}</td>
                </tr>
              ))}
            </tbody>
          </table>

          <div className="sep" />

          {/* Accuracy Metrics */}
          <div className="section-label">Model Evaluation</div>
          {metrics ? (
            <div>
              <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 8, marginBottom: 16 }}>
                <div className="metric-tile" style={{ padding: '8px 10px' }}>
                  <div className="metric-label">
                    {metrics.is_seed_agreement ? 'Seed Agreement' : 'Validated Accuracy'}
                  </div>
                  <div className="metric-value" style={{ fontSize: 18 }}>{(metrics.overall_accuracy * 100).toFixed(1)}%</div>
                </div>
                <div className="metric-tile" style={{ padding: '8px 10px' }}>
                  <div className="metric-label">Cohen Kappa</div>
                  <div className="metric-value" style={{ fontSize: 18 }}>{metrics.kappa.toFixed(3)}</div>
                </div>
              </div>

              {/* F1 per class */}
              <div className="section-label" style={{ fontSize: 10 }}>Per-Class F1 Score</div>
              <table className="data-table" style={{ marginBottom: 16 }}>
                <tbody>
                  {Object.entries(metrics.f1_per_class || {}).map(([c, val]) => (
                    <tr key={c}>
                      <td>{c}</td>
                      <td className="num mono">{(val * 100).toFixed(1)}%</td>
                    </tr>
                  ))}
                </tbody>
              </table>

              <div className="sep" />
              <div className="section-label">Feature Distributions (Tuning Histograms)</div>
              <DebugHistogramPanel studyArea={studyArea} />
            </div>
          ) : (
            <div className="empty-state" style={{ padding: 12 }}>
              <div style={{ fontSize: 12 }}>Metrics will appear after classification runs.</div>
            </div>
          )}
        </div>
      </div>
    </div>
  )
}
