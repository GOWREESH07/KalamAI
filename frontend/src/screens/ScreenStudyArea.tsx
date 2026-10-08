import React, { useEffect, useRef, useState } from 'react'
import { api, StepStatus, StudyAreaType, MaskStatsResponse } from '../api'
import BasemapSwitcher from '../components/BasemapSwitcher'
import { BASEMAP_SOURCES, getBasemapLayers, updateBasemapVisibility, BasemapType } from '../utils/basemaps'

interface Props {
  pipelineStatus: StepStatus[]
  onStatusChange: () => void
  studyArea: StudyAreaType
  setStudyArea?: (sa: StudyAreaType) => void
}

const MONTHS_LABELS = [
  'Jun 25','Jul 25','Aug 25','Sep 25','Oct 25','Nov 25','Dec 25',
  'Jan 26','Feb 26','Mar 26','Apr 26','May 26','Jun 26','Jul 26','Aug 26','Sep 26',
]

export default function ScreenStudyArea({ pipelineStatus, studyArea }: Props) {
  const mapRef = useRef<HTMLDivElement>(null)
  const mapInstanceRef = useRef<any>(null)
  const [basemap, setBasemap] = useState<BasemapType>('esri-world')
  const [coords, setCoords] = useState({ lon: 77.45, lat: 8.70 })
  const [availability, setAvailability] = useState<{ date: string; cloud_free_pct: number }[]>([])
  const [layer, setLayer] = useState<'rgb' | 'ndvi'>('rgb')
  const [monthIdx, setMonthIdx] = useState(0)
  const [featuresReady, setFeaturesReady] = useState(false)
  const [maskStats, setMaskStats] = useState<MaskStatsResponse | null>(null)

  useEffect(() => {
    api.getDataAvailability(studyArea).then(setAvailability).catch(() => {})
    if (studyArea === 'taluks') {
      api.getTalukStats().then(setMaskStats).catch(() => {})
    }
  }, [pipelineStatus, studyArea])

  const AOI_PILOT = { lon_min: 77.47, lat_min: 8.680, lon_max: 77.53, lat_max: 8.725 }
  const TALUKS_BBOX = { lon_min: 77.1697, lat_min: 8.5339, lon_max: 77.6619, lat_max: 8.8820 }

  const isTaluks = studyArea === 'taluks'
  const bbox = isTaluks ? TALUKS_BBOX : AOI_PILOT

  useEffect(() => {
    const preproc = pipelineStatus.find(s => s.name === 'preprocess')
    const feats = pipelineStatus.find(s => s.name === 'features')
    setFeaturesReady(feats?.status === 'done' || preproc?.status === 'done' || true)
  }, [pipelineStatus])

  useEffect(() => {
    if (!mapRef.current) return
    import('maplibre-gl').then(async ({ default: maplibregl }) => {
      if (mapInstanceRef.current) {
        mapInstanceRef.current.remove()
        mapInstanceRef.current = null
      }

      const initialCenter: [number, number] = [(bbox.lon_min + bbox.lon_max) / 2, (bbox.lat_min + bbox.lat_max) / 2]
      const initialZoom = isTaluks ? 10 : 12

      const map = new maplibregl.Map({
        container: mapRef.current!,
        maxZoom: 19,
        style: {
          version: 8,
          sources: BASEMAP_SOURCES,
          layers: getBasemapLayers(basemap),
        },
        center: initialCenter,
        zoom: initialZoom,
      })

      map.on('load', async () => {
        // Satellite layer overlay (True Colour / NDVI)
        const overlayId = 'kalam-overlay'
        const currentUrl = layer === 'rgb'
          ? api.getLayerUrl('rgb', undefined, studyArea)
          : api.getLayerUrl('ndvi_month', monthIdx, studyArea)

        map.addSource(overlayId, {
          type: 'image',
          url: currentUrl,
          coordinates: [
            [bbox.lon_min, bbox.lat_max],
            [bbox.lon_max, bbox.lat_max],
            [bbox.lon_max, bbox.lat_min],
            [bbox.lon_min, bbox.lat_min],
          ],
        })
        map.addLayer({
          id: overlayId + '-layer',
          type: 'raster',
          source: overlayId,
          paint: { 'raster-opacity': 0.85 },
        })

        if (isTaluks) {
          // Taluk administrative boundaries
          try {
            const taluksGeo = await api.getTalukBoundaries()
            map.addSource('taluks-boundary', {
              type: 'geojson',
              data: taluksGeo,
            })
            map.addLayer({
              id: 'taluks-fill',
              type: 'fill',
              source: 'taluks-boundary',
              paint: {
                'fill-color': '#3A7CA5',
                'fill-opacity': 0.05,
              },
            })
            map.addLayer({
              id: 'taluks-outline',
              type: 'line',
              source: 'taluks-boundary',
              paint: {
                'line-color': '#1F3A5F',
                'line-width': 2.5,
              },
            })
          } catch (e) {
            console.error(e)
          }

          // Light grey hatched excluded forest/hills mask
          try {
            const maskGeo = await api.getTalukMask()
            map.addSource('excluded-mask', {
              type: 'geojson',
              data: maskGeo,
            })
            map.addLayer({
              id: 'excluded-mask-fill',
              type: 'fill',
              source: 'excluded-mask',
              paint: {
                'fill-color': '#475569',
                'fill-opacity': 0.4,
              },
            })
            map.addLayer({
              id: 'excluded-mask-line',
              type: 'line',
              source: 'excluded-mask',
              paint: {
                'line-color': '#64748B',
                'line-width': 1.5,
                'line-dasharray': [3, 2],
              },
            })
          } catch (e) {
            console.error(e)
          }
        } else {
          // Pilot AOI outline
          map.addSource('aoi', {
            type: 'geojson',
            data: {
              type: 'Feature',
              geometry: {
                type: 'Polygon',
                coordinates: [[
                  [AOI_PILOT.lon_min, AOI_PILOT.lat_min],
                  [AOI_PILOT.lon_max, AOI_PILOT.lat_min],
                  [AOI_PILOT.lon_max, AOI_PILOT.lat_max],
                  [AOI_PILOT.lon_min, AOI_PILOT.lat_max],
                  [AOI_PILOT.lon_min, AOI_PILOT.lat_min],
                ]],
              },
              properties: {},
            },
          })
          map.addLayer({
            id: 'aoi-outline',
            type: 'line',
            source: 'aoi',
            paint: { 'line-color': '#1F3A5F', 'line-width': 2 },
          })
        }

        map.fitBounds([
          [bbox.lon_min, bbox.lat_min],
          [bbox.lon_max, bbox.lat_max],
        ], { padding: 40 })

        mapInstanceRef.current = map
      })

      map.on('mousemove', (e) => {
        setCoords({ lon: parseFloat(e.lngLat.lng.toFixed(5)), lat: parseFloat(e.lngLat.lat.toFixed(5)) })
      })
    })

    return () => {
      if (mapInstanceRef.current) {
        mapInstanceRef.current.remove()
        mapInstanceRef.current = null
      }
    }
  }, [studyArea])

  // Update basemap layer visibility on switcher change
  useEffect(() => {
    updateBasemapVisibility(mapInstanceRef.current, basemap)
  }, [basemap])

  // Switch layer & update image on map
  useEffect(() => {
    if (!mapInstanceRef.current) return
    const map = mapInstanceRef.current
    const src = map.getSource('kalam-overlay')
    if (!src || !src.updateImage) return
    const url = layer === 'rgb'
      ? api.getLayerUrl('rgb', undefined, studyArea)
      : api.getLayerUrl('ndvi_month', monthIdx, studyArea)
    src.updateImage({
      url,
      coordinates: [
        [bbox.lon_min, bbox.lat_max],
        [bbox.lon_max, bbox.lat_max],
        [bbox.lon_max, bbox.lat_min],
        [bbox.lon_min, bbox.lat_min],
      ]
    })
  }, [layer, monthIdx, studyArea])

  return (
    <div style={{ display: 'flex', height: '100%', overflow: 'hidden' }}>
      <div className="map-wrap" ref={mapRef}>
        <BasemapSwitcher basemap={basemap} onChange={setBasemap} />

        <div className="coord-readout">
          {coords.lon.toFixed(5)}°E &nbsp; {coords.lat.toFixed(5)}°N
        </div>

        {/* Legend for Taluks */}
        {isTaluks && (
          <div
            style={{
              position: 'absolute',
              bottom: 24,
              left: 24,
              background: 'rgba(10, 20, 36, 0.92)',
              backdropFilter: 'blur(6px)',
              padding: '10px 14px',
              borderRadius: 6,
              border: '1px solid #1F3A5F',
              fontSize: 11,
              color: '#E8F1F5',
              zIndex: 10,
              display: 'flex',
              flexDirection: 'column',
              gap: 8,
              boxShadow: '0 4px 12px rgba(0,0,0,0.4)',
            }}
          >
            <div style={{ fontWeight: 600, fontSize: 11, color: '#A5C2DE', textTransform: 'uppercase', letterSpacing: '0.04em' }}>
              Study Area Map Layers
            </div>
            <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
              <span style={{ width: 14, height: 3, background: '#1F3A5F', display: 'inline-block' }} />
              <span>Taluk Administrative Boundaries (OSM level 6)</span>
            </div>
            <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
              <span style={{ width: 14, height: 10, background: '#475569', border: '1px dashed #64748B', display: 'inline-block' }} />
              <span>Excluded: forest / hills (Western Ghats)</span>
            </div>
            <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
              <span style={{ width: 14, height: 10, background: 'rgba(58, 124, 165, 0.2)', border: '1px solid #3A7CA5', display: 'inline-block' }} />
              <span>Analysed Farmed Plains (Tamirabarani Basin)</span>
            </div>
          </div>
        )}
      </div>

      <div className="right-panel">
        <div className="panel-section">
          <div className="page-title" style={{ fontSize: 16 }}>
            {isTaluks ? 'Study Area & Data (Full Taluks)' : 'Study Area & Data (Pilot)'}
          </div>
          <div className="page-subtitle">Ambasamudram & Cheranmahadevi, Tirunelveli</div>
        </div>

        <div className="panel-scroll">
          {/* Layer toggle — identical for both */}
          <div className="section-label">Layer</div>
          <div className="layer-toggle" style={{ marginBottom: 16 }}>
            <button className={`layer-toggle-btn${layer === 'rgb' ? ' active' : ''}`} onClick={() => setLayer('rgb')}>True Colour</button>
            <button className={`layer-toggle-btn${layer === 'ndvi' ? ' active' : ''}`} onClick={() => setLayer('ndvi')}>NDVI</button>
          </div>

          {/* Month slider — identical for both */}
          {layer === 'ndvi' && (
            <div style={{ marginBottom: 16 }}>
              <div className="section-label" style={{ marginBottom: 6 }}>Month: {MONTHS_LABELS[monthIdx]}</div>
              <div className="month-slider-wrap">
                <input
                  type="range"
                  min={0}
                  max={MONTHS_LABELS.length - 1}
                  value={monthIdx}
                  onChange={e => setMonthIdx(+e.target.value)}
                  style={{ width: '100%' }}
                />
              </div>
              <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 10, fontFamily: 'var(--font-mono)', color: 'var(--text-secondary)', marginTop: 4 }}>
                <span>Jun 25</span><span>Sep 26</span>
              </div>
            </div>
          )}

          {/* Data availability — identical for both */}
          <div className="section-label">Data Availability</div>
          <div className="avail-strip" style={{ marginBottom: 4, overflowX: 'auto', paddingBottom: 4 }}>
            {MONTHS_LABELS.map((lbl, i) => {
              const pct = availability[i]?.cloud_free_pct ?? 100
              const h = Math.max(4, Math.round((pct / 100) * 44))
              return (
                <div key={i} style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 2 }}>
                  <div
                    className="avail-bar"
                    style={{ height: h, background: pct < 70 ? '#E06C75' : pct < 90 ? '#E5C07B' : 'var(--accent)' }}
                    title={`${lbl} (${availability[i]?.date || ''}): ${pct}% cloud-free`}
                  />
                </div>
              )
            })}
          </div>
          <div style={{ display: 'flex', overflowX: 'auto', gap: 0, marginBottom: 16 }}>
            {MONTHS_LABELS.map((lbl, i) => (
              <div key={i} style={{ width: 18, fontSize: 8, color: 'var(--text-secondary)', fontFamily: 'var(--font-mono)', textAlign: 'center', flexShrink: 0 }}>
                {lbl.slice(0, 1)}
              </div>
            ))}
          </div>

          <div className="sep" />

          {/* AOI Details table — identical format, dynamic values */}
          <div className="section-label">Study Area Details</div>
          <table className="data-table" style={{ marginBottom: 16 }}>
            <tbody>
              <tr>
                <td>Location</td>
                <td className="num mono">Ambasamudram & Cheranmahadevi</td>
              </tr>
              <tr>
                <td>Lon range</td>
                <td className="num mono">{bbox.lon_min.toFixed(2)}°–{bbox.lon_max.toFixed(2)}°E</td>
              </tr>
              <tr>
                <td>Lat range</td>
                <td className="num mono">{bbox.lat_min.toFixed(3)}°–{bbox.lat_max.toFixed(3)}°N</td>
              </tr>
              <tr>
                <td>CRS</td>
                <td className="num mono">EPSG:32643 (UTM 43N)</td>
              </tr>
              <tr>
                <td>Resolution</td>
                <td className="num mono">10 m</td>
              </tr>
              <tr>
                <td>{isTaluks ? 'Analysed Farmed Plains' : 'Total Area'}</td>
                <td className="num mono" style={{ fontWeight: 600, color: 'var(--c-paddy)' }}>
                  {isTaluks ? '548.7 km² (54,869 ha)' : '33.47 km² (3,347 ha)'}
                </td>
              </tr>
              {isTaluks && (
                <>
                  <tr>
                    <td>Gross Administrative Extent</td>
                    <td className="num mono">1,183.4 km²</td>
                  </tr>
                  <tr>
                    <td>Excluded Forest & Hills</td>
                    <td className="num mono" style={{ color: '#E06C75' }}>634.7 km² (53.6%)</td>
                  </tr>
                  <tr>
                    <td>Lowland Trees Kept (&lt;120m)</td>
                    <td className="num mono" style={{ color: 'var(--c-banana)' }}>135.1 km² (Banana-safe)</td>
                  </tr>
                </>
              )}
            </tbody>
          </table>

          {/* Extra taluk breakdown cards if in taluks mode */}
          {isTaluks && (
            <>
              <div className="section-label" style={{ fontSize: 10 }}>Taluk Administrative Breakdown</div>
              <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 8, marginBottom: 12 }}>
                <div style={{ background: '#0F1E36', padding: '8px 10px', borderRadius: 4, border: '1px solid #1F3A5F' }}>
                  <div style={{ fontSize: 9, color: '#7B93B2', textTransform: 'uppercase' }}>Ambasamudram</div>
                  <div style={{ fontSize: 13, fontWeight: 600, fontFamily: 'var(--font-mono)', color: '#E8F1F5' }}>
                    913.9 km²
                  </div>
                  <div style={{ fontSize: 9, color: '#5B7A9C' }}>309.1 km² farmed</div>
                </div>

                <div style={{ background: '#0F1E36', padding: '8px 10px', borderRadius: 4, border: '1px solid #1F3A5F' }}>
                  <div style={{ fontSize: 9, color: '#7B93B2', textTransform: 'uppercase' }}>Cheranmahadevi</div>
                  <div style={{ fontSize: 13, fontWeight: 600, fontFamily: 'var(--font-mono)', color: '#E8F1F5' }}>
                    269.5 km²
                  </div>
                  <div style={{ fontSize: 9, color: '#5B7A9C' }}>239.6 km² farmed</div>
                </div>
              </div>
            </>
          )}

          {/* Sensor specs */}
          <div className="section-label">Sensors</div>
          <table className="data-table">
            <tbody>
              <tr><td>Optical Sensor</td><td className="num mono">Sentinel-2 L2A (10 m)</td></tr>
              <tr><td>SAR Sensor</td><td className="num mono">Sentinel-1 RTC (10 m)</td></tr>
              <tr><td>Orbits</td><td className="num mono">Ascending & Descending</td></tr>
              <tr><td>Composites</td><td className="num mono">16 Monthly (Jun 25 – Sep 26)</td></tr>
            </tbody>
          </table>
        </div>
      </div>
    </div>
  )
}
