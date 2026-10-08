import React, { useState, useEffect, useRef } from 'react'
import { api, StepStatus, Metrics } from '../api'
import { Plus, Trash2, Save } from 'lucide-react'
import BasemapSwitcher from '../components/BasemapSwitcher'
import { BASEMAP_SOURCES, getBasemapLayers, updateBasemapVisibility, BasemapType } from '../utils/basemaps'

interface Props {
  pipelineStatus: StepStatus[]
  onStatusChange: () => void
  studyArea?: any
  setStudyArea?: any
}

const CLASS_OPTIONS = [
  { id: 'paddy', name: 'Paddy', color: '#8DB33F' },
  { id: 'banana', name: 'Banana', color: '#2E6B3F' },
  { id: 'other_veg', name: 'Other Veg', color: '#BFCB8F' },
  { id: 'builtup', name: 'Built-up', color: '#B4554B' },
  { id: 'water', name: 'Water', color: '#3A7CA5' },
  { id: 'fallow', name: 'Fallow/Bare', color: '#CBA670' },
]

const AOI_PILOT = { lon_min: 77.47, lat_min: 8.680, lon_max: 77.53, lat_max: 8.725 }
const TALUKS_BBOX = { lon_min: 77.1697, lat_min: 8.5339, lon_max: 77.6619, lat_max: 8.8820 }

export default function ScreenTraining({ pipelineStatus, studyArea = 'pilot' }: Props) {
  const mapRef = useRef<HTMLDivElement>(null)
  const mapInstance = useRef<any>(null)
  const [basemap, setBasemap] = useState<BasemapType>('esri-world')
  const [selectedClass, setSelectedClass] = useState('paddy')
  const [labels, setLabels] = useState<any[]>([])
  const [coords, setCoords] = useState({ lon: 77.45, lat: 8.70 })
  const [drawingPoints, setDrawingPoints] = useState<[number, number][]>([])
  const [saving, setSaving] = useState(false)
  const [metricsData, setMetricsData] = useState<Metrics | null>(null)

  const isTaluks = studyArea === 'taluks'
  const bbox = isTaluks ? TALUKS_BBOX : AOI_PILOT

  useEffect(() => {
    api.getLabels().then((res: any) => {
      if (res && res.features) setLabels(res.features)
    }).catch(() => {})
    api.getMetrics(studyArea).then(setMetricsData).catch(() => {})
  }, [pipelineStatus, studyArea])

  useEffect(() => {
    if (!mapRef.current) return
    import('maplibre-gl').then(async ({ default: maplibregl }) => {
      if (mapInstance.current) {
        mapInstance.current.remove()
        mapInstance.current = null
      }

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
        zoom: isTaluks ? 10 : 12,
      })

      map.on('load', async () => {
        if (isTaluks) {
          try {
            const taluksGeo = await api.getTalukBoundaries()
            map.addSource('taluk-bounds', { type: 'geojson', data: taluksGeo })
            map.addLayer({
              id: 'taluk-lines',
              type: 'line',
              source: 'taluk-bounds',
              paint: { 'line-color': '#1F3A5F', 'line-width': 2 },
            })
          } catch (e) {
            console.error(e)
          }
        } else {
          map.addSource('pilot-bounds', {
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
            id: 'pilot-lines',
            type: 'line',
            source: 'pilot-bounds',
            paint: { 'line-color': '#1F3A5F', 'line-width': 2 },
          })
        }

        map.fitBounds([
          [bbox.lon_min, bbox.lat_min],
          [bbox.lon_max, bbox.lat_max]
        ], { padding: 40 })
      })

      map.on('mousemove', (e: any) => {
        setCoords({ lon: parseFloat(e.lngLat.lng.toFixed(5)), lat: parseFloat(e.lngLat.lat.toFixed(5)) })
      })

      map.on('click', (e: any) => {
        const pt: [number, number] = [e.lngLat.lng, e.lngLat.lat]
        setDrawingPoints(prev => [...prev, pt])
      })

      mapInstance.current = map
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

  const [selectedSplit, setSelectedSplit] = useState<'train' | 'val'>('train')

  const handleAddPolygon = () => {
    if (drawingPoints.length < 3) {
      alert('Click at least 3 points on the map to define a polygon.')
      return
    }
    const closed = [...drawingPoints, drawingPoints[0]]
    const newFeature = {
      type: 'Feature',
      geometry: {
        type: 'Polygon',
        coordinates: [closed],
      },
      properties: {
        class: selectedClass,
        name: CLASS_OPTIONS.find(c => c.id === selectedClass)?.name || selectedClass,
        split: selectedSplit,
      },
    }
    setLabels(prev => [...prev, newFeature])
    setDrawingPoints([])
  }

  const handleSave = async () => {
    setSaving(true)
    try {
      await api.saveLabels({ type: 'FeatureCollection', features: labels })
      alert('Labels saved successfully!')
    } catch (e) {
      alert('Error saving labels: ' + e)
    } finally {
      setSaving(false)
    }
  }

  const handleDelete = (index: number) => {
    setLabels(prev => prev.filter((_, i) => i !== index))
  }

  return (
    <div style={{ display: 'flex', height: '100%', overflow: 'hidden' }}>
      <div style={{ flex: '0 0 60%', position: 'relative' }}>
        <div className="map-wrap" ref={mapRef} style={{ height: '100%' }}>
          <BasemapSwitcher basemap={basemap} onChange={setBasemap} />

          <div className="coord-readout">
            {coords.lon.toFixed(5)}°E &nbsp; {coords.lat.toFixed(5)}°N
          </div>
          {drawingPoints.length > 0 && (
            <div style={{
              position: 'absolute', top: 12, left: 12, background: 'rgba(19,35,58,0.9)',
              color: '#FFF', padding: '6px 12px', borderRadius: 4, zIndex: 10, fontSize: 12
            }}>
              Drawing [{selectedSplit.toUpperCase()}]: {drawingPoints.length} vertices. Click "Finish Polygon" when done.
            </div>
          )}
        </div>
      </div>

      <div className="right-panel" style={{ flex: 1, width: 'auto' }}>
        <div className="panel-section">
          <div className="page-title" style={{ fontSize: 16 }}>Training Data & Seeds</div>
          <div className="page-subtitle">Bootstrapped phenology seeds & manual validation polygons</div>
        </div>

        <div className="panel-scroll">
          <div className="section-label">Add Manual Polygon</div>
          <div style={{ display: 'flex', gap: 6, marginBottom: 8 }}>
            <select
              value={selectedClass}
              onChange={e => setSelectedClass(e.target.value)}
              style={{
                flex: 1.2, padding: '6px 8px', borderRadius: 4, border: '1px solid var(--border)',
                fontFamily: 'var(--font-ui)', fontSize: 12, background: 'var(--panel)', color: 'var(--text-primary)'
              }}
            >
              {CLASS_OPTIONS.map(c => (
                <option key={c.id} value={c.id}>{c.name}</option>
              ))}
            </select>

            <select
              value={selectedSplit}
              onChange={e => setSelectedSplit(e.target.value as 'train' | 'val')}
              style={{
                flex: 0.9, padding: '6px 8px', borderRadius: 4, border: '1px solid var(--border)',
                fontFamily: 'var(--font-ui)', fontSize: 12, background: 'var(--panel)', color: 'var(--text-primary)',
                fontWeight: 500,
              }}
            >
              <option value="train">Train (3x wt)</option>
              <option value="val">Validation</option>
            </select>

            <button className="btn btn-secondary btn-sm" onClick={handleAddPolygon}>
              <Plus size={14} /> Finish
            </button>
          </div>

          <div style={{ display: 'flex', gap: 8, marginBottom: 16 }}>
            <button className="btn btn-primary btn-sm" onClick={handleSave} disabled={saving}>
              <Save size={14} /> Save Labels
            </button>
            {drawingPoints.length > 0 && (
              <button className="btn btn-secondary btn-sm" onClick={() => setDrawingPoints([])}>
                Cancel Draw
              </button>
            )}
          </div>

          <div className="sep" />

          <div className="section-label">Manual Polygons ({labels.length})</div>
          {labels.length === 0 ? (
            <div className="empty-state" style={{ padding: 12 }}>
              <div style={{ fontSize: 12 }}>No manual labels added yet. Click on the map to draft polygons.</div>
            </div>
          ) : (
            <table className="data-table" style={{ marginBottom: 16 }}>
              <thead>
                <tr>
                  <th>#</th>
                  <th>Class</th>
                  <th>Split</th>
                  <th>Points</th>
                  <th>Action</th>
                </tr>
              </thead>
              <tbody>
                {labels.map((l, i) => (
                  <tr key={i}>
                    <td className="mono">{i + 1}</td>
                    <td>{l.properties?.name || l.properties?.class}</td>
                    <td>
                      <span style={{
                        fontSize: 10, padding: '2px 6px', borderRadius: 2,
                        background: l.properties?.split === 'val' ? '#D0E2FF' : '#E0E8D9',
                        color: l.properties?.split === 'val' ? '#0043CE' : '#2E7D4F',
                        fontFamily: 'var(--font-mono)', fontWeight: 600,
                      }}>
                        {(l.properties?.split || 'train').toUpperCase()}
                      </span>
                    </td>
                    <td className="mono num">{l.geometry?.coordinates?.[0]?.length || 0}</td>
                    <td>
                      <button
                        onClick={() => handleDelete(i)}
                        style={{ border: 'none', background: 'transparent', cursor: 'pointer', color: 'var(--status-failed)' }}
                      >
                        <Trash2 size={13} />
                      </button>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          )}

          <div className="sep" />

          {/* Seed vs Manual counts per class */}
          <div className="section-label">Training Labels Breakdown (Seed vs Manual)</div>
          {metricsData?.counts_per_class ? (
            <table className="data-table" style={{ marginBottom: 16 }}>
              <thead>
                <tr>
                  <th>Class</th>
                  <th style={{ textAlign: 'right' }}>Phenology Seeds</th>
                  <th style={{ textAlign: 'right' }}>Manual (3x wt)</th>
                  <th style={{ textAlign: 'right' }}>Total</th>
                </tr>
              </thead>
              <tbody>
                {Object.entries(metricsData.counts_per_class).map(([cname, cnts]) => (
                  <tr key={cname}>
                    <td>
                      <span style={{
                        display: 'inline-block', width: 8, height: 8, borderRadius: '50%',
                        background: CLASS_OPTIONS.find(c => c.name === cname)?.color || '#888',
                        marginRight: 6,
                      }} />
                      {cname}
                    </td>
                    <td className="mono num">{cnts.seed.toLocaleString()}</td>
                    <td className="mono num" style={{ color: cnts.manual > 0 ? 'var(--status-done)' : 'inherit', fontWeight: cnts.manual > 0 ? 600 : 400 }}>
                      {cnts.manual.toLocaleString()}
                    </td>
                    <td className="mono num font-semibold">{cnts.total.toLocaleString()}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          ) : (
            <div style={{ fontSize: 11, color: 'var(--text-secondary)', marginBottom: 16 }}>
              Run classification to view the active seed vs manual training count breakdown.
            </div>
          )}

          <div className="sep" />

          <div className="section-label">Phenological Seed Rules</div>
          <div style={{ fontSize: 11, color: 'var(--text-secondary)', lineHeight: 1.6 }}>
            <div><strong>Paddy:</strong> NDVI peak ≥ 0.6 + trough ≤ 0.3 before peak + flood signal (VH min &lt; −19 dB or LSWI &gt; 0)</div>
            <div><strong>Banana:</strong> NDVI &gt; 0.50 (≥7 of rolling 12 mo) + NDVI std &lt; 0.15 + VH min &gt; −18 dB</div>
            <div><strong>Water:</strong> NDWI &gt; 0.1 in ≥ 4 months</div>
            <div><strong>Built-up:</strong> NDVI max &lt; 0.3 + NDWI &lt; 0</div>
            <div><strong>Fallow:</strong> NDVI max &lt; 0.35</div>
          </div>
        </div>
      </div>
    </div>
  )
}
