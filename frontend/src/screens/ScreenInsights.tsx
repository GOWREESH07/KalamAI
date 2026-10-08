import React, { useState, useEffect, useRef } from 'react'
import { api, InsightSummaryItem, InsightDetail, StudyAreaType } from '../api'
import BasemapSwitcher from '../components/BasemapSwitcher'
import { BasemapType, BASEMAP_SOURCES, getBasemapLayers, updateBasemapVisibility } from '../utils/basemaps'
import { Download, Layers, CheckCircle2, AlertCircle, AlertTriangle, ChevronDown, ChevronUp, FileText, MapPin, RefreshCw, Calendar, Droplet, Activity, PieChart, ShieldAlert } from 'lucide-react'

interface Props {
  studyArea: StudyAreaType
}

export default function ScreenInsights({ studyArea }: Props) {
  const mapRef = useRef<HTMLDivElement>(null)
  const mapInstance = useRef<any>(null)

  const [basemap, setBasemap] = useState<BasemapType>('esri-world')
  const [insights, setInsights] = useState<InsightSummaryItem[]>([])
  const [selectedId, setSelectedId] = useState<string>('cropping_intensity')
  const [detail, setDetail] = useState<InsightDetail | null>(null)
  const [loading, setLoading] = useState<boolean>(false)
  const [computingAll, setComputingAll] = useState<boolean>(false)
  const [opacity, setOpacity] = useState<number>(0.65)
  const [coords, setCoords] = useState<{ lon: number; lat: number }>({ lon: 77.5, lat: 8.7 })
  const [showMethod, setShowMethod] = useState<boolean>(true)
  const [showLimitations, setShowLimitations] = useState<boolean>(false)

  const AOI = studyArea === 'taluks'
    ? { lon_min: 77.1697, lat_min: 8.5339, lon_max: 77.6619, lat_max: 8.8820 }
    : { lon_min: 77.47, lat_min: 8.680, lon_max: 77.53, lat_max: 8.725 }

  // Load insights summary list
  useEffect(() => {
    api.getInsights(studyArea).then(res => {
      setInsights(res)
      if (res.length > 0 && !res.find(i => i.id === selectedId)) {
        setSelectedId(res[0].id)
      }
    }).catch(err => console.error('Error fetching insights list:', err))
  }, [studyArea])

  // Load active insight detail
  useEffect(() => {
    if (!selectedId) return
    setLoading(true)
    api.getInsightDetail(selectedId, studyArea).then(res => {
      setDetail(res)
      setLoading(false)
    }).catch(err => {
      console.error('Error fetching insight detail:', err)
      setLoading(false)
    })
  }, [selectedId, studyArea])

  // Initialize MapLibre
  useEffect(() => {
    if (!mapRef.current) return
    let map: any = null

    import('maplibre-gl').then(({ default: maplibregl }) => {
      if (mapInstance.current) {
        mapInstance.current.remove()
        mapInstance.current = null
      }

      const centerLon = (AOI.lon_min + AOI.lon_max) / 2
      const centerLat = (AOI.lat_min + AOI.lat_max) / 2
      const zoom = studyArea === 'taluks' ? 10.5 : 12.5

      map = new maplibregl.Map({
        container: mapRef.current!,
        attributionControl: false, // Custom compact attribution
        style: {
          version: 8,
          sources: BASEMAP_SOURCES,
          layers: getBasemapLayers(basemap),
        },
        center: [centerLon, centerLat],
        zoom,
        maxZoom: 19,
      })

      map.on('load', () => {
        const pngUrl = api.getInsightPngUrl(selectedId, studyArea)
        map.addSource('insight-raster', {
          type: 'image',
          url: pngUrl,
          coordinates: [
            [AOI.lon_min, AOI.lat_max],
            [AOI.lon_max, AOI.lat_max],
            [AOI.lon_max, AOI.lat_min],
            [AOI.lon_min, AOI.lat_min],
          ],
        })

        map.addLayer({
          id: 'insight-raster-layer',
          type: 'raster',
          source: 'insight-raster',
          paint: {
            'raster-opacity': opacity,
            'raster-resampling': 'nearest',
          },
        })
      })

      map.on('mousemove', (e: any) => {
        setCoords({
          lon: parseFloat(e.lngLat.lng.toFixed(5)),
          lat: parseFloat(e.lngLat.lat.toFixed(5)),
        })
      })

      mapInstance.current = map
    })

    return () => {
      if (mapInstance.current) {
        mapInstance.current.remove()
        mapInstance.current = null
      }
    }
  }, [selectedId, studyArea])

  // Handle Basemap Switch
  useEffect(() => {
    if (!mapInstance.current) return
    updateBasemapVisibility(mapInstance.current, basemap)
  }, [basemap])

  // Update opacity
  useEffect(() => {
    if (!mapInstance.current) return
    const map = mapInstance.current
    if (map.getLayer('insight-raster-layer')) {
      map.setPaintProperty('insight-raster-layer', 'raster-opacity', opacity)
    }
  }, [opacity])

  const handleComputeAll = async () => {
    setComputingAll(true)
    try {
      await api.computeAllInsights(studyArea)
      const res = await api.getInsights(studyArea)
      setInsights(res)
      if (selectedId) {
        const d = await api.getInsightDetail(selectedId, studyArea)
        setDetail(d)
      }
    } catch (err) {
      console.error('Failed computing all insights:', err)
    } finally {
      setComputingAll(false)
    }
  }

  // Calculate sum of areas to assert conservation of mass/area
  const areaSum = detail?.chart_data?.reduce((acc: number, item: any) => {
    return acc + (item.area_ha || 0)
  }, 0) || 0

  const totalExtentHa = detail?.total_cropland_ha || detail?.total_paddy_ha || detail?.total_area_ha || detail?.total_evaluated_ha || 0
  const areaMismatch = totalExtentHa > 0 && Math.abs(areaSum - totalExtentHa) / totalExtentHa > 0.005

  return (
    <div style={{ display: 'flex', height: '100%', overflow: 'hidden' }}>
      {/* LEFT COLUMN: INSIGHTS LIST */}
      <div style={{
        flex: '0 0 280px',
        borderRight: '1px solid var(--border)',
        background: 'var(--panel)',
        display: 'flex',
        flexDirection: 'column',
        overflow: 'hidden',
      }}>
        <div style={{ padding: '12px 14px', borderBottom: '1px solid var(--border)', background: 'var(--bg)' }}>
          <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: 4 }}>
            <div style={{ fontSize: 13, fontWeight: 700, color: 'var(--accent)', textTransform: 'uppercase', letterSpacing: 0.5 }}>
              Insights Modules
            </div>
            <button
              onClick={handleComputeAll}
              disabled={computingAll}
              className="btn btn-secondary btn-sm"
              style={{ fontSize: 11, padding: '3px 8px', gap: 4 }}
              title="Re-compute all 9 insights and regenerate Atlas"
            >
              <RefreshCw size={11} className={computingAll ? 'spin' : ''} />
              {computingAll ? 'Running...' : 'Run All'}
            </button>
          </div>
          <div style={{ fontSize: 11, color: 'var(--text-secondary)' }}>
            Analytical derivatives for {studyArea === 'taluks' ? 'Ambasamudram & Cheranmahadevi' : 'Cheranmahadevi Pilot'}
          </div>
        </div>

        <div style={{ flex: 1, overflowY: 'auto', padding: 8, display: 'flex', flexDirection: 'column', gap: 6 }}>
          {insights.map((item) => {
            const isSelected = item.id === selectedId
            const statusColor = item.status === 'ready'
              ? 'var(--status-done, #2A9D8F)'
              : item.status === 'partial'
                ? '#E9C46A'
                : 'var(--status-failed, #B4554B)'

            return (
              <div
                key={item.id}
                onClick={() => setSelectedId(item.id)}
                style={{
                  padding: '10px 12px',
                  borderRadius: 4,
                  border: isSelected ? '1px solid var(--accent)' : '1px solid var(--border)',
                  background: isSelected ? 'var(--accent-subtle)' : 'var(--bg)',
                  cursor: 'pointer',
                  transition: 'all 0.15s ease',
                }}
              >
                <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: 3 }}>
                  <span style={{
                    fontSize: 12,
                    fontWeight: isSelected ? 700 : 600,
                    color: isSelected ? 'var(--accent)' : 'var(--text-primary)',
                  }}>
                    {item.title}
                  </span>
                  <span
                    style={{
                      width: 8,
                      height: 8,
                      borderRadius: '50%',
                      background: statusColor,
                      display: 'inline-block',
                      boxShadow: isSelected ? `0 0 6px ${statusColor}` : 'none',
                    }}
                    title={`Status: ${item.status || 'ready'}`}
                  />
                </div>
                <div style={{ fontSize: 11, color: 'var(--text-secondary)', lineHeight: 1.35 }}>
                  {item.subtitle}
                </div>
              </div>
            )
          })}
        </div>

        {/* ATLAS PDF DOWNLOAD BANNER */}
        <div style={{ padding: '10px 12px', borderTop: '1px solid var(--border)', background: 'var(--bg)' }}>
          <a
            href={api.getAtlasPdfUrl(studyArea)}
            target="_blank"
            rel="noreferrer"
            className="btn btn-primary"
            style={{ width: '100%', justifyContent: 'center', fontSize: 12, gap: 6, textDecoration: 'none' }}
          >
            <FileText size={14} /> Download Insights Atlas PDF
          </a>
        </div>
      </div>

      {/* CENTRE COLUMN: MAP DISPLAY */}
      <div style={{ flex: '1 1 45%', position: 'relative', overflow: 'hidden' }}>
        <div className="map-wrap" ref={mapRef} style={{ height: '100%', width: '100%' }}>
          <BasemapSwitcher basemap={basemap} onChange={setBasemap} />

          <div className="coord-readout">
            {coords.lon.toFixed(5)}°E &nbsp; {coords.lat.toFixed(5)}°N
          </div>

          {/* Opacity Control Overlay */}
          <div style={{
            position: 'absolute',
            bottom: 32,
            right: 12,
            background: 'var(--panel)',
            border: '1px solid var(--border)',
            borderRadius: 4,
            padding: '6px 10px',
            display: 'flex',
            alignItems: 'center',
            gap: 8,
            zIndex: 10,
            boxShadow: '0 2px 8px rgba(0,0,0,0.1)',
          }}>
            <span style={{ fontSize: 11, fontWeight: 600, color: 'var(--text-secondary)' }}>Opacity</span>
            <input
              type="range"
              min={0.1}
              max={1.0}
              step={0.05}
              value={opacity}
              onChange={e => setOpacity(parseFloat(e.target.value))}
              style={{ width: 75, cursor: 'pointer' }}
            />
            <span className="mono" style={{ fontSize: 11, width: 30 }}>{Math.round(opacity * 100)}%</span>
          </div>

          {/* Compact Bottom-Right Attribution */}
          <div style={{
            position: 'absolute',
            bottom: 6,
            right: 12,
            fontSize: 9,
            color: 'var(--text-secondary)',
            background: 'rgba(255,255,255,0.7)',
            padding: '1px 4px',
            borderRadius: 2,
            zIndex: 5,
          }}>
            © Esri, Maxar, Earthstar | Copernicus Sentinel 2025–26 | Kalam
          </div>

          {/* Legend Overlay */}
          {detail?.legend && detail.legend.length > 0 && (
            <div style={{
              position: 'absolute',
              bottom: 12,
              left: 12,
              background: 'var(--panel)',
              border: '1px solid var(--border)',
              borderRadius: 4,
              padding: '8px 12px',
              zIndex: 10,
              maxWidth: 280,
              boxShadow: '0 2px 8px rgba(0,0,0,0.1)',
            }}>
              <div style={{ fontSize: 11, fontWeight: 700, color: 'var(--text-primary)', marginBottom: 6, textTransform: 'uppercase', letterSpacing: 0.4 }}>
                Map Classification Legend
              </div>
              <div style={{ display: 'flex', flexDirection: 'column', gap: 4 }}>
                {detail.legend.map((l, i) => (
                  <div key={i} style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                    <span style={{
                      width: 12,
                      height: 12,
                      borderRadius: 2,
                      background: l.color,
                      border: '1px solid rgba(0,0,0,0.15)',
                      flexShrink: 0,
                    }} />
                    <span style={{ fontSize: 11, color: 'var(--text-primary)' }}>{l.label}</span>
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>
      </div>

      {/* RIGHT COLUMN: ANALYTICS, FINDINGS & EXPORTS */}
      <div className="right-panel" style={{ flex: '0 0 390px', width: 390, display: 'flex', flexDirection: 'column' }}>
        <div className="panel-section" style={{ borderBottom: '1px solid var(--border)', paddingBottom: 10 }}>
          <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
            <div className="page-title" style={{ fontSize: 16 }}>{detail?.title || 'Insight Analytics'}</div>
          </div>
          <div className="page-subtitle" style={{ marginBottom: 6 }}>{detail?.subtitle || 'Agricultural pattern analytics'}</div>

          {/* EXTENT AND MODEL BADGE */}
          {detail?.extent_label && (
            <div style={{
              display: 'inline-flex',
              alignItems: 'center',
              gap: 5,
              padding: '3px 8px',
              borderRadius: 3,
              background: 'var(--bg)',
              border: '1px solid var(--border)',
              fontSize: 10,
              fontWeight: 600,
              color: 'var(--text-secondary)',
            }}>
              <span style={{ width: 5, height: 5, borderRadius: '50%', background: 'var(--accent)' }} />
              {detail.extent_label}
            </div>
          )}
        </div>

        <div className="panel-scroll" style={{ flex: 1, padding: 14 }}>
          {loading ? (
            <div className="empty-state" style={{ padding: 24 }}>
              <div>Loading insight analytics...</div>
            </div>
          ) : detail ? (
            <div>
              {/* AREA SUM MISMATCH ALERT */}
              {areaMismatch && (
                <div style={{
                  background: '#FFF3CD',
                  borderLeft: '3px solid #E9C46A',
                  padding: '8px 10px',
                  borderRadius: '0 4px 4px 0',
                  marginBottom: 12,
                  fontSize: 11,
                  color: '#856404',
                  display: 'flex',
                  alignItems: 'center',
                  gap: 6,
                }}>
                  <AlertTriangle size={14} style={{ flexShrink: 0 }} />
                  <span>Area sum warning: component sum ({areaSum.toFixed(1)} ha) deviates &gt;0.5% from total extent ({totalExtentHa.toFixed(1)} ha).</span>
                </div>
              )}

              {/* KEY FINDING STATEMENT BANNER */}
              <div style={{
                background: 'var(--accent-subtle)',
                borderLeft: '3px solid var(--accent)',
                padding: '10px 12px',
                borderRadius: '0 4px 4px 0',
                marginBottom: 14,
              }}>
                <div style={{ fontSize: 10, fontWeight: 700, textTransform: 'uppercase', color: 'var(--accent)', marginBottom: 3, letterSpacing: 0.5 }}>
                  Key Finding
                </div>
                <div style={{ fontSize: 12, lineHeight: 1.5, color: 'var(--text-primary)', fontWeight: 500 }}>
                  {detail.key_finding}
                </div>
              </div>

              {/* STAT TILES PER INSIGHT */}
              {detail.id === 'cropping_intensity' && (
                <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 8, marginBottom: 14 }}>
                  <div className="metric-tile" style={{ padding: '8px 10px' }}>
                    <div className="metric-label">Double-Cropped</div>
                    <div className="metric-value" style={{ fontSize: 20, color: '#457B9D' }}>{detail.double_cropped_pct}%</div>
                  </div>
                  <div className="metric-tile" style={{ padding: '8px 10px' }}>
                    <div className="metric-label">Perennial Banana</div>
                    <div className="metric-value" style={{ fontSize: 20, color: '#2E6B3F' }}>{detail.perennial_pct}%</div>
                  </div>
                </div>
              )}

              {detail.id === 'transplanting_calendar' && (
                <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr 1fr', gap: 6, marginBottom: 14 }}>
                  <div className="metric-tile" style={{ padding: '6px 8px' }}>
                    <div className="metric-label">Kar Peak</div>
                    <div className="metric-value" style={{ fontSize: 16, color: '#EE9B00' }}>{detail.kar_median_month}</div>
                  </div>
                  <div className="metric-tile" style={{ padding: '6px 8px' }}>
                    <div className="metric-label">Pishanam Peak</div>
                    <div className="metric-value" style={{ fontSize: 16, color: '#005F73' }}>{detail.pishanam_median_month}</div>
                  </div>
                  <div className="metric-tile" style={{ padding: '6px 8px' }}>
                    <div className="metric-label">Pishanam Delayed</div>
                    <div className="metric-value" style={{ fontSize: 16, color: '#B4554B' }}>{detail.pishanam_delayed_pct}%</div>
                  </div>
                </div>
              )}

              {detail.id === 'water_demand' && (
                <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 8, marginBottom: 14 }}>
                  <div className="metric-tile" style={{ padding: '8px 10px' }}>
                    <div className="metric-label">Total Water Budget</div>
                    <div className="metric-value" style={{ fontSize: 20, color: '#1F3A5F' }}>{detail.total_water_mcm} <span style={{ fontSize: 11 }}>MCM</span></div>
                  </div>
                  <div className="metric-tile" style={{ padding: '8px 10px' }}>
                    <div className="metric-label">Paddy Share</div>
                    <div className="metric-value" style={{ fontSize: 20, color: '#8DB33F' }}>{detail.paddy_water_share_pct}%</div>
                  </div>
                </div>
              )}

              {detail.id === 'field_size' && (
                <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 8, marginBottom: 14 }}>
                  <div className="metric-tile" style={{ padding: '8px 10px' }}>
                    <div className="metric-label">Smallholders (&lt;1 ha)</div>
                    <div className="metric-value" style={{ fontSize: 20, color: '#2A9D8F' }}>{detail.smallholder_count_pct}%</div>
                  </div>
                  <div className="metric-tile" style={{ padding: '8px 10px' }}>
                    <div className="metric-label">Mean Parcel Size</div>
                    <div className="metric-value" style={{ fontSize: 20, color: '#1F3A5F' }}>{detail.average_parcel_ha} <span style={{ fontSize: 11 }}>ha</span></div>
                  </div>
                </div>
              )}

              {detail.id === 'crop_health' && (
                <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 8, marginBottom: 14 }}>
                  <div className="metric-tile" style={{ padding: '8px 10px' }}>
                    <div className="metric-label">Healthy Canopy</div>
                    <div className="metric-value" style={{ fontSize: 20, color: '#2E6B3F' }}>{detail.healthy_share_pct}%</div>
                  </div>
                  <div className="metric-tile" style={{ padding: '8px 10px' }}>
                    <div className="metric-label">Moisture / Vigour Stress</div>
                    <div className="metric-value" style={{ fontSize: 20, color: '#E76F51' }}>{detail.stressed_share_pct}%</div>
                  </div>
                </div>
              )}

              {detail.id === 'verification_priority' && (
                <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 8, marginBottom: 14 }}>
                  <div className="metric-tile" style={{ padding: '8px 10px' }}>
                    <div className="metric-label">Critical Audit Parcels</div>
                    <div className="metric-value" style={{ fontSize: 20, color: '#B4554B' }}>{detail.critical_parcels}</div>
                  </div>
                  <div className="metric-tile" style={{ padding: '8px 10px' }}>
                    <div className="metric-label">High Priority</div>
                    <div className="metric-value" style={{ fontSize: 20, color: '#E76F51' }}>{detail.high_priority_parcels}</div>
                  </div>
                </div>
              )}

              {/* DISTRIBUTION CHART */}
              {detail.chart_data && detail.chart_data.length > 0 && (
                <div style={{ marginBottom: 14 }}>
                  <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: 6 }}>
                    <div className="section-label" style={{ margin: 0 }}>Distribution Breakdown</div>
                    {detail.denominator_label && (
                      <span style={{ fontSize: 10, color: 'var(--text-secondary)' }}>
                        Denominator: {detail.denominator_label}
                      </span>
                    )}
                  </div>
                  <div style={{ display: 'flex', flexDirection: 'column', gap: 6 }}>
                    {detail.chart_data.map((c: any, i: number) => {
                      const label = c.category || c.component || c.distance || c.village || `Item ${i + 1}`
                      const pct = c.pct !== undefined ? c.pct : c.count_pct !== undefined ? c.count_pct : (c.paddy_pct || 0)
                      const barColor = c.color || '#457B9D'
                      const valStr = c.area_ha !== undefined
                        ? `${c.area_ha} ha (${pct}%)`
                        : c.water_mcm !== undefined
                          ? `${c.water_mcm} MCM (${pct}%)`
                          : `${pct}%`

                      return (
                        <div key={i} style={{ fontSize: 11 }}>
                          <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 2 }}>
                            <span style={{ color: 'var(--text-primary)', fontWeight: 500 }}>{label}</span>
                            <span className="mono" style={{ color: 'var(--text-secondary)' }}>{valStr}</span>
                          </div>
                          <div style={{ width: '100%', height: 6, background: 'var(--border)', borderRadius: 3, overflow: 'hidden' }}>
                            <div style={{ width: `${Math.min(100, Math.max(0, pct))}%`, height: '100%', background: barColor, borderRadius: 3 }} />
                          </div>
                        </div>
                      )
                    })}
                  </div>
                </div>
              )}

              {/* TOP 50 VERIFICATION TABLE */}
              {detail.id === 'verification_priority' && detail.top50 && detail.top50.length > 0 && (
                <div style={{ marginBottom: 14 }}>
                  <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: 6 }}>
                    <div className="section-label" style={{ margin: 0 }}>Top Field Audit Targets</div>
                    <span className="mono" style={{ fontSize: 10, color: 'var(--text-secondary)' }}>{detail.top50.length} flagged</span>
                  </div>
                  <div style={{ maxHeight: 180, overflowY: 'auto', border: '1px solid var(--border)', borderRadius: 4, marginTop: 6 }}>
                    <table className="data-table" style={{ margin: 0, fontSize: 11 }}>
                      <thead>
                        <tr>
                          <th>#ID</th>
                          <th>Class</th>
                          <th>Score</th>
                          <th>GPS Centroid</th>
                        </tr>
                      </thead>
                      <tbody>
                        {detail.top50.slice(0, 15).map((r: any, i: number) => (
                          <tr key={i}>
                            <td className="mono">#{r.parcel_id}</td>
                            <td>{r.predicted_class}</td>
                            <td className="mono num" style={{ color: r.priority_score > 0.4 ? '#B4554B' : '#E76F51' }}>
                              {r.priority_score}
                            </td>
                            <td className="mono" style={{ fontSize: 10 }}>{r.lon.toFixed(4)}, {r.lat.toFixed(4)}</td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                </div>
              )}

              {/* VILLAGES TABLE */}
              {detail.id === 'village_stats' && detail.villages && detail.villages.length > 0 && (
                <div style={{ marginBottom: 14 }}>
                  <div className="section-label" style={{ marginBottom: 6 }}>Settlement Cropland Summary</div>
                  <div style={{ maxHeight: 180, overflowY: 'auto', border: '1px solid var(--border)', borderRadius: 4 }}>
                    <table className="data-table" style={{ margin: 0, fontSize: 11 }}>
                      <thead>
                        <tr>
                          <th>Village</th>
                          <th className="num">Paddy (ha)</th>
                          <th className="num">Banana (ha)</th>
                          <th>Dominant Crop</th>
                        </tr>
                      </thead>
                      <tbody>
                        {detail.villages.slice(0, 10).map((v: any, i: number) => (
                          <tr key={i}>
                            <td>{v.village}</td>
                            <td className="num mono">{v.paddy_ha}</td>
                            <td className="num mono">{v.banana_ha}</td>
                            <td style={{ fontSize: 10 }}>{v.dominant_type}</td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                </div>
              )}

              <div className="sep" />

              {/* COLLAPSIBLE METHODOLOGY */}
              <div style={{ marginBottom: 8 }}>
                <div
                  onClick={() => setShowMethod(!showMethod)}
                  style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', cursor: 'pointer', padding: '4px 0' }}
                >
                  <span className="section-label" style={{ margin: 0 }}>Scientific Methodology</span>
                  {showMethod ? <ChevronUp size={14} /> : <ChevronDown size={14} />}
                </div>
                {showMethod && (
                  <div style={{ fontSize: 11, color: 'var(--text-secondary)', lineHeight: 1.5, marginTop: 4 }}>
                    {detail.method}
                  </div>
                )}
              </div>

              {/* COLLAPSIBLE LIMITATIONS */}
              <div style={{ marginBottom: 14 }}>
                <div
                  onClick={() => setShowLimitations(!showLimitations)}
                  style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', cursor: 'pointer', padding: '4px 0' }}
                >
                  <span className="section-label" style={{ margin: 0 }}>Assumptions & Limitations</span>
                  {showLimitations ? <ChevronUp size={14} /> : <ChevronDown size={14} />}
                </div>
                {showLimitations && (
                  <div style={{ fontSize: 11, color: 'var(--text-secondary)', lineHeight: 1.5, marginTop: 4 }}>
                    {detail.limitations}
                  </div>
                )}
              </div>

              <div className="sep" />

              {/* EXPORT BUTTONS */}
              <div className="section-label" style={{ marginBottom: 6 }}>Export Datasets</div>
              <div style={{ display: 'flex', gap: 6, flexWrap: 'wrap' }}>
                <a
                  href={api.getInsightPngUrl(detail.id, studyArea)}
                  download={`${detail.id}_map.png`}
                  className="btn btn-secondary btn-sm"
                  style={{ textDecoration: 'none', gap: 4 }}
                >
                  <Download size={12} /> Map PNG
                </a>
                <a
                  href={api.getInsightCsvUrl(detail.id, studyArea)}
                  download={`${detail.id}_stats.csv`}
                  className="btn btn-secondary btn-sm"
                  style={{ textDecoration: 'none', gap: 4 }}
                >
                  <Download size={12} /> CSV Stats
                </a>
                {detail.id === 'verification_priority' && (
                  <a
                    href={api.getInsightKmlUrl(detail.id, studyArea)}
                    download="field_audit_priority.kml"
                    className="btn btn-secondary btn-sm"
                    style={{ textDecoration: 'none', gap: 4 }}
                  >
                    <MapPin size={12} /> KML (GPS)
                  </a>
                )}
              </div>
            </div>
          ) : (
            <div className="empty-state" style={{ padding: 24 }}>
              <div>Select an insight card on the left to inspect its analytics.</div>
            </div>
          )}
        </div>
      </div>
    </div>
  )
}
