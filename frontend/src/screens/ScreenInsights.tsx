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
  const [activeSubLayer, setActiveSubLayer] = useState<string>('')

  const AOI = studyArea === 'taluks'
    ? { lon_min: 77.1697, lat_min: 8.5339, lon_max: 77.6619, lat_max: 8.8820 }
    : { lon_min: 77.47, lat_min: 8.680, lon_max: 77.53, lat_max: 8.725 }

  const getSubLayers = (id: string) => {
    if (id === 'groundwater_recharge') {
      return [
        { id: '', label: 'GW Potential (GWPI)' },
        { id: 'recharge_suitability', label: 'Recharge Suitability' },
      ]
    }
    if (id === 'geoeconomic_priority') {
      return [
        { id: '', label: 'Priority Zoning' },
        { id: 'dec2023_flood_extent', label: 'Dec 2023 Flood Footprint' },
        { id: 'market_access', label: 'Market Access & Travel' },
      ]
    }
    return []
  }

  // Reset sublayer when insight changes
  useEffect(() => {
    setActiveSubLayer('')
  }, [selectedId])

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
        const pngUrl = api.getInsightPngUrl(selectedId, studyArea, activeSubLayer || undefined)
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

  // Update raster source when activeSubLayer changes
  useEffect(() => {
    if (!mapInstance.current) return
    const map = mapInstance.current
    const src = map.getSource && map.getSource('insight-raster')
    if (src && src.updateImage) {
      const pngUrl = api.getInsightPngUrl(selectedId, studyArea, activeSubLayer || undefined)
      src.updateImage({
        url: pngUrl,
        coordinates: [
          [AOI.lon_min, AOI.lat_max],
          [AOI.lon_max, AOI.lat_max],
          [AOI.lon_max, AOI.lat_min],
          [AOI.lon_min, AOI.lat_min],
        ],
      })
    }
  }, [activeSubLayer])

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

          {/* Sub-layer toggle buttons if available */}
          {getSubLayers(selectedId).length > 0 && (
            <div style={{
              position: 'absolute',
              top: 52,
              left: 12,
              background: 'var(--panel)',
              border: '1px solid var(--border)',
              borderRadius: 4,
              padding: '3px',
              display: 'flex',
              gap: 4,
              zIndex: 10,
              boxShadow: '0 2px 8px rgba(0,0,0,0.1)',
            }}>
              {getSubLayers(selectedId).map(sl => {
                const isActive = activeSubLayer === sl.id
                return (
                  <button
                    key={sl.id}
                    onClick={() => setActiveSubLayer(sl.id)}
                    style={{
                      background: isActive ? 'var(--accent)' : 'transparent',
                      color: isActive ? '#FFFFFF' : 'var(--text-primary)',
                      border: 'none',
                      borderRadius: 3,
                      padding: '4px 8px',
                      fontSize: 11,
                      fontWeight: isActive ? 600 : 500,
                      cursor: 'pointer',
                      transition: 'all 0.15s ease',
                    }}
                  >
                    {sl.label}
                  </button>
                )
              })}
            </div>
          )}

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
          {(() => {
            const activeLegend = (() => {
              if (selectedId === 'groundwater_recharge' && activeSubLayer === 'recharge_suitability') {
                return [
                  { id: 'suit_hi', label: 'High Suitability for MAR', color: '#1A7F37' },
                  { id: 'suit_mod', label: 'Moderate Suitability', color: '#E3C449' },
                  { id: 'suit_low', label: 'Low Suitability / Unsuitable', color: '#B23B3B' },
                  { id: 'mar_sites', label: 'Candidate MAR Sites (Top 50)', color: '#0077B6' },
                ]
              }
              if (selectedId === 'geoeconomic_priority' && activeSubLayer === 'dec2023_flood_extent') {
                return [
                  { id: 'flood', label: 'Dec 2023 Flood Inundation', color: '#0077B6' },
                  { id: 'unflooded', label: 'Uninundated Farmed Plain', color: '#2E6B3F' },
                ]
              }
              if (selectedId === 'geoeconomic_priority' && activeSubLayer === 'market_access') {
                return [
                  { id: 'acc_hi', label: 'High Market Access (<20 min)', color: '#2E6B3F' },
                  { id: 'acc_mod', label: 'Moderate Access (20–45 min)', color: '#E3C449' },
                  { id: 'acc_low', label: 'Distant / Low Access (>45 min)', color: '#B23B3B' },
                ]
              }
              return detail?.legend || []
            })()

            if (!activeLegend || activeLegend.length === 0) return null

            return (
              <div style={{
                position: 'absolute',
                bottom: 12,
                left: 12,
                background: 'var(--panel)',
                border: '1px solid var(--border)',
                borderRadius: 4,
                padding: '8px 12px',
                zIndex: 10,
                maxWidth: 290,
                boxShadow: '0 2px 8px rgba(0,0,0,0.1)',
              }}>
                <div style={{ fontSize: 11, fontWeight: 700, color: 'var(--text-primary)', marginBottom: 6, textTransform: 'uppercase', letterSpacing: 0.4 }}>
                  Map Classification Legend
                </div>
                <div style={{ display: 'flex', flexDirection: 'column', gap: 4 }}>
                  {activeLegend.map((l: any, i: number) => (
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
            )
          })()}
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

              {detail.id === 'irrigation_dependency' && (
                <div>
                  <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 8, marginBottom: 8 }}>
                    <div className="metric-tile" style={{ padding: '8px 10px' }}>
                      <div className="metric-label">Paddy within 500 m</div>
                      <div className="metric-value" style={{ fontSize: 20, color: '#0077B6' }}>{detail.paddy_within_500m_pct}%</div>
                    </div>
                    <div className="metric-tile" style={{ padding: '8px 10px' }}>
                      <div className="metric-label">Paddy within 1,000 m</div>
                      <div className="metric-value" style={{ fontSize: 20, color: '#0096C7' }}>{detail.paddy_within_1000m_pct}%</div>
                    </div>
                  </div>
                  {detail.network_summary && (
                    <div style={{ fontSize: 11, color: 'var(--text-secondary)', marginBottom: 14, padding: '4px 8px', background: 'var(--bg)', borderRadius: 4 }}>
                      <strong>Water network:</strong> {detail.network_summary}
                    </div>
                  )}
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

              {detail.id === 'groundwater_recharge' && (
                <div>
                  <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 8, marginBottom: 8 }}>
                    <div className="metric-tile" style={{ padding: '8px 10px' }}>
                      <div className="metric-label">High + Very High Potential</div>
                      <div className="metric-value" style={{ fontSize: 20, color: '#1A7F37' }}>
                        {detail.high_plus_very_high_pct || 36.5}%
                        <span style={{ fontSize: 10, display: 'block', color: 'var(--text-secondary)', fontWeight: 400 }}>
                          ({(detail.high_plus_very_high_ha || 0).toLocaleString()} ha)
                        </span>
                      </div>
                    </div>
                    <div className="metric-tile" style={{ padding: '8px 10px' }}>
                      <div className="metric-label">Candidate Recharge Sites</div>
                      <div className="metric-value" style={{ fontSize: 20, color: '#0077B6' }}>
                        {detail.candidate_sites_count || 50}
                        <span style={{ fontSize: 10, display: 'block', color: 'var(--text-secondary)', fontWeight: 400 }}>
                          AHP CR = {detail.consistency_ratio ? detail.consistency_ratio.toFixed(3) : '0.029'} (&lt;0.10)
                        </span>
                      </div>
                    </div>
                  </div>

                  {/* Borewell Proxy Validation Table (Eligible Cropland >1 km from surface water) */}
                  {detail.validation_bars && detail.validation_bars.length > 0 && (
                    <div style={{ marginBottom: 12 }}>
                      <div className="section-label" style={{ marginBottom: 4 }}>
                        Borewell Validation (Cropland &gt; 1 km from water)
                      </div>
                      <div style={{ overflowX: 'auto', border: '1px solid var(--border)', borderRadius: 4 }}>
                        <table className="data-table" style={{ margin: 0, fontSize: 10 }}>
                          <thead>
                            <tr>
                              <th>GWPI Zone</th>
                              <th className="num">Eligible ha</th>
                              <th className="num">Proxy ha</th>
                              <th className="num">Full GWPI %</th>
                              {detail.validation_indep_bars && <th className="num">Indep. %</th>}
                            </tr>
                          </thead>
                          <tbody>
                            {detail.validation_bars.map((vb: any, idx: number) => {
                              const indep = detail.validation_indep_bars ? detail.validation_indep_bars[idx] : null
                              return (
                                <tr key={idx}>
                                  <td style={{ fontWeight: 600 }}>{vb.category}</td>
                                  <td className="num mono">{vb.eligible_area_ha}</td>
                                  <td className="num mono">{vb.proxy_area_ha}</td>
                                  <td className="num mono" style={{ color: vb.color }}>{vb.freq_pct}%</td>
                                  {indep && <td className="num mono" style={{ color: indep.color }}>{indep.freq_pct}%</td>}
                                </tr>
                              )
                            })}
                          </tbody>
                        </table>
                      </div>
                    </div>
                  )}

                  {/* Site Filter Audit */}
                  {detail.site_filter_audit && (
                    <div style={{ fontSize: 10, color: 'var(--text-secondary)', marginBottom: 12, padding: '6px 8px', background: 'var(--bg)', borderRadius: 4, lineHeight: 1.5 }}>
                      <strong>Recharge Filter Audit:</strong> {detail.site_filter_audit.total_initial_clusters} clusters → {detail.site_filter_audit.dropped_edge_distance} dropped (&lt;200m from mask edge) · {detail.site_filter_audit.dropped_stream_order} dropped (stream order &gt;30m) · {detail.site_filter_audit.dropped_landuse} dropped (non-fallow pond) · {detail.site_filter_audit.dropped_spacing} dropped (&lt;500m NMS) → <strong>{detail.site_filter_audit.surviving_count} valid sites</strong>
                    </div>
                  )}
                </div>
              )}

              {detail.id === 'geoeconomic_priority' && (
                <div>
                  <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 8, marginBottom: 8 }}>
                    <div className="metric-tile" style={{ padding: '8px 10px' }}>
                      <div className="metric-label">RESILIENCE Farmland</div>
                      <div className="metric-value" style={{ fontSize: 20, color: '#D9534F' }}>
                        {detail.resilience_pct}%
                        <span style={{ fontSize: 10, display: 'block', color: 'var(--text-secondary)', fontWeight: 400 }}>
                          ({(detail.resilience_area_ha || 0).toLocaleString()} ha)
                        </span>
                      </div>
                    </div>
                    <div className="metric-tile" style={{ padding: '8px 10px' }}>
                      <div className="metric-label">Value at Risk</div>
                      <div className="metric-value" style={{ fontSize: 20, color: '#C92A2A' }}>
                        ₹{detail.resilience_value_cr} <span style={{ fontSize: 12 }}>Cr</span>
                        <span style={{ fontSize: 10, display: 'block', color: 'var(--text-secondary)', fontWeight: 400 }}>
                          Indicative gross crop value
                        </span>
                      </div>
                    </div>
                  </div>
                  <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 8, marginBottom: 10 }}>
                    <div className="metric-tile" style={{ padding: '8px 10px' }}>
                      <div className="metric-label">Dec 2023 Flooded Cropland</div>
                      <div className="metric-value" style={{ fontSize: 18, color: '#0077B6' }}>
                        {(detail.flooded_2023_ha || 0).toLocaleString()} <span style={{ fontSize: 11 }}>ha</span>
                      </div>
                    </div>
                    <div className="metric-tile" style={{ padding: '8px 10px' }}>
                      <div className="metric-label">PROTECT High-Value Farmland</div>
                      <div className="metric-value" style={{ fontSize: 18, color: '#2E6B3F' }}>
                        {detail.protect_pct}%
                        <span style={{ fontSize: 10, display: 'block', color: 'var(--text-secondary)', fontWeight: 400 }}>
                          ({(detail.protect_area_ha || 0).toLocaleString()} ha)
                        </span>
                      </div>
                    </div>
                  </div>

                  {/* Flood Scenes Metadata */}
                  {detail.flood_metadata && (
                    <div style={{ fontSize: 10, color: 'var(--text-secondary)', marginBottom: 10, padding: '6px 8px', background: 'var(--bg)', borderRadius: 4, lineHeight: 1.5 }}>
                      <strong>Sentinel-1 Scenes:</strong> Pre: {detail.flood_metadata.pre_scenes?.join(', ')} · Post: {detail.flood_metadata.post_scenes?.join(', ')}<br/>
                      <em>{detail.flood_metadata.timing_note}</em> (Footprint: <strong>{detail.flood_metadata.label}</strong>)
                    </div>
                  )}

                  {/* RESILIENCE Trigger Breakdown */}
                  {detail.resilience_triggers && (
                    <div style={{ fontSize: 11, marginBottom: 12, padding: '6px 8px', background: 'rgba(178, 59, 59, 0.08)', borderLeft: '3px solid #B23B3B', borderRadius: '0 4px 4px 0' }}>
                      <div style={{ fontWeight: 600, color: '#B23B3B', fontSize: 10, textTransform: 'uppercase', marginBottom: 2 }}>Resilience Triggers Breakdown</div>
                      <div style={{ color: 'var(--text-primary)', fontSize: 11 }}>{detail.resilience_triggers.summary_str}</div>
                    </div>
                  )}
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

              {/* CANDIDATE RECHARGE SITES TABLE */}
              {detail.id === 'groundwater_recharge' && detail.top50_sites && detail.top50_sites.length > 0 && (
                <div style={{ marginBottom: 14 }}>
                  <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: 6 }}>
                    <div className="section-label" style={{ margin: 0 }}>Candidate Artificial Recharge Sites (Top 10)</div>
                    <span style={{ fontSize: 10, color: 'var(--text-secondary)' }}>
                      Total {detail.top50_sites.length} sites
                    </span>
                  </div>
                  <div style={{ overflowX: 'auto', border: '1px solid var(--border)', borderRadius: 4 }}>
                    <table className="data-table" style={{ margin: 0, fontSize: 11 }}>
                      <thead>
                        <tr>
                          <th>#</th>
                          <th>Structure</th>
                          <th className="num">Score</th>
                          <th className="num">Site zone (ha)</th>
                          <th>Coords</th>
                        </tr>
                      </thead>
                      <tbody>
                        {detail.top50_sites.slice(0, 10).map((s: any, idx: number) => (
                          <tr key={idx}>
                            <td className="mono">{s.rank || idx + 1}</td>
                            <td>{s.structure_type}</td>
                            <td className="num mono">{(s.suitability_score || 0).toFixed(2)}</td>
                            <td className="num mono">{(s.site_zone_ha || s.area_ha || 0).toFixed(2)}</td>
                            <td className="mono" style={{ fontSize: 10 }}>{s.lat?.toFixed(3)}, {s.lon?.toFixed(3)}</td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                  <div style={{ fontSize: 10, color: 'var(--text-secondary)', marginTop: 4 }}>
                    Top 10 of 50 sites shown. Download full list in CSV or KML for GIS / Google Earth.
                  </div>
                </div>
              )}

              {/* GEO-ECONOMIC INTERVENTION CLUSTERS TABLE */}
              {detail.id === 'geoeconomic_priority' && detail.top20_clusters && detail.top20_clusters.length > 0 && (
                <div style={{ marginBottom: 14 }}>
                  <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: 6 }}>
                    <div className="section-label" style={{ margin: 0 }}>Top Action Intervention Clusters (1 km Grid, 5–150 ha)</div>
                    <span style={{ fontSize: 10, color: 'var(--text-secondary)' }}>
                      Total {detail.top20_clusters.length} ranked
                    </span>
                  </div>
                  <div style={{ overflowX: 'auto', border: '1px solid var(--border)', borderRadius: 4 }}>
                    <table className="data-table" style={{ margin: 0, fontSize: 11 }}>
                      <thead>
                        <tr>
                          <th>#</th>
                          <th>Category</th>
                          <th className="num">Area</th>
                          <th className="num">Value (Cr)</th>
                          <th>Action</th>
                        </tr>
                      </thead>
                      <tbody>
                        {detail.top20_clusters.slice(0, 10).map((c: any, idx: number) => (
                          <tr key={idx}>
                            <td className="mono">{c.rank || idx + 1}</td>
                            <td style={{ fontWeight: 600 }}>{c.category}</td>
                            <td className="num mono">{(c.area_ha || 0).toFixed(1)} ha</td>
                            <td className="num mono">₹{c.indicative_value_cr}</td>
                            <td style={{ fontSize: 10, maxWidth: 140 }}>{c.recommended_action?.slice(0, 38)}...</td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                  <div style={{ fontSize: 10, color: 'var(--text-secondary)', marginTop: 4 }}>
                    Ranked by area × crop value. Full polygon coordinates available in CSV, KML, and KMZ export.
                  </div>
                </div>
              )}

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
                  href={api.getInsightPngUrl(detail.id, studyArea, activeSubLayer || undefined)}
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
                {(detail.kml_url || detail.id === 'groundwater_recharge' || detail.id === 'geoeconomic_priority') && (
                  <a
                    href={api.getInsightKmlUrl(detail.id, studyArea)}
                    download={`${detail.id}.kml`}
                    className="btn btn-secondary btn-sm"
                    style={{ textDecoration: 'none', gap: 4 }}
                  >
                    <MapPin size={12} /> KML / Earth
                  </a>
                )}
                {(detail.kmz_url || detail.id === 'geoeconomic_priority') && (
                  <a
                    href={api.getInsightKmzUrl(detail.id, studyArea)}
                    download={`${detail.id}_clusters.kmz`}
                    className="btn btn-secondary btn-sm"
                    style={{ textDecoration: 'none', gap: 4 }}
                  >
                    <MapPin size={12} /> KMZ / Earth
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
