import React, { useState, useRef, useEffect, useCallback } from 'react'
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, Area, ComposedChart
} from 'recharts'
import { api, Profile, StepStatus, ClassMeansResponse } from '../api'
import BasemapSwitcher from '../components/BasemapSwitcher'
import { BASEMAP_SOURCES, getBasemapLayers, updateBasemapVisibility, BasemapType } from '../utils/basemaps'

interface Props {
  pipelineStatus: StepStatus[]
  onStatusChange: () => void
  studyArea?: any
  setStudyArea?: any
}

const CLASS_COLORS: Record<string, string> = {
  Paddy: '#8DB33F',
  Banana: '#2E6B3F',
  'Other Veg': '#BFCB8F',
  'Built-up': '#B4554B',
  Water: '#3A7CA5',
  'Fallow/Bare': '#CBA670',
}

// Format "2025-06" -> "Jun '25"
function formatMonthLabel(mStr: string): string {
  if (!mStr) return ''
  const parts = mStr.split('-')
  if (parts.length < 2) return mStr
  const year = parts[0].slice(2)
  const monthNames = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
  const mIdx = parseInt(parts[1], 10) - 1
  return `${monthNames[mIdx] || parts[1]} '${year}`
}

const AOI_PILOT = { lon_min: 77.47, lat_min: 8.680, lon_max: 77.53, lat_max: 8.725 }
const TALUKS_BBOX = { lon_min: 77.1697, lat_min: 8.5339, lon_max: 77.6619, lat_max: 8.8820 }

export default function ScreenCropSignatures({ pipelineStatus, studyArea = 'pilot' }: Props) {
  const mapRef = useRef<HTMLDivElement>(null)
  const mapInstance = useRef<any>(null)
  const markerRef = useRef<any>(null)

  const isTaluks = studyArea === 'taluks'
  const currentAOI = isTaluks ? TALUKS_BBOX : AOI_PILOT

  const [basemap, setBasemap] = useState<BasemapType>('esri-world')
  const [profile, setProfile] = useState<Profile | null>(null)
  const [clickCoords, setClickCoords] = useState<{ lon: number; lat: number } | null>(null)
  const [coords, setCoords] = useState({ lon: 77.45, lat: 8.70 })
  const [loading, setLoading] = useState(false)
  const [err, setErr] = useState<string | null>(null)

  const [classMeans, setClassMeans] = useState<ClassMeansResponse | null>(null)
  const [configData, setConfigData] = useState<any>(null)
  const [loadingMeans, setLoadingMeans] = useState(false)
  const [visibleClasses, setVisibleClasses] = useState<string[]>(['Paddy', 'Banana'])

  const toggleClass = (cname: string) => {
    setVisibleClasses(prev =>
      prev.includes(cname) ? prev.filter(c => c !== cname) : [...prev, cname]
    )
  }

  const featuresReady = pipelineStatus.find(s => s.name === 'features')?.status === 'done'

  // Fetch /config for dynamic "Why it works" text
  useEffect(() => {
    api.getConfig(studyArea).then(setConfigData).catch(() => {})
  }, [studyArea])

  // Fetch real class mean profiles from classified pixels only when classification status changes
  const classifyStatus = pipelineStatus.find(s => s.name === 'classify')?.status
  useEffect(() => {
    setLoadingMeans(true)
    api.getClassMeans(studyArea)
      .then(res => setClassMeans(res))
      .catch(() => {})
      .finally(() => setLoadingMeans(false))
  }, [classifyStatus, studyArea])

  const fetchProfile = useCallback(async (lon: number, lat: number) => {
    // Check AOI bounds before making request
    if (lon < currentAOI.lon_min || lon > currentAOI.lon_max || lat < currentAOI.lat_min || lat > currentAOI.lat_max) {
      setProfile(null)
      const boundMsg = isTaluks
        ? 'Click within Ambasamudram & Cheranmahadevi taluk boundaries.'
        : `Click within the blue AOI boundary box (${AOI_PILOT.lon_min.toFixed(2)}–${AOI_PILOT.lon_max.toFixed(2)}°E, ${AOI_PILOT.lat_min.toFixed(3)}–${AOI_PILOT.lat_max.toFixed(3)}°N).`
      setErr(`Point is outside the study area. ${boundMsg}`)
      return
    }

    setLoading(true)
    setErr(null)
    try {
      const p = await api.getProfile(lon, lat, studyArea)
      setProfile(p)
    } catch (e: any) {
      setProfile(null)
      const boundMsg = isTaluks
        ? 'Click within Ambasamudram & Cheranmahadevi taluk boundaries.'
        : `Click within the blue AOI boundary box (${AOI_PILOT.lon_min.toFixed(2)}–${AOI_PILOT.lon_max.toFixed(2)}°E, ${AOI_PILOT.lat_min.toFixed(3)}–${AOI_PILOT.lat_max.toFixed(3)}°N).`
      if (String(e).includes('outside_aoi') || String(e).includes('400')) {
        setErr(`Point is outside the study area. ${boundMsg}`)
      } else {
        setErr(String(e))
      }
    } finally {
      setLoading(false)
    }
  }, [currentAOI, isTaluks, studyArea])

  const fetchProfileRef = useRef(fetchProfile)
  useEffect(() => {
    fetchProfileRef.current = fetchProfile
  }, [fetchProfile])

  useEffect(() => {
    if (!mapRef.current) return
    let isMounted = true

    import('maplibre-gl').then(async ({ default: maplibregl }) => {
      if (!isMounted || !mapRef.current) return
      if (mapInstance.current) {
        mapInstance.current.remove()
        mapInstance.current = null
      }

      const map = new maplibregl.Map({
        container: mapRef.current,
        maxZoom: 19,
        style: {
          version: 8,
          sources: BASEMAP_SOURCES,
          layers: getBasemapLayers(basemap),
        },
        bounds: [
          [currentAOI.lon_min, currentAOI.lat_min],
          [currentAOI.lon_max, currentAOI.lat_max]
        ],
        fitBoundsOptions: { padding: 40 },
      })
      map.getCanvas().style.cursor = 'crosshair'

      map.on('load', async () => {
        if (isTaluks) {
          try {
            const taluksGeo = await api.getTalukBoundaries()
            map.addSource('aoi-boundary', { type: 'geojson', data: taluksGeo })
            map.addLayer({
              id: 'aoi-boundary-fill',
              type: 'fill',
              source: 'aoi-boundary',
              paint: { 'fill-color': '#3A7CA5', 'fill-opacity': 0.05 },
            })
            map.addLayer({
              id: 'aoi-boundary-line',
              type: 'line',
              source: 'aoi-boundary',
              paint: { 'line-color': '#1F3A5F', 'line-width': 2.5 },
            })
          } catch (e) {
            console.error(e)
          }
        } else {
          // Draw 2px #1F3A5F line, no fill
          map.addSource('aoi-boundary', {
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
            id: 'aoi-boundary-line',
            type: 'line',
            source: 'aoi-boundary',
            paint: {
              'line-color': '#1F3A5F',
              'line-width': 2,
            },
          })
        }
      })

      map.on('click', (e) => {
        const { lng, lat } = e.lngLat
        setClickCoords({ lon: lng, lat: lat })

        // Crosshair marker
        if (!markerRef.current) {
          const el = document.createElement('div')
          el.innerHTML = `
            <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="#FFFFFF" stroke-width="2" style="filter: drop-shadow(0 0 2px rgba(0,0,0,0.8));">
              <circle cx="12" cy="12" r="6" stroke="#1F3A5F" stroke-width="2" fill="none"/>
              <line x1="12" y1="2" x2="12" y2="7" stroke="#1F3A5F" stroke-width="2"/>
              <line x1="12" y1="17" x2="12" y2="22" stroke="#1F3A5F" stroke-width="2"/>
              <line x1="2" y1="12" x2="7" y2="12" stroke="#1F3A5F" stroke-width="2"/>
              <line x1="17" y1="12" x2="22" y2="12" stroke="#1F3A5F" stroke-width="2"/>
            </svg>
          `
          el.style.width = '24px'
          el.style.height = '24px'
          el.style.cursor = 'crosshair'
          markerRef.current = new maplibregl.Marker({ element: el })
            .setLngLat([lng, lat])
            .addTo(map)
        } else {
          markerRef.current.setLngLat([lng, lat])
        }

        fetchProfileRef.current(lng, lat)
      })

      map.on('mousemove', (e) => {
        setCoords({ lon: parseFloat(e.lngLat.lng.toFixed(5)), lat: parseFloat(e.lngLat.lat.toFixed(5)) })
      })

      mapInstance.current = map
    })

    return () => {
      isMounted = false
      if (markerRef.current) {
        markerRef.current.remove()
        markerRef.current = null
      }
      if (mapInstance.current) {
        mapInstance.current.remove()
        mapInstance.current = null
      }
    }
  }, [studyArea])

  useEffect(() => {
    updateBasemapVisibility(mapInstance.current, basemap)
  }, [basemap])

  // Custom Dot for profile chart (hollow if interpolated/cloud gap-filled)
  const renderProfileDot = (props: any): React.ReactElement => {
    const { cx, cy, payload, dataKey } = props
    if (cx === undefined || cy === undefined || payload[dataKey] === null) {
      return <g key={`empty-${dataKey}-${payload?.month}`} />
    }
    const isHollow = payload.is_interpolated
    const color = dataKey === 'NDVI' ? '#8DB33F' : dataKey === 'NDWI' ? '#3A7CA5' : '#B7791F'
    if (isHollow) {
      return (
        <circle
          key={`${dataKey}-${payload.month}`}
          cx={cx}
          cy={cy}
          r={3.5}
          stroke={color}
          strokeWidth={1.5}
          fill="#FFFFFF"
        />
      )
    }
    return (
      <circle
        key={`${dataKey}-${payload.month}`}
        cx={cx}
        cy={cy}
        r={2.5}
        fill={color}
      />
    )
  }

  // Profile chart data
  const profileChartData = profile
    ? profile.months.map((m, i) => ({
        month: formatMonthLabel(m),
        rawMonth: m,
        NDVI: profile.ndvi[i],
        NDWI: profile.ndwi[i],
        VH: profile.vh[i],
        is_interpolated: profile.is_interpolated ? profile.is_interpolated[i] : false,
      }))
    : []

  // Real class mean profiles data
  const classMeansNdviData = classMeans
    ? classMeans.months.map((m, i) => {
        const row: Record<string, any> = { month: formatMonthLabel(m) }
        for (const [cname, cstats] of Object.entries(classMeans.classes)) {
          if (!cstats.insufficient_samples && cstats.ndvi_mean && cstats.ndvi_std) {
            const mean = cstats.ndvi_mean[i]
            const std = cstats.ndvi_std[i]
            const valShare = (cstats as any).valid_shares ? (cstats as any).valid_shares[i] : 1.0
            if (mean !== null && std !== null) {
              row[`${cname}_mean`] = mean
              row[`${cname}_upper`] = mean + std
              row[`${cname}_lower`] = mean - std
              row[`${cname}_band`] = [mean - std, mean + std]
              row[`${cname}_valid_share`] = valShare
            }
          }
        }
        return row
      })
    : []

  const classMeansVhData = classMeans
    ? classMeans.months.map((m, i) => {
        const row: Record<string, any> = { month: formatMonthLabel(m) }
        for (const [cname, cstats] of Object.entries(classMeans.classes)) {
          if (!cstats.insufficient_samples && cstats.vh_mean && cstats.vh_std) {
            const mean = cstats.vh_mean[i]
            const std = cstats.vh_std[i]
            const valShare = (cstats as any).valid_shares ? (cstats as any).valid_shares[i] : 1.0
            if (mean !== null && std !== null) {
              row[`${cname}_mean`] = mean
              row[`${cname}_upper`] = mean + std
              row[`${cname}_lower`] = mean - std
              row[`${cname}_valid_share`] = valShare
            }
          }
        }
        return row
      })
    : []

  // Dynamic "Why it works" text from config
  const pcfg = configData?.weak_labels?.paddy || {}
  const bcfg = configData?.weak_labels?.banana || {}

  return (
    <div style={{ display: 'flex', height: '100%', overflow: 'hidden' }}>
      {/* Map panel */}
      <div style={{ flex: '0 0 52%', position: 'relative' }}>
        <div className="map-wrap" ref={mapRef} style={{ height: '100%' }}>
          <BasemapSwitcher basemap={basemap} onChange={setBasemap} />

          <div className="coord-readout">
            {coords.lon.toFixed(5)}°E &nbsp; {coords.lat.toFixed(5)}°N
          </div>
          {!featuresReady && (
            <div style={{
              position: 'absolute', inset: 0, display: 'flex', alignItems: 'center',
              justifyContent: 'center', background: 'rgba(19,35,58,0.55)', zIndex: 10,
            }}>
              <div style={{ background: '#13233A', color: '#E8E4DC', padding: '16px 24px', borderRadius: 4, fontSize: 13 }}>
                Run pipeline to compute features & enable click-to-profile
              </div>
            </div>
          )}
        </div>
      </div>

      {/* Right data panel */}
      <div className="right-panel" style={{ width: 'auto', flex: 1 }}>
        <div className="panel-section">
          <div className="page-title" style={{ fontSize: 16 }}>Crop Temporal Signatures</div>
          <div className="page-subtitle">Real satellite time-series & classified pixel statistics</div>
        </div>

        <div className="panel-scroll">
          {/* Section 1: Pixel Profile */}
          <div className="section-label">Clicked Pixel Temporal Profile</div>
          {loading ? (
            <div className="inline-status"><span className="status-dot running" />Loading pixel profile…</div>
          ) : err ? (
            <div className="card" style={{ background: '#FFF5F5', borderColor: '#F5C6CB', marginBottom: 16 }}>
              <div style={{ fontSize: 12, color: 'var(--status-failed)', fontWeight: 500 }}>
                {err}
              </div>
              <div style={{ fontSize: 11, color: 'var(--text-secondary)', marginTop: 4 }}>
                {isTaluks
                  ? 'Click anywhere within the highlighted taluk boundaries on the map.'
                  : `Click within the blue AOI boundary box (${AOI_PILOT.lon_min.toFixed(2)}–${AOI_PILOT.lon_max.toFixed(2)}°E, ${AOI_PILOT.lat_min.toFixed(3)}–${AOI_PILOT.lat_max.toFixed(3)}°N).`}
              </div>
            </div>
          ) : profile ? (
            <div style={{ marginBottom: 16 }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 6 }}>
                <span style={{ fontSize: 11, color: 'var(--text-secondary)', fontFamily: 'var(--font-mono)' }}>
                  {clickCoords?.lon.toFixed(5)}°E, {clickCoords?.lat.toFixed(5)}°N (px: r{profile.pixel.row}, c{profile.pixel.col})
                </span>
                <span style={{ fontSize: 10, color: 'var(--text-secondary)', fontStyle: 'italic' }}>
                  hollow = gap-filled (cloud)
                </span>
              </div>

              <ResponsiveContainer width="100%" height={175}>
                <LineChart data={profileChartData} margin={{ top: 6, right: 0, left: -22, bottom: 0 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" strokeWidth={0.8} />
                  <XAxis dataKey="month" tick={{ fontSize: 9, fontFamily: 'var(--font-mono)' }} />
                  {/* Left Y-Axis: NDVI & NDWI fixed -0.2 to 1.0 */}
                  <YAxis
                    yAxisId="left"
                    domain={[-0.2, 1.0]}
                    ticks={[-0.2, 0.1, 0.4, 0.7, 1.0]}
                    tick={{ fontSize: 9, fontFamily: 'var(--font-mono)' }}
                  />
                  {/* Right Y-Axis: VH in dB fixed -28 to -5 */}
                  <YAxis
                    yAxisId="right"
                    orientation="right"
                    domain={[-28, -5]}
                    ticks={[-28, -20, -13, -5]}
                    tick={{ fontSize: 9, fontFamily: 'var(--font-mono)' }}
                  />
                  <Tooltip contentStyle={{ fontSize: 11, fontFamily: 'var(--font-mono)', border: '1px solid var(--border)' }} />
                  <Legend iconSize={10} wrapperStyle={{ fontSize: 11 }} />
                  {/* Linear curves with dots */}
                  <Line
                    yAxisId="left"
                    type="linear"
                    dataKey="NDVI"
                    stroke="#8DB33F"
                    strokeWidth={1.5}
                    dot={renderProfileDot}
                  />
                  <Line
                    yAxisId="left"
                    type="linear"
                    dataKey="NDWI"
                    stroke="#3A7CA5"
                    strokeWidth={1.5}
                    dot={renderProfileDot}
                  />
                  <Line
                    yAxisId="right"
                    type="linear"
                    dataKey="VH"
                    name="VH (dB)"
                    stroke="#B7791F"
                    strokeWidth={1.5}
                    dot={renderProfileDot}
                  />
                </LineChart>
              </ResponsiveContainer>
            </div>
          ) : (
            <div className="empty-state" style={{ padding: 14 }}>
              <div style={{ fontSize: 12 }}>Click any pixel on the map to extract its temporal signature</div>
            </div>
          )}

          <div className="sep" />

          {/* Section 2: Real Class Mean Profiles — NDVI */}
          <div className="section-label">Real Class Mean Profiles — NDVI (±1 Std Dev)</div>
          <div style={{ fontSize: 11, color: 'var(--text-secondary)', marginBottom: 8 }}>
            Derived from classified pixels. Showing ±1σ spatial standard deviation for active classes.
          </div>

          {/* Class Toggle Chips */}
          <div style={{ display: 'flex', gap: 6, flexWrap: 'wrap', marginBottom: 12 }}>
            {Object.keys(CLASS_COLORS).map(cname => {
              const active = visibleClasses.includes(cname)
              const col = CLASS_COLORS[cname]
              return (
                <button
                  key={cname}
                  onClick={() => toggleClass(cname)}
                  style={{
                    display: 'flex',
                    alignItems: 'center',
                    gap: 6,
                    padding: '3px 8px',
                    borderRadius: 14,
                    border: `1.5px solid ${col}`,
                    background: active ? col : 'transparent',
                    color: active ? '#FFFFFF' : 'var(--text-primary)',
                    fontSize: 11,
                    fontFamily: 'var(--font-ui)',
                    fontWeight: active ? 600 : 500,
                    cursor: 'pointer',
                    transition: 'all 0.15s ease',
                  }}
                >
                  <span
                    style={{
                      width: 6,
                      height: 6,
                      borderRadius: '50%',
                      background: active ? '#FFFFFF' : col,
                    }}
                  />
                  {cname}
                </button>
              )
            })}
          </div>

          {loadingMeans ? (
            <div className="inline-status"><span className="status-dot running" />Calculating class curves…</div>
          ) : classMeans ? (
            <div>
              <ResponsiveContainer width="100%" height={220}>
                <ComposedChart data={classMeansNdviData} margin={{ top: 6, right: 12, left: -22, bottom: 0 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" strokeWidth={0.8} />
                  <XAxis dataKey="month" tick={{ fontSize: 9, fontFamily: 'var(--font-mono)' }} />
                  <YAxis domain={[-0.2, 1.0]} ticks={[-0.2, 0.1, 0.4, 0.7, 1.0]} tick={{ fontSize: 9, fontFamily: 'var(--font-mono)' }} />
                  <Tooltip
                    contentStyle={{ fontSize: 11, fontFamily: 'var(--font-mono)', border: '1px solid var(--border)' }}
                    formatter={(value: any, name: string, item: any) => {
                      if (typeof value !== 'number') return [value, name]
                      const cname = name.split(' ')[0]
                      const share = item?.payload?.[`${cname}_valid_share`]
                      const shareStr = share !== undefined ? ` (${(share * 100).toFixed(0)}% valid px)` : ''
                      return [`${value.toFixed(3)}${shareStr}`, name]
                    }}
                  />
                  <Legend iconSize={10} wrapperStyle={{ fontSize: 11 }} />

                  {/* Lines & Shaded Bands for VISIBLE classes only */}
                  {Object.entries(classMeans.classes).map(([cname, cstats]) => {
                    if (cstats.insufficient_samples || !visibleClasses.includes(cname)) return null
                    const col = CLASS_COLORS[cname] || cstats.color
                    return (
                      <React.Fragment key={cname}>
                        <Area
                          type="linear"
                          dataKey={`${cname}_band`}
                          name={`${cname} ±1σ`}
                          stroke="none"
                          fill={col}
                          fillOpacity={0.16}
                          legendType="none"
                        />
                        <Line
                          type="linear"
                          dataKey={`${cname}_mean`}
                          name={`${cname} (n=${cstats.n.toLocaleString()})`}
                          stroke={col}
                          strokeWidth={2.0}
                          dot={{ r: 2.5 }}
                        />
                      </React.Fragment>
                    )
                  })}
                </ComposedChart>
              </ResponsiveContainer>

              {/* Insufficient sample warnings */}
              {Object.entries(classMeans.classes).filter(([, s]) => s.insufficient_samples).length > 0 && (
                <div style={{ fontSize: 10, color: 'var(--text-secondary)', marginTop: 4 }}>
                  * Classes with insufficient samples (&lt; 100 pixels):{' '}
                  {Object.entries(classMeans.classes)
                    .filter(([, s]) => s.insufficient_samples)
                    .map(([name, s]) => `${name} (n=${s.n})`)
                    .join(', ')}
                </div>
              )}
            </div>
          ) : (
            <div className="empty-state" style={{ padding: 14 }}>
              <div style={{ fontSize: 12 }}>Run classification to generate real class mean curves</div>
            </div>
          )}

          <div className="sep" />

          {/* Section 3: Real Class Mean Profiles — SAR VH Backscatter */}
          <div className="section-label">Real Class Mean Profiles — SAR VH Backscatter (dB)</div>
          <div style={{ fontSize: 11, color: 'var(--text-secondary)', marginBottom: 8 }}>
            Radar cross-section temporal trajectory for visible classes.
          </div>

          {classMeans && (
            <ResponsiveContainer width="100%" height={220}>
              <LineChart data={classMeansVhData} margin={{ top: 6, right: 12, left: -22, bottom: 0 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" strokeWidth={0.8} />
                <XAxis dataKey="month" tick={{ fontSize: 9, fontFamily: 'var(--font-mono)' }} />
                <YAxis domain={[-26, -10]} ticks={[-26, -21, -16, -10]} tick={{ fontSize: 9, fontFamily: 'var(--font-mono)' }} />
                <Tooltip
                  contentStyle={{ fontSize: 11, fontFamily: 'var(--font-mono)', border: '1px solid var(--border)' }}
                  formatter={(value: any, name: string, item: any) => {
                    if (typeof value !== 'number') return [value, name]
                    const cname = name.split(' ')[0]
                    const share = item?.payload?.[`${cname}_valid_share`]
                    const shareStr = share !== undefined ? ` (${(share * 100).toFixed(0)}% valid px)` : ''
                    return [`${value.toFixed(2)} dB${shareStr}`, name]
                  }}
                />
                <Legend iconSize={10} wrapperStyle={{ fontSize: 11 }} />
                {Object.entries(classMeans.classes).map(([cname, cstats]) => {
                  if (cstats.insufficient_samples || !visibleClasses.includes(cname)) return null
                  const col = CLASS_COLORS[cname] || cstats.color
                  return (
                    <Line
                      key={cname}
                      type="linear"
                      dataKey={`${cname}_mean`}
                      name={`${cname} VH (n=${cstats.n.toLocaleString()})`}
                      stroke={col}
                      strokeWidth={2.0}
                      dot={{ r: 2.5 }}
                    />
                  )
                })}
              </LineChart>
            </ResponsiveContainer>
          )}

          <div className="sep" />

          {/* Section 4: Dynamic "Why it works" read from config */}
          <div className="section-label">Phenological Discrimination Logic (from config.yaml)</div>
          <div style={{ fontSize: 11, color: 'var(--text-secondary)', lineHeight: 1.6 }}>
            <div>
              <strong>Paddy Dynamics:</strong> SAR VH minimum &lt; {pcfg.vh_min_db ?? -19.0} dB during initial flooding
              accompanied by standing water index (LSWI/NDWI &gt; 0.0), followed by an NDVI surge to &gt; {pcfg.ndvi_peak_threshold ?? 0.60}
              within {pcfg.ndvi_rise_months ?? 3} months of transplanting, with seasonal NDVI amplitude &gt; {pcfg.ndvi_amplitude ?? 0.40}.
            </div>
            <div style={{ marginTop: 6 }}>
              <strong>Banana Plantation Dynamics:</strong> NDVI &gt; {bcfg.ndvi_threshold ?? 0.50} sustained across
              &ge; {bcfg.ndvi_months_above ?? 7} months within the best rolling {bcfg.rolling_months ?? 12}-month window,
              phenological stability with NDVI standard deviation &lt; {bcfg.ndvi_std_max ?? 0.15}, and absence of flooding signal
              (VH min &gt; {bcfg.vh_min_threshold_db ?? -18.0} dB).
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}
