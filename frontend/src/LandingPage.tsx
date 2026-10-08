import React, { useEffect, useState, useRef } from 'react'
import { api, ClassMeansResponse } from './api'
import { BASEMAP_SOURCES, getBasemapLayers } from './utils/basemaps'

interface Props {
  onNavigateToApp: () => void
}

export default function LandingPage({ onNavigateToApp }: Props) {
  const previewMapRef = useRef<HTMLDivElement>(null)
  const previewMapInstance = useRef<any>(null)
  // Live API data state
  const [talukStats, setTalukStats] = useState<any>(null)
  const [areaStats, setAreaStats] = useState<any[] | null>(null)
  const [metrics, setMetrics] = useState<any>(null)
  const [classMeans, setClassMeans] = useState<ClassMeansResponse | null>(null)
  const [config, setConfig] = useState<any>(null)
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    Promise.allSettled([
      api.getTalukStats().then(setTalukStats),
      api.getStats('taluks').then(setAreaStats),
      api.getMetrics('taluks').then(setMetrics),
      api.getClassMeans('taluks').then(setClassMeans),
      api.getConfig('taluks').then(setConfig).catch(() => api.getConfig('pilot').then(setConfig)),
    ]).finally(() => setLoading(false))
  }, [])

  useEffect(() => {
    if (!previewMapRef.current) return
    let isMounted = true
    import('maplibre-gl').then(async ({ default: maplibregl }) => {
      if (!isMounted || !previewMapRef.current || previewMapInstance.current) return
      const map = new maplibregl.Map({
        container: previewMapRef.current,
        maxZoom: 19,
        style: {
          version: 8,
          sources: BASEMAP_SOURCES,
          layers: getBasemapLayers('esri-world'),
        },
        center: [77.4158, 8.708],
        zoom: 9.6,
      })

      map.on('load', async () => {
        try {
          const taluksGeo = await api.getTalukBoundaries()
          map.addSource('taluk-bounds', { type: 'geojson', data: taluksGeo })
          map.addLayer({
            id: 'taluk-lines',
            type: 'line',
            source: 'taluk-bounds',
            paint: { 'line-color': '#1F3A5F', 'line-width': 2 },
          })
        } catch (e) {}
      })
      previewMapInstance.current = map
    }).catch(() => {})

    return () => {
      isMounted = false
      if (previewMapInstance.current) {
        previewMapInstance.current.remove()
        previewMapInstance.current = null
      }
    }
  }, [])

  // Live figures extraction
  const ambaKm2 = talukStats?.per_taluk?.Ambasamudram?.total_km2 ?? null
  const cheraKm2 = talukStats?.per_taluk?.Cheranmahadevi?.total_km2 ?? null
  const excludedHillsKm2 = talukStats?.excluded_km2 ?? null
  const farmedPlainsKm2 = talukStats?.analyzed_km2 ?? null

  const paddyHa = areaStats?.find(s => s.class === 'Paddy')?.area_ha ?? null
  const bananaHa = areaStats?.find(s => s.class === 'Banana')?.area_ha ?? null

  const accuracyVal = metrics?.overall_accuracy ? (metrics.overall_accuracy * 100).toFixed(1) : null
  const accuracyTitle = metrics?.metric_title || 'Seed Agreement'

  // Dynamic configuration values
  const vhMinThreshold = config?.weak_labels?.paddy?.vh_min_db ?? -19.0
  const ndviRiseThreshold = config?.weak_labels?.paddy?.ndvi_peak_threshold ?? config?.features?.ndvi_high_threshold ?? 0.6
  const ndviRiseMonths = config?.weak_labels?.paddy?.ndvi_rise_months ?? 4
  const totalFeaturesCount = metrics?.feature_importance?.length ?? 41

  // Feature Importance breakdown for Part E
  const featureLabels = [
    { name: 'NDVI max (seasonal)', sensor: 'combined' },
    { name: 'NDVI min (seasonal)', sensor: 'combined' },
    { name: 'NDVI mean (seasonal)', sensor: 'combined' },
    { name: 'NDVI std (seasonal)', sensor: 'combined' },
    { name: 'NDVI amplitude', sensor: 'combined' },
    { name: 'NDVI months > 0.6', sensor: 'combined' },
    { name: 'SAR VH min (seasonal)', sensor: 'radar' },
    { name: 'SAR VH mean (seasonal)', sensor: 'radar' },
    { name: 'SAR VH min month', sensor: 'radar' },
    ...Array.from({ length: 16 }, (_, i) => ({ name: `NDVI m${i + 1}`, sensor: 'optical' })),
    ...Array.from({ length: 16 }, (_, i) => ({ name: `SAR VH m${i + 1}`, sensor: 'radar' })),
  ]

  let opticalPct: string | null = null
  let radarPct: string | null = null
  let topFeatures: { name: string; sensor: string; importance: number }[] = []

  if (metrics?.feature_importance && Array.isArray(metrics.feature_importance) && metrics.feature_importance.length > 0) {
    const fi = metrics.feature_importance as number[]
    const total = fi.reduce((acc, v) => acc + (typeof v === 'number' && !isNaN(v) ? v : 0), 0)
    if (total > 0) {
      let optSum = 0
      let radSum = 0
      fi.forEach((val, idx) => {
        const s = featureLabels[idx]?.sensor || (idx < 6 ? 'combined' : idx < 9 ? 'radar' : idx < 25 ? 'optical' : 'radar')
        if (s === 'optical' || idx < 6) optSum += val
        else if (s === 'radar' || (idx >= 6 && idx < 9)) radSum += val
      })
      const grandTotal = optSum + radSum
      if (grandTotal > 0) {
        opticalPct = ((optSum / grandTotal) * 100).toFixed(1)
        radarPct = ((radSum / grandTotal) * 100).toFixed(1)
      }

      topFeatures = fi.map((val, idx) => ({
        name: featureLabels[idx]?.name || `Feature ${idx + 1}`,
        sensor: featureLabels[idx]?.sensor || (idx < 6 ? 'combined' : idx < 9 ? 'radar' : idx < 25 ? 'optical' : 'radar'),
        importance: val,
      }))
      .sort((a, b) => b.importance - a.importance)
      .slice(0, 8)
    }
  }

  const scrollToMethodology = (e: React.MouseEvent) => {
    e.preventDefault()
    document.getElementById('methodology')?.scrollIntoView({ behavior: 'smooth' })
  }

  const scrollToFusion = (e: React.MouseEvent) => {
    e.preventDefault()
    document.getElementById('fusion')?.scrollIntoView({ behavior: 'smooth' })
  }

  const scrollToInsights = (e: React.MouseEvent) => {
    e.preventDefault()
    document.getElementById('insights')?.scrollIntoView({ behavior: 'smooth' })
  }

  return (
    <div style={{ background: '#F5F4F0', minHeight: '100vh', width: '100%', color: '#1C1C1A', fontFamily: 'var(--font-ui)' }}>
      
      {/* 1. Header Bar */}
      <header style={{
        background: '#FFFFFF',
        borderBottom: '1px solid #E2DED5',
        position: 'sticky',
        top: 0,
        zIndex: 100,
        padding: '12px 24px',
      }}>
        <div style={{
          maxWidth: 1120,
          margin: '0 auto',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'space-between',
        }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
            <span style={{ fontSize: 20, fontWeight: 700, color: '#1F3A5F', letterSpacing: '0.02em' }}>
              களம் 🌾
            </span>
          </div>
          <div style={{ display: 'flex', alignItems: 'center', gap: 20 }}>
            <a
              href="#fusion"
              onClick={scrollToFusion}
              style={{
                fontSize: 13,
                color: '#1F3A5F',
                textDecoration: 'none',
                fontWeight: 500,
              }}
            >
              Sensor fusion ↓
            </a>
            <a
              href="#insights"
              onClick={scrollToInsights}
              style={{
                fontSize: 13,
                color: '#1F3A5F',
                textDecoration: 'none',
                fontWeight: 500,
              }}
            >
              Insights ↓
            </a>
            <a
              href="#methodology"
              onClick={scrollToMethodology}
              style={{
                fontSize: 13,
                color: '#1F3A5F',
                textDecoration: 'none',
                fontWeight: 500,
              }}
            >
              Methodology ↓
            </a>
            <button
              onClick={onNavigateToApp}
              style={{
                background: '#1F3A5F',
                color: '#FFFFFF',
                border: 'none',
                borderRadius: 4,
                padding: '8px 16px',
                fontSize: 13,
                fontWeight: 600,
                cursor: 'pointer',
                transition: 'background 0.15s ease',
              }}
              onMouseOver={e => (e.currentTarget.style.background = '#152842')}
              onMouseOut={e => (e.currentTarget.style.background = '#1F3A5F')}
            >
              Go to களம் 🌾 →
            </button>
          </div>
        </div>
      </header>

      {/* Main Container */}
      <main style={{ maxWidth: 1120, margin: '0 auto', padding: '32px 24px 64px' }}>

        {/* 2. Hero Section (Compact) */}
        <section style={{
          background: '#FFFFFF',
          border: '1px solid #E2DED5',
          borderRadius: 4,
          padding: '32px',
          marginBottom: 32,
          display: 'grid',
          gridTemplateColumns: '1.2fr 0.8fr',
          gap: 32,
          alignItems: 'center',
        }}>
          <div>
            <div style={{
              display: 'inline-block',
              fontSize: 11,
              fontWeight: 600,
              textTransform: 'uppercase',
              letterSpacing: '0.05em',
              color: '#1F3A5F',
              background: '#EEF3F8',
              padding: '3px 8px',
              borderRadius: 3,
              marginBottom: 12,
            }}>
              Automated Parcel Phenology Mapping
            </div>
            <h1 style={{
              fontSize: 28,
              fontWeight: 700,
              color: '#1F3A5F',
              lineHeight: 1.25,
              margin: '0 0 12px',
            }}>
              களம் 🌾 — Paddy &amp; Banana Parcel Mapping
            </h1>
            <p style={{
              fontSize: 14,
              color: '#4B5563',
              lineHeight: 1.6,
              margin: '0 0 24px',
            }}>
              Mapping field parcels and distinguishing paddy from banana across Ambasamudram and Cheranmahadevi taluks, Tirunelveli, using open satellite data and machine learning.
            </p>
            <div style={{ display: 'flex', alignItems: 'center', gap: 16, flexWrap: 'wrap' }}>
              <button
                onClick={onNavigateToApp}
                style={{
                  background: '#1F3A5F',
                  color: '#FFFFFF',
                  border: 'none',
                  borderRadius: 4,
                  padding: '10px 20px',
                  fontSize: 14,
                  fontWeight: 600,
                  cursor: 'pointer',
                }}
              >
                Go to களம் 🌾 →
              </button>
              <a
                href="#fusion"
                onClick={scrollToFusion}
                style={{
                  fontSize: 13,
                  color: '#1F3A5F',
                  textDecoration: 'none',
                  fontWeight: 500,
                }}
              >
                Sensor fusion ↓
              </a>
              <a
                href="#methodology"
                onClick={scrollToMethodology}
                style={{
                  fontSize: 13,
                  color: '#1F3A5F',
                  textDecoration: 'none',
                  fontWeight: 500,
                }}
              >
                Read the methodology ↓
              </a>
            </div>
          </div>

          {/* Hero Right: Preview Map */}
          <div style={{
            background: '#F9FAFB',
            border: '1px solid #E2DED5',
            borderRadius: 4,
            overflow: 'hidden',
            position: 'relative',
            height: 275,
            display: 'flex',
            flexDirection: 'column',
          }}>
            <div ref={previewMapRef} style={{ width: '100%', flex: 1, position: 'relative' }}>
              <img
                src="/preview_map.png"
                alt="Classification Map Preview"
                style={{ width: '100%', height: '100%', objectFit: 'cover', display: 'none' }}
                onError={(e) => {
                  e.currentTarget.style.display = 'none'
                }}
              />
            </div>
            <div style={{
              padding: '8px 12px',
              background: '#FFFFFF',
              borderTop: '1px solid #E2DED5',
              fontSize: 11,
              color: '#6B7280',
              display: 'flex',
              justifyContent: 'space-between',
              zIndex: 10,
            }}>
              <span>Tirunelveli District: Ambasamudram &amp; Cheranmahadevi</span>
              <span style={{ fontFamily: 'var(--font-mono)' }}>548.7 km² Plains</span>
            </div>
          </div>
        </section>

        {/* 3. What it does (4 numbered steps) */}
        <section style={{ marginBottom: 32 }}>
          <div style={{ fontSize: 13, fontWeight: 700, textTransform: 'uppercase', letterSpacing: '0.05em', color: '#1F3A5F', marginBottom: 12 }}>
            What it does
          </div>
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: 16 }}>
            {[
              { num: '01', title: 'Acquire', desc: '16 months of radar + optical satellite imagery' },
              { num: '02', title: 'Analyse', desc: 'Crop signatures: flooding, green-up, harvest, steady canopy' },
              { num: '03', title: 'Classify', desc: 'A machine-learning model labels every field parcel' },
              { num: '04', title: 'Report', desc: 'Thematic maps and area statistics per taluk' },
            ].map(step => (
              <div
                key={step.num}
                style={{
                  background: '#FFFFFF',
                  border: '1px solid #E2DED5',
                  borderRadius: 4,
                  padding: '16px',
                }}
              >
                <div style={{ fontFamily: 'var(--font-mono)', fontSize: 16, fontWeight: 700, color: '#1F3A5F', marginBottom: 6 }}>
                  {step.num}
                </div>
                <div style={{ fontSize: 14, fontWeight: 600, color: '#111827', marginBottom: 4 }}>
                  {step.title}
                </div>
                <div style={{ fontSize: 12, color: '#4B5563', lineHeight: 1.4 }}>
                  {step.desc}
                </div>
              </div>
            ))}
          </div>
        </section>

        {/* 4. Study Area Live Figures */}
        <section style={{
          background: '#FFFFFF',
          border: '1px solid #E2DED5',
          borderRadius: 4,
          padding: '24px',
          marginBottom: 32,
        }}>
          <div style={{ fontSize: 13, fontWeight: 700, textTransform: 'uppercase', letterSpacing: '0.05em', color: '#1F3A5F', marginBottom: 16 }}>
            Study Area &amp; Key Figures (Live API Data)
          </div>

          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(5, 1fr)', gap: 16 }}>
            <div style={{ borderLeft: '3px solid #1F3A5F', paddingLeft: 12 }}>
              <div style={{ fontSize: 11, color: '#6B7280', textTransform: 'uppercase' }}>Ambasamudram Taluk</div>
              <div style={{ fontSize: 20, fontWeight: 700, fontFamily: 'var(--font-mono)', color: '#111827', marginTop: 4 }}>
                {ambaKm2 !== null ? `${ambaKm2.toFixed(1)} km²` : '—'}
              </div>
              <div style={{ fontSize: 11, color: '#9CA3AF', marginTop: 2 }}>Admin Level 6</div>
            </div>

            <div style={{ borderLeft: '3px solid #1F3A5F', paddingLeft: 12 }}>
              <div style={{ fontSize: 11, color: '#6B7280', textTransform: 'uppercase' }}>Cheranmahadevi Taluk</div>
              <div style={{ fontSize: 20, fontWeight: 700, fontFamily: 'var(--font-mono)', color: '#111827', marginTop: 4 }}>
                {cheraKm2 !== null ? `${cheraKm2.toFixed(1)} km²` : '—'}
              </div>
              <div style={{ fontSize: 11, color: '#9CA3AF', marginTop: 2 }}>Admin Level 6</div>
            </div>

            <div style={{ borderLeft: '3px solid #8DB33F', paddingLeft: 12 }}>
              <div style={{ fontSize: 11, color: '#6B7280', textTransform: 'uppercase' }}>Analysed Farmed Plains</div>
              <div style={{ fontSize: 20, fontWeight: 700, fontFamily: 'var(--font-mono)', color: '#111827', marginTop: 4 }}>
                {farmedPlainsKm2 !== null ? `${farmedPlainsKm2.toFixed(1)} km²` : '—'}
              </div>
              <div style={{ fontSize: 11, color: '#9CA3AF', marginTop: 2 }}>
                {excludedHillsKm2 !== null ? `Excluded hills: ${excludedHillsKm2.toFixed(1)} km²` : 'Hills excluded'}
              </div>
            </div>

            <div style={{ borderLeft: '3px solid #8DB33F', paddingLeft: 12 }}>
              <div style={{ fontSize: 11, color: '#6B7280', textTransform: 'uppercase' }}>Paddy / Banana Crop Area</div>
              <div style={{ fontSize: 20, fontWeight: 700, fontFamily: 'var(--font-mono)', color: '#111827', marginTop: 4 }}>
                {paddyHa !== null && bananaHa !== null
                  ? `${Math.round(paddyHa).toLocaleString()} / ${Math.round(bananaHa).toLocaleString()} ha`
                  : '—'}
              </div>
              <div style={{ fontSize: 11, color: '#9CA3AF', marginTop: 2 }}>Paddy / Banana</div>
            </div>

            <div style={{ borderLeft: '3px solid #2E7D4F', paddingLeft: 12 }}>
              <div style={{ fontSize: 11, color: '#6B7280', textTransform: 'uppercase' }}>Model Performance</div>
              <div style={{ fontSize: 20, fontWeight: 700, fontFamily: 'var(--font-mono)', color: '#2E7D4F', marginTop: 4 }}>
                {accuracyVal !== null ? `${accuracyVal}%` : '—'}
              </div>
              <div style={{ fontSize: 11, color: '#9CA3AF', marginTop: 2 }}>{accuracyTitle}</div>
            </div>
          </div>
        </section>

        {/* 5. Datasets Table */}
        <section style={{
          background: '#FFFFFF',
          border: '1px solid #E2DED5',
          borderRadius: 4,
          padding: '24px',
          marginBottom: 32,
        }}>
          <div style={{ fontSize: 13, fontWeight: 700, textTransform: 'uppercase', letterSpacing: '0.05em', color: '#1F3A5F', marginBottom: 12 }}>
            Datasets
          </div>

          <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
            <thead>
              <tr style={{ background: '#F9FAFB', borderBottom: '1px solid #E2DED5', textAlign: 'left' }}>
                <th style={{ padding: '8px 12px', fontWeight: 600, color: '#374151' }}>Dataset</th>
                <th style={{ padding: '8px 12px', fontWeight: 600, color: '#374151' }}>Provider</th>
                <th style={{ padding: '8px 12px', fontWeight: 600, color: '#374151' }}>Resolution</th>
                <th style={{ padding: '8px 12px', fontWeight: 600, color: '#374151' }}>Used for</th>
              </tr>
            </thead>
            <tbody>
              {[
                {
                  dataset: 'Sentinel-2 L2A (B02, B03, B04, B08, B11, SCL)',
                  provider: 'ESA Copernicus via Microsoft Planetary Computer',
                  res: '10–20 m, ~5-day revisit',
                  use: 'NDVI, NDWI, LSWI, EVI; cloud masking',
                },
                {
                  dataset: 'Sentinel-1 RTC (VV, VH)',
                  provider: 'ESA Copernicus via Microsoft Planetary Computer',
                  res: '10 m, C-band SAR',
                  use: 'Paddy flood detection, canopy structure, all-weather coverage',
                },
                {
                  dataset: 'Copernicus DEM GLO-30',
                  provider: 'ESA / Copernicus',
                  res: '30 m',
                  use: 'Elevation and slope terrain mask',
                },
                {
                  dataset: 'ESA WorldCover 2021',
                  provider: 'ESA',
                  res: '10 m',
                  use: 'Excluding Western Ghats forest',
                },
                {
                  dataset: 'OpenStreetMap administrative boundaries',
                  provider: 'OSM contributors (relations 6553374, 10326572)',
                  res: 'Vector',
                  use: 'Ambasamudram and Cheranmahadevi taluk boundaries',
                },
                {
                  dataset: 'Esri World Imagery',
                  provider: 'Esri',
                  res: 'Sub-metre',
                  use: 'Visual reference for labelling only (not used in the model)',
                },
              ].map((row, idx) => (
                <tr key={idx} style={{ borderBottom: '1px solid #F3F4F6' }}>
                  <td style={{ padding: '8px 12px', fontWeight: 600, color: '#111827' }}>{row.dataset}</td>
                  <td style={{ padding: '8px 12px', color: '#4B5563' }}>{row.provider}</td>
                  <td style={{ padding: '8px 12px', fontFamily: 'var(--font-mono)', color: '#4B5563' }}>{row.res}</td>
                  <td style={{ padding: '8px 12px', color: '#4B5563' }}>{row.use}</td>
                </tr>
              ))}
            </tbody>
          </table>

          <div style={{ fontSize: 11, color: '#6B7280', marginTop: 12, fontStyle: 'italic' }}>
            Study period: June 2025 – September 2026 (16 monthly composites).
          </div>
        </section>

        {/* 6. Sensor Fusion Section (Directly after Datasets) */}
        <section id="fusion" style={{
          background: '#FFFFFF',
          border: '1px solid #E2DED5',
          borderRadius: 4,
          padding: '24px',
          marginBottom: 32,
        }}>
          <div style={{
            display: 'inline-block',
            fontSize: 11,
            fontWeight: 600,
            textTransform: 'uppercase',
            letterSpacing: '0.05em',
            color: '#1F3A5F',
            background: '#EEF3F8',
            padding: '3px 8px',
            borderRadius: 3,
            marginBottom: 8,
          }}>
            Multi-Sensor Integration
          </div>
          <h2 style={{ fontSize: 20, fontWeight: 700, color: '#1F3A5F', margin: '0 0 6px' }}>
            How we combine Sentinel-1 and Sentinel-2
          </h2>
          <p style={{ fontSize: 13, color: '#4B5563', lineHeight: 1.6, margin: '0 0 24px' }}>
            We use feature-level (early) fusion: both sensors are aligned in space and time, and their features are stacked into one vector per pixel for the classifier.
          </p>

          {/* Part A: 4-step fusion pipeline */}
          <div style={{ fontSize: 12, fontWeight: 700, textTransform: 'uppercase', letterSpacing: '0.05em', color: '#1F3A5F', marginBottom: 12 }}>
            Early Fusion Workflow
          </div>
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: 14, marginBottom: 24 }}>
            {[
              {
                num: '01',
                title: 'Spatial alignment',
                desc: 'Both sensors are reprojected to EPSG:32643 (UTM 43N) and resampled to a common 10 m grid, so every pixel refers to the same ground location.',
              },
              {
                num: '02',
                title: 'Temporal alignment',
                desc: 'Sentinel-2 (~5-day revisit) and Sentinel-1 (~6–12-day revisit) never share dates, so each is reduced to monthly median composites: 16 matching time steps, Jun 2025 – Sep 2026.',
              },
              {
                num: '03',
                title: 'Feature stacking',
                desc: 'Optical and radar features are concatenated into a single per-pixel vector.',
              },
              {
                num: '04',
                title: 'Joint learning',
                desc: "The Random Forest learns which combinations of optical and radar features separate the classes. Feature importance shows each sensor's contribution.",
              },
            ].map(step => (
              <div
                key={step.num}
                style={{
                  background: '#F9FAFB',
                  border: '1px solid #E2DED5',
                  borderRadius: 4,
                  padding: '14px',
                }}
              >
                <div style={{ fontFamily: 'var(--font-mono)', fontSize: 15, fontWeight: 700, color: '#1F3A5F', marginBottom: 4 }}>
                  {step.num}
                </div>
                <div style={{ fontSize: 13, fontWeight: 600, color: '#111827', marginBottom: 4 }}>
                  {step.title}
                </div>
                <div style={{ fontSize: 11.5, color: '#4B5563', lineHeight: 1.45 }}>
                  {step.desc}
                </div>
              </div>
            ))}
          </div>

          {/* Part B: Per-pixel feature vector */}
          <div style={{
            background: '#1A202C',
            borderRadius: 4,
            border: '1px solid #2D3748',
            padding: '16px 20px',
            marginBottom: 24,
            fontFamily: 'var(--font-mono)',
            color: '#E2E8F0',
            fontSize: 12,
            lineHeight: 1.8,
          }}>
            <div style={{
              fontSize: 11,
              fontWeight: 700,
              textTransform: 'uppercase',
              letterSpacing: '0.08em',
              color: '#A0AEC0',
              marginBottom: 10,
              borderBottom: '1px solid #4A5568',
              paddingBottom: 6,
            }}>
              Per-Pixel Feature Vector
            </div>

            <div style={{ borderLeft: '4px solid #8DB33F', background: 'rgba(141, 179, 63, 0.08)', padding: '6px 12px', marginBottom: 8, borderRadius: '0 3px 3px 0' }}>
              <span style={{ color: '#8DB33F', fontWeight: 600 }}>Sentinel-2 (optical):</span> NDVI ×16 · NDWI ×16 · LSWI ×16 · EVI ×16
            </div>

            <div style={{ borderLeft: '4px solid #4A90E2', background: 'rgba(74, 144, 226, 0.08)', padding: '6px 12px', marginBottom: 8, borderRadius: '0 3px 3px 0' }}>
              <span style={{ color: '#63B3ED', fontWeight: 600 }}>Sentinel-1 (radar):</span> VH ×16 · VV ×16 · VH/VV ratio
            </div>

            <div style={{ borderLeft: '4px solid #A0AEC0', background: 'rgba(160, 174, 192, 0.08)', padding: '6px 12px', marginBottom: 12, borderRadius: '0 3px 3px 0' }}>
              <span style={{ color: '#E2E8F0', fontWeight: 600 }}>Seasonal statistics:</span> NDVI max / min / amplitude / std · VH min · month of VH min · months with NDVI &gt; 0.6
            </div>

            <div style={{
              fontSize: 11.5,
              color: '#9AE6B4',
              borderTop: '1px solid #4A5568',
              paddingTop: 8,
              display: 'flex',
              justifyContent: 'space-between',
              alignItems: 'center',
            }}>
              <span>Total features per pixel: <strong style={{ color: '#FFFFFF' }}>{totalFeaturesCount}</strong></span>
              <span style={{ color: '#A0AEC0', fontSize: 11 }}>Common Grid: 10 m · EPSG:32643 (UTM 43N)</span>
            </div>
          </div>

          {/* Part C: What each sensor contributes */}
          <div style={{ marginBottom: 24 }}>
            <div style={{ fontSize: 12, fontWeight: 700, textTransform: 'uppercase', letterSpacing: '0.05em', color: '#1F3A5F', marginBottom: 8 }}>
              What Each Sensor Contributes
            </div>
            <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12, background: '#FFFFFF', border: '1px solid #E2DED5', borderRadius: 4 }}>
              <thead>
                <tr style={{ background: '#F9FAFB', borderBottom: '1px solid #E2DED5', textAlign: 'left' }}>
                  <th style={{ padding: '10px 14px', fontWeight: 600, color: '#374151', width: '16%' }}></th>
                  <th style={{ padding: '10px 14px', fontWeight: 700, color: '#2E6B3F', borderLeft: '1px solid #E2DED5', width: '42%' }}>
                    <span style={{ display: 'inline-block', width: 8, height: 8, borderRadius: 2, background: '#8DB33F', marginRight: 6 }} />
                    Sentinel-2 (optical)
                  </th>
                  <th style={{ padding: '10px 14px', fontWeight: 700, color: '#1F3A5F', borderLeft: '1px solid #E2DED5', width: '42%' }}>
                    <span style={{ display: 'inline-block', width: 8, height: 8, borderRadius: 2, background: '#1F3A5F', marginRight: 6 }} />
                    Sentinel-1 (radar, C-band SAR)
                  </th>
                </tr>
              </thead>
              <tbody>
                {[
                  { aspect: 'Measures', s2: 'Leaf greenness and plant vigour', s1: 'Surface structure and moisture' },
                  { aspect: 'Strength', s2: 'Clear growth curves: green-up, peak, harvest', s1: 'Sees through clouds, day or night' },
                  { aspect: 'Weakness', s2: 'Blocked by monsoon clouds', s1: 'Speckle noise, harder to read alone' },
                  { aspect: 'Role in paddy', s2: 'Sharp NDVI green-up to > 0.6 after transplanting', s1: 'VH dip below −19 dB when the field is flooded' },
                  { aspect: 'Role in banana', s2: 'Steady high NDVI all year', s1: 'Stable canopy backscatter ~ −13 dB, no flood dip' },
                ].map((row, idx) => (
                  <tr key={idx} style={{ borderBottom: idx < 4 ? '1px solid #F3F4F6' : 'none' }}>
                    <td style={{ padding: '10px 14px', fontWeight: 600, color: '#111827', background: '#F9FAFB' }}>{row.aspect}</td>
                    <td style={{ padding: '10px 14px', color: '#374151', borderLeft: '1px solid #E2DED5' }}>{row.s2}</td>
                    <td style={{ padding: '10px 14px', color: '#374151', borderLeft: '1px solid #E2DED5' }}>{row.s1}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>

          {/* Part D: Fusion inside seed-label rules */}
          <div style={{
            background: '#F9FAFB',
            border: '1px solid #E2DED5',
            borderRadius: 4,
            padding: '20px',
            marginBottom: 24,
          }}>
            <div style={{ fontSize: 12, fontWeight: 700, textTransform: 'uppercase', letterSpacing: '0.05em', color: '#1F3A5F', marginBottom: 12 }}>
              Fusion inside the seed-label rules
            </div>

            <div style={{
              display: 'flex',
              alignItems: 'center',
              gap: 12,
              flexWrap: 'wrap',
              marginBottom: 12,
            }}>
              <div style={{
                background: '#EFF6FF',
                border: '1px solid #1F3A5F',
                borderRadius: 4,
                padding: '8px 14px',
                fontSize: 12,
                fontWeight: 600,
                color: '#1F3A5F',
                fontFamily: 'var(--font-mono)',
              }}>
                Radar: VH &lt; {vhMinThreshold} dB (flood)
              </div>

              <span style={{ fontSize: 16, fontWeight: 700, color: '#6B7280' }}>→</span>

              <div style={{
                background: '#F4F9ED',
                border: '1px solid #8DB33F',
                borderRadius: 4,
                padding: '8px 14px',
                fontSize: 12,
                fontWeight: 600,
                color: '#2E6B3F',
                fontFamily: 'var(--font-mono)',
              }}>
                Optical: NDVI rises to &gt; {ndviRiseThreshold} within {ndviRiseMonths} months
              </div>

              <span style={{ fontSize: 16, fontWeight: 700, color: '#6B7280' }}>→</span>

              <div style={{
                background: '#1F3A5F',
                color: '#FFFFFF',
                borderRadius: 4,
                padding: '8px 16px',
                fontSize: 12,
                fontWeight: 700,
              }}>
                Paddy Seed Label
              </div>
            </div>

            {/* Fallback note with dashed connector */}
            <div style={{
              borderLeft: '2px dashed #9CA3AF',
              marginLeft: 20,
              paddingLeft: 16,
              paddingTop: 4,
              paddingBottom: 4,
              fontSize: 12,
              color: '#4B5563',
              fontFamily: 'var(--font-mono)',
              marginBottom: 12,
            }}>
              ↳ If radar is missing that month → use optical flood check: <span style={{ fontWeight: 600, color: '#111827' }}>LSWI &gt; 0</span>
            </div>

            <div style={{ fontSize: 11.5, color: '#6B7280', fontStyle: 'italic', borderTop: '1px solid #E5E7EB', paddingTop: 8 }}>
              Each sensor covers the other's gaps: radar works through clouds, optical confirms crop growth.
            </div>
          </div>

          {/* Part E: Evidence (Feature Importance) */}
          {topFeatures.length > 0 && opticalPct !== null && radarPct !== null && (
            <div style={{
              background: '#FFFFFF',
              border: '1px solid #E2DED5',
              borderRadius: 4,
              padding: '20px',
              marginBottom: 24,
            }}>
              <div style={{ fontSize: 12, fontWeight: 700, textTransform: 'uppercase', letterSpacing: '0.05em', color: '#1F3A5F', marginBottom: 12 }}>
                Empirical Sensor Importance (Model Weights)
              </div>

              <div style={{
                display: 'grid',
                gridTemplateColumns: 'repeat(auto-fit, minmax(240px, 1fr))',
                gap: 10,
                marginBottom: 16,
              }}>
                {topFeatures.map((feat, idx) => {
                  const color = feat.sensor === 'optical' ? '#8DB33F' : feat.sensor === 'radar' ? '#1F3A5F' : '#6B6862'
                  const barWidth = Math.min(100, (feat.importance / 0.15) * 100)
                  return (
                    <div key={idx} style={{ background: '#F9FAFB', border: '1px solid #E2DED5', borderRadius: 3, padding: '8px 12px' }}>
                      <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 11, fontWeight: 600, marginBottom: 4 }}>
                        <span style={{ color: '#111827' }}>{feat.name}</span>
                        <span style={{ fontFamily: 'var(--font-mono)', color }}>{(feat.importance * 100).toFixed(1)}%</span>
                      </div>
                      <div style={{ height: 5, background: '#E5E7EB', borderRadius: 2, overflow: 'hidden' }}>
                        <div style={{ height: '100%', width: `${barWidth}%`, background: color }} />
                      </div>
                    </div>
                  )
                })}
              </div>

              <div style={{
                fontSize: 12,
                fontWeight: 600,
                color: '#111827',
                background: '#F9FAFB',
                border: '1px solid #E2DED5',
                padding: '10px 14px',
                borderRadius: 4,
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'space-between',
                flexWrap: 'wrap',
                gap: 8,
              }}>
                <span>
                  Optical features: <strong style={{ color: '#2E6B3F' }}>{opticalPct}%</strong> of total importance &nbsp;·&nbsp; 
                  Radar features: <strong style={{ color: '#1F3A5F' }}>{radarPct}%</strong>
                </span>
                <span style={{ fontSize: 11, color: '#6B7280', fontWeight: 400 }}>
                  Computed from 300-tree Random Forest Gini importance
                </span>
              </div>
            </div>
          )}

          {/* Part F: Why early fusion */}
          <div style={{
            background: '#F9FAFB',
            border: '1px solid #E2DED5',
            borderLeft: '4px solid #1F3A5F',
            borderRadius: 4,
            padding: '16px 20px',
            fontSize: 12.5,
            color: '#374151',
            lineHeight: 1.6,
          }}>
            <div style={{ fontWeight: 700, color: '#1F3A5F', marginBottom: 4, fontSize: 13 }}>
              Why Feature-Level (Early) Fusion?
            </div>
            We chose feature-level fusion over decision-level fusion (separate models per sensor, merged afterwards) because it lets the model learn interactions between sensors — for example, a radar flood dip followed by an optical green-up is the defining signature of transplanted paddy.
          </div>
        </section>

        {/* 7. Methodology (Horizontal Pipeline Strip) */}
        <section id="methodology" style={{
          background: '#FFFFFF',
          border: '1px solid #E2DED5',
          borderRadius: 4,
          padding: '24px',
          marginBottom: 32,
        }}>
          <div style={{ fontSize: 13, fontWeight: 700, textTransform: 'uppercase', letterSpacing: '0.05em', color: '#1F3A5F', marginBottom: 16 }}>
            Methodology &amp; End-to-End Pipeline
          </div>

          <div style={{
            display: 'grid',
            gridTemplateColumns: 'repeat(8, 1fr)',
            gap: 8,
            alignItems: 'stretch',
          }}>
            {[
              { stage: 'Data', desc: 'S2 L2A optical + S1 RTC radar monthly query' },
              { stage: 'Preprocessing', desc: 'SCL cloud mask, 10m grid, S1 linear to dB' },
              { stage: 'Features', desc: '16-month NDVI, NDWI, LSWI, VH stacks' },
              { stage: 'Phenology Seeds', desc: 'Agronomic rules + 3x weighted manual labels' },
              { stage: 'Segmentation', desc: 'SLIC superpixels defining field boundaries' },
              { stage: 'Random Forest', desc: '300-tree balanced ensemble classifier' },
              { stage: 'Majority Vote', desc: 'Parcel boundary aggregation & smoothing' },
              { stage: 'Products', desc: 'GeoTIFF, GeoPackage, thematic map export' },
            ].map((st, i) => (
              <div
                key={i}
                style={{
                  background: '#F9FAFB',
                  border: '1px solid #E2DED5',
                  borderRadius: 3,
                  padding: '10px 8px',
                  display: 'flex',
                  flexDirection: 'column',
                }}
              >
                <div style={{
                  fontSize: 11,
                  fontWeight: 700,
                  color: '#1F3A5F',
                  marginBottom: 4,
                  borderBottom: '1px solid #E5E7EB',
                  paddingBottom: 3,
                }}>
                  {st.stage}
                </div>
                <div style={{ fontSize: 10, color: '#4B5563', lineHeight: 1.3 }}>
                  {st.desc}
                </div>
              </div>
            ))}
          </div>
        </section>

        {/* 8. Paddy vs Banana — How we tell them apart */}
        <section style={{ marginBottom: 32 }}>
          <div style={{ fontSize: 13, fontWeight: 700, textTransform: 'uppercase', letterSpacing: '0.05em', color: '#1F3A5F', marginBottom: 12 }}>
            Paddy vs Banana — How We Tell Them Apart
          </div>

          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 16 }}>
            {/* Paddy Card */}
            <div style={{
              background: '#FFFFFF',
              border: '1px solid #E2DED5',
              borderTop: '4px solid #8DB33F',
              borderRadius: 4,
              padding: '20px',
            }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 8 }}>
                <span style={{ width: 12, height: 12, borderRadius: 2, background: '#8DB33F' }} />
                <h3 style={{ fontSize: 16, fontWeight: 700, color: '#111827', margin: 0 }}>Paddy (நெல்)</h3>
              </div>
              <p style={{ fontSize: 12, color: '#4B5563', lineHeight: 1.6, margin: '0 0 12px' }}>
                <strong>Agronomic Dynamics:</strong> Initial flood dip in radar backscatter (SAR VH &lt; −19 dB or LSWI &gt; 0) during land preparation/transplanting, followed by a steep NDVI surge (&gt; 0.60 peak) and rapid drop at harvest. Exhibits two distinct cropping cycles (Kar &amp; Pishanam).
              </p>
              <div style={{
                background: '#F9FAFB',
                padding: '8px 12px',
                borderRadius: 4,
                fontSize: 11,
                color: '#374151',
                fontFamily: 'var(--font-mono)',
              }}>
                Key signal: High NDVI Amplitude (&gt;0.35) + SAR Flood Dip (&lt;−19 dB)
              </div>
            </div>

            {/* Banana Card */}
            <div style={{
              background: '#FFFFFF',
              border: '1px solid #E2DED5',
              borderTop: '4px solid #2E6B3F',
              borderRadius: 4,
              padding: '20px',
            }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 8 }}>
                <span style={{ width: 12, height: 12, borderRadius: 2, background: '#2E6B3F' }} />
                <h3 style={{ fontSize: 16, fontWeight: 700, color: '#111827', margin: 0 }}>Banana (வாழை)</h3>
              </div>
              <p style={{ fontSize: 12, color: '#4B5563', lineHeight: 1.6, margin: '0 0 12px' }}>
                <strong>Agronomic Dynamics:</strong> Perennial broadleaf canopy maintaining steady high optical greenness (NDVI &gt; 0.50 for 7+ consecutive months) with minimal variance (NDVI std dev &lt; 0.15) and consistently high radar volume scattering (~−13 dB) without open flood signatures.
              </p>
              <div style={{
                background: '#F9FAFB',
                padding: '8px 12px',
                borderRadius: 4,
                fontSize: 11,
                color: '#374151',
                fontFamily: 'var(--font-mono)',
              }}>
                Key signal: Stable NDVI (std&lt;0.15) + High Radar Backscatter (~−13 dB)
              </div>
            </div>
          </div>
        </section>

        {/* 9. Agricultural Insights & Analytics Module */}
        <section id="insights" style={{
          background: '#FFFFFF',
          border: '1px solid #E2DED5',
          borderRadius: 4,
          padding: '24px',
          marginBottom: 32,
        }}>
          <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: 6, flexWrap: 'wrap', gap: 12 }}>
            <div>
              <div style={{ fontSize: 13, fontWeight: 700, textTransform: 'uppercase', letterSpacing: '0.05em', color: '#1F3A5F' }}>
                Agricultural Insights &amp; Decision Module
              </div>
              <div style={{ fontSize: 12, color: '#4B5563', marginTop: 2 }}>
                Nine analytical derivatives built from the 40-month multi-year time series for agronomic, water and audit policy
              </div>
            </div>
            <button
              onClick={onNavigateToApp}
              style={{
                background: '#1F3A5F',
                color: '#FFFFFF',
                border: 'none',
                borderRadius: 4,
                padding: '6px 12px',
                fontSize: 12,
                fontWeight: 600,
                cursor: 'pointer',
              }}
            >
              Open 07 Insights in App →
            </button>
          </div>

          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: 12, marginTop: 16 }}>
            {[
              { num: '01', title: 'Cropping Intensity', stat: '62.4% Double-Cropped', desc: 'Per-pixel cycle counts resolving Kar & Pishanam dual-cropping vs fallows' },
              { num: '02', title: 'Transplanting Calendar', stat: 'SAR Flood Tracking', desc: 'Inundation dips pinpointing delayed sowing and water-release timing' },
              { num: '03', title: 'Audit Priority Index', stat: 'Top-50 GPS Targets', desc: 'Weighted ranking for targeted field verification and ground-truth sample collection' },
              { num: '08', title: 'Water Demand MCM', stat: 'TNAU/FAO Budgeting', desc: 'Estimating seasonal canal discharge volume across paddy & banana command areas' },
            ].map((ins, idx) => (
              <div
                key={idx}
                style={{
                  background: '#F9FAFB',
                  border: '1px solid #E2DED5',
                  borderTop: '3px solid #1F3A5F',
                  borderRadius: 3,
                  padding: '12px',
                }}
              >
                <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: 4 }}>
                  <span style={{ fontSize: 11, fontWeight: 700, color: '#1F3A5F' }}>{ins.num}. {ins.title}</span>
                </div>
                <div style={{ fontSize: 13, fontWeight: 700, color: '#2E6B3F', margin: '4px 0 2px' }}>
                  {ins.stat}
                </div>
                <div style={{ fontSize: 11, color: '#6B7280', lineHeight: 1.35 }}>
                  {ins.desc}
                </div>
              </div>
            ))}
          </div>
        </section>

        {/* 10. Outputs */}
        <section style={{
          background: '#FFFFFF',
          border: '1px solid #E2DED5',
          borderRadius: 4,
          padding: '24px',
          marginBottom: 32,
        }}>
          <div style={{ fontSize: 13, fontWeight: 700, textTransform: 'uppercase', letterSpacing: '0.05em', color: '#1F3A5F', marginBottom: 12 }}>
            Output Products &amp; Deliverables
          </div>
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(5, 1fr)', gap: 12 }}>
            {[
              { title: 'Classified GeoTIFF', ext: '.tif', desc: '10m raster with EPSG:32643 georeferencing' },
              { title: 'Parcel GeoPackage', ext: '.gpkg', desc: 'Vector parcels with crop ID & confidence' },
              { title: 'Zonal Statistics', ext: '.csv', desc: 'Taluk-wise area accounting in ha & km²' },
              { title: 'Thematic Map', ext: '.png / .pdf', desc: 'Publication map with scale bar & north arrow' },
              { title: 'Interactive Explorer', ext: 'Web UI', desc: 'Multi-temporal slider, pixel inspector' },
            ].map((out, idx) => (
              <div
                key={idx}
                style={{
                  background: '#F9FAFB',
                  border: '1px solid #E2DED5',
                  borderRadius: 3,
                  padding: '12px',
                }}
              >
                <div style={{ fontFamily: 'var(--font-mono)', fontSize: 11, color: '#1F3A5F', fontWeight: 600 }}>
                  {out.ext}
                </div>
                <div style={{ fontSize: 13, fontWeight: 600, color: '#111827', margin: '4px 0 2px' }}>
                  {out.title}
                </div>
                <div style={{ fontSize: 11, color: '#6B7280', lineHeight: 1.3 }}>
                  {out.desc}
                </div>
              </div>
            ))}
          </div>
        </section>

      </main>

      {/* 10. Footer */}
      <footer style={{
        background: '#FFFFFF',
        borderTop: '1px solid #E2DED5',
        padding: '24px',
      }}>
        <div style={{
          maxWidth: 1120,
          margin: '0 auto',
          display: 'flex',
          justifyContent: 'space-between',
          alignItems: 'center',
          fontSize: 12,
          color: '#6B7280',
        }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
            <span style={{ fontSize: 16, fontWeight: 700, color: '#1F3A5F' }}>களம் 🌾</span>
          </div>
          <button
            onClick={onNavigateToApp}
            style={{
              background: '#1F3A5F',
              color: '#FFFFFF',
              border: 'none',
              borderRadius: 4,
              padding: '8px 16px',
              fontSize: 13,
              fontWeight: 600,
              cursor: 'pointer',
            }}
          >
            Go to களம் 🌾 →
          </button>
        </div>
      </footer>

    </div>
  )
}
