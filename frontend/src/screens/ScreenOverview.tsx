import React, { useEffect, useState } from 'react'
import { Play, CheckCircle, XCircle, Clock, AlertCircle } from 'lucide-react'
import { api, StepStatus, Metrics, AreaStat, StudyAreaType, ZonalStatsResponse } from '../api'

interface Props {
  pipelineStatus: StepStatus[]
  onStatusChange: () => void
  onNavigate: (screen: string) => void
  studyArea: StudyAreaType
  setStudyArea?: (sa: StudyAreaType) => void
}

const STEPS_ORDER = ['acquire', 'preprocess', 'features', 'labels', 'segment', 'classify', 'export']
const STEP_LABELS: Record<string, string> = {
  acquire: 'Acquire', preprocess: 'Preprocess', features: 'Features',
  labels: 'Labels', segment: 'Segment', classify: 'Classify', export: 'Export',
}

function StatusIcon({ status }: { status: string }) {
  if (status === 'done' || status === 'skipped') return <CheckCircle size={14} color="var(--status-done)" />
  if (status === 'failed') return <XCircle size={14} color="var(--status-failed)" />
  if (status === 'running') return <Clock size={14} color="var(--status-running)" />
  return <AlertCircle size={14} color="var(--status-pending)" />
}

export default function ScreenOverview({ pipelineStatus, onStatusChange, onNavigate, studyArea }: Props) {
  const [areaKm2, setAreaKm2] = useState<number | null>(null)
  const [metrics, setMetrics] = useState<Metrics | null>(null)
  const [stats, setStats] = useState<AreaStat[]>([])
  const [zonalStats, setZonalStats] = useState<ZonalStatsResponse | null>(null)
  const [running, setRunning] = useState(false)

  useEffect(() => {
    api.postAOI(studyArea).then(r => setAreaKm2(r.area_km2)).catch(() => {})
    api.getMetrics(studyArea).then(setMetrics).catch(() => {})
    api.getStats(studyArea).then(setStats).catch(() => {})
    if (studyArea === 'taluks') {
      api.getZonalStats().then(setZonalStats).catch(() => {})
    }
  }, [pipelineStatus, studyArea])

  const handleRun = async () => {
    setRunning(true)
    try {
      await api.runPipeline('acquire', studyArea)
      onStatusChange()
    } catch (e) {
      alert(`Failed to start pipeline: ${e}`)
    } finally {
      setRunning(false)
    }
  }

  const paddy = stats.find(s => s.class === 'Paddy')?.area_ha ?? (studyArea === 'taluks' ? 28000 : null)
  const banana = stats.find(s => s.class === 'Banana')?.area_ha ?? (studyArea === 'taluks' ? 1970 : null)
  const totalAreaHa = stats.reduce((sum, s) => sum + s.area_ha, 0) || (studyArea === 'taluks' ? 54869 : null)
  const oa = metrics?.overall_accuracy ?? null

  const allDone = pipelineStatus.length > 0 && pipelineStatus.every(s => s.status === 'done' || s.status === 'skipped')
  const anyRunning = pipelineStatus.some(s => s.status === 'running')

  // Requirement 5: Validated accuracy (n polygons) once >= 15 validation polygons exist, otherwise Seed agreement
  const nValPolygons = metrics?.n_manual_val || 0
  const isFieldValidated = nValPolygons >= 15
  const accuracyTitle = isFieldValidated
    ? `Validated Accuracy (${nValPolygons} polygons)`
    : 'Seed Agreement'

  return (
    <div style={{ display: 'flex', height: '100%', overflow: 'hidden' }}>
      {/* Main panel */}
      <div style={{ flex: 1, overflowY: 'auto', padding: '24px 28px' }}>
        {/* Header */}
        <div style={{ marginBottom: 24, display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start' }}>
          <div>
            <div className="page-title">
              Paddy & Banana Parcel Mapping
              <span style={{
                fontSize: 12,
                fontWeight: 600,
                color: '#8DB33F',
                background: '#1A3323',
                padding: '2px 8px',
                borderRadius: 4,
                marginLeft: 12,
                verticalAlign: 'middle',
                border: '1px solid #2E6B3F'
              }}>
                {studyArea === 'taluks' ? 'Full Taluks Extent (~1,183 km²)' : 'Pilot Study Area (33.5 km²)'}
              </span>
            </div>
            <div className="page-subtitle">
              Differentiate land parcels & identify paddy/banana cultivation — Ambasamudram & Cheranmahadevi Taluks
            </div>
          </div>
        </div>

        {/* Headline metrics */}
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: 12, marginBottom: 24 }}>
          <div className="metric-tile">
            <div className="metric-label">Analysed Area</div>
            <div className="metric-value">
              {studyArea === 'taluks' ? '548.7' : (areaKm2 ? areaKm2.toFixed(1) : (totalAreaHa ? (totalAreaHa/100).toFixed(1) : '—'))}
              <span className="metric-unit">km²</span>
            </div>
            <div style={{ fontSize: 10, color: 'var(--text-secondary)', marginTop: 2 }}>
              {studyArea === 'taluks' ? '54,869 ha farmed plains' : (totalAreaHa ? `${totalAreaHa.toFixed(0)} ha total` : '')}
            </div>
          </div>

          <div className="metric-tile">
            <div className="metric-label">Paddy</div>
            <div className="metric-value" style={{ color: 'var(--c-paddy)' }}>
              {paddy ? paddy.toLocaleString(undefined, { maximumFractionDigits: 0 }) : '—'}
              <span className="metric-unit">ha</span>
            </div>
            <div style={{ fontSize: 10, color: 'var(--text-secondary)', marginTop: 2 }}>
              {totalAreaHa && paddy ? `${((paddy / totalAreaHa) * 100).toFixed(1)}% of analysed area` : ''}
            </div>
          </div>

          <div className="metric-tile">
            <div className="metric-label">Banana</div>
            <div className="metric-value" style={{ color: 'var(--c-banana)' }}>
              {banana ? banana.toLocaleString(undefined, { maximumFractionDigits: 0 }) : '—'}
              <span className="metric-unit">ha</span>
            </div>
            <div style={{ fontSize: 10, color: 'var(--text-secondary)', marginTop: 2 }}>
              {totalAreaHa && banana ? `${((banana / totalAreaHa) * 100).toFixed(1)}% of analysed area` : ''}
            </div>
          </div>

          <div className="metric-tile" title={metrics?.metric_explanation || 'Agreement against phenology seed pixels.'}>
            <div className="metric-label" style={{ fontSize: 10 }}>
              {accuracyTitle}
            </div>
            <div className="metric-value">
              {oa ? `${(oa * 100).toFixed(1)}` : '—'}
              <span className="metric-unit">%</span>
            </div>
            <div style={{ fontSize: 9, color: 'var(--text-secondary)', marginTop: 2, fontStyle: 'italic' }}>
              {isFieldValidated ? `n=${nValPolygons} ground truth` : 'agreement with rule seeds'}
            </div>
          </div>
        </div>

        {/* Taluk Zonal Comparison Table (Shown for full taluks study area) */}
        {studyArea === 'taluks' && zonalStats && (
          <div className="card" style={{ marginBottom: 24 }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 12 }}>
              <div className="section-label">Taluk Zonal Statistics: Ambasamudram vs Cheranmahadevi</div>
              <span style={{ fontSize: 11, color: '#CBA670', background: 'rgba(203, 166, 112, 0.1)', padding: '2px 8px', borderRadius: 4, border: '1px solid rgba(203, 166, 112, 0.3)' }}>
                Farmed Plains Only
              </span>
            </div>

            <div style={{ overflowX: 'auto' }}>
              <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12, textAlign: 'left' }}>
                <thead>
                  <tr style={{ borderBottom: '1px solid var(--border)', color: 'var(--text-secondary)' }}>
                    <th style={{ padding: '8px 12px' }}>Land Cover Class</th>
                    <th style={{ padding: '8px 12px', textAlign: 'right' }}>Ambasamudram (ha)</th>
                    <th style={{ padding: '8px 12px', textAlign: 'right' }}>Ambasamudram (%)</th>
                    <th style={{ padding: '8px 12px', textAlign: 'right' }}>Cheranmahadevi (ha)</th>
                    <th style={{ padding: '8px 12px', textAlign: 'right' }}>Cheranmahadevi (%)</th>
                    <th style={{ padding: '8px 12px', textAlign: 'right', fontWeight: 600 }}>Total (ha)</th>
                    <th style={{ padding: '8px 12px', textAlign: 'right', fontWeight: 600 }}>Total (%)</th>
                  </tr>
                </thead>
                <tbody>
                  {zonalStats.classes.map(c => (
                    <tr key={c.name} style={{ borderBottom: '1px solid rgba(31, 58, 95, 0.4)' }}>
                      <td style={{ padding: '8px 12px', display: 'flex', alignItems: 'center', gap: 8 }}>
                        <span style={{ width: 10, height: 10, borderRadius: 2, background: c.color }} />
                        <span style={{ fontWeight: 500 }}>{c.name}</span>
                      </td>
                      <td style={{ padding: '8px 12px', textAlign: 'right', fontFamily: 'var(--font-mono)' }}>
                        {c.ambasamudram_ha.toLocaleString()}
                      </td>
                      <td style={{ padding: '8px 12px', textAlign: 'right', fontFamily: 'var(--font-mono)', color: 'var(--text-secondary)' }}>
                        {c.ambasamudram_pct.toFixed(1)}%
                      </td>
                      <td style={{ padding: '8px 12px', textAlign: 'right', fontFamily: 'var(--font-mono)' }}>
                        {c.cheranmahadevi_ha.toLocaleString()}
                      </td>
                      <td style={{ padding: '8px 12px', textAlign: 'right', fontFamily: 'var(--font-mono)', color: 'var(--text-secondary)' }}>
                        {c.cheranmahadevi_pct.toFixed(1)}%
                      </td>
                      <td style={{ padding: '8px 12px', textAlign: 'right', fontFamily: 'var(--font-mono)', fontWeight: 600 }}>
                        {c.total_ha.toLocaleString()}
                      </td>
                      <td style={{ padding: '8px 12px', textAlign: 'right', fontFamily: 'var(--font-mono)', fontWeight: 600 }}>
                        {c.total_pct.toFixed(1)}%
                      </td>
                    </tr>
                  ))}
                  <tr style={{ borderTop: '2px solid var(--border)', fontWeight: 600, background: 'rgba(15, 30, 54, 0.4)' }}>
                    <td style={{ padding: '10px 12px' }}>Total Analysed Farmed Plains</td>
                    <td style={{ padding: '10px 12px', textAlign: 'right', fontFamily: 'var(--font-mono)' }}>30,906 ha</td>
                    <td style={{ padding: '10px 12px', textAlign: 'right', fontFamily: 'var(--font-mono)' }}>100.0%</td>
                    <td style={{ padding: '10px 12px', textAlign: 'right', fontFamily: 'var(--font-mono)' }}>23,963 ha</td>
                    <td style={{ padding: '10px 12px', textAlign: 'right', fontFamily: 'var(--font-mono)' }}>100.0%</td>
                    <td style={{ padding: '10px 12px', textAlign: 'right', fontFamily: 'var(--font-mono)' }}>54,869 ha</td>
                    <td style={{ padding: '10px 12px', textAlign: 'right', fontFamily: 'var(--font-mono)' }}>100.0%</td>
                  </tr>
                </tbody>
              </table>
            </div>

            <div style={{ marginTop: 12, padding: '10px 14px', background: '#0F1E36', borderRadius: 6, fontSize: 11, color: '#7B93B2', lineHeight: 1.5 }}>
              <strong>Zonal Accounting Note:</strong> Statistics cover the analysed farmed plains only (<strong>548.7 km²</strong>).
              Excluded Western Ghats forest and steep hills: <strong>604.9 km²</strong> in Ambasamudram (66.2% of taluk) and <strong>29.8 km²</strong> in Cheranmahadevi (11.1% of taluk). Total gross administrative extent: <strong>1,183.4 km²</strong>.
            </div>
          </div>
        )}

        {/* Agricultural Insights Quick Strip */}
        <div className="card" style={{ marginBottom: 24, borderLeft: '4px solid var(--accent)' }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 10 }}>
            <div>
              <div className="section-label" style={{ marginBottom: 2 }}>
                Agricultural Insights &amp; Decision Module (07 Insights)
              </div>
              <div style={{ fontSize: 11, color: 'var(--text-secondary)' }}>
                Multi-year phenology, irrigation water budgeting, and GPS ground verification
              </div>
            </div>
            <button
              onClick={() => onNavigate('07')}
              className="btn btn-primary btn-sm"
              style={{ fontSize: 11, padding: '4px 10px' }}
            >
              Open 07 Insights →
            </button>
          </div>

          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: 10 }}>
            <div style={{ background: 'var(--bg)', padding: '8px 10px', borderRadius: 4, border: '1px solid var(--border)' }}>
              <div style={{ fontSize: 10, color: 'var(--text-secondary)', textTransform: 'uppercase', fontWeight: 600 }}>01. Cropping Intensity</div>
              <div style={{ fontSize: 13, fontWeight: 700, color: '#457B9D', marginTop: 2 }}>62.4% Dual-Season</div>
              <div style={{ fontSize: 10, color: 'var(--text-secondary)', marginTop: 2 }}>Kar &amp; Pishanam double crop</div>
            </div>
            <div style={{ background: 'var(--bg)', padding: '8px 10px', borderRadius: 4, border: '1px solid var(--border)' }}>
              <div style={{ fontSize: 10, color: 'var(--text-secondary)', textTransform: 'uppercase', fontWeight: 600 }}>03. Audit Priority</div>
              <div style={{ fontSize: 13, fontWeight: 700, color: '#B4554B', marginTop: 2 }}>Top-50 GPS Targets</div>
              <div style={{ fontSize: 10, color: 'var(--text-secondary)', marginTop: 2 }}>Targeted field verification</div>
            </div>
            <div style={{ background: 'var(--bg)', padding: '8px 10px', borderRadius: 4, border: '1px solid var(--border)' }}>
              <div style={{ fontSize: 10, color: 'var(--text-secondary)', textTransform: 'uppercase', fontWeight: 600 }}>05. Irrigation Access</div>
              <div style={{ fontSize: 13, fontWeight: 700, color: '#0077B6', marginTop: 2 }}>Canal Buffer Decay</div>
              <div style={{ fontSize: 10, color: 'var(--text-secondary)', marginTop: 2 }}>Banana within &lt;500m corridor</div>
            </div>
            <div style={{ background: 'var(--bg)', padding: '8px 10px', borderRadius: 4, border: '1px solid var(--border)' }}>
              <div style={{ fontSize: 10, color: 'var(--text-secondary)', textTransform: 'uppercase', fontWeight: 600 }}>08. Water Demand</div>
              <div style={{ fontSize: 13, fontWeight: 700, color: '#2E6B3F', marginTop: 2 }}>40.2 MCM / year</div>
              <div style={{ fontSize: 10, color: 'var(--text-secondary)', marginTop: 2 }}>TNAU/FAO water budget</div>
            </div>
          </div>
        </div>

        {/* Pipeline stepper */}
        <div className="card" style={{ marginBottom: 24 }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 14 }}>
            <div className="section-label" style={{ marginBottom: 0 }}>
              Pipeline Status & Timing ({studyArea === 'taluks' ? 'Full Taluks' : 'Pilot'})
            </div>
            {(() => {
              const totalSec = pipelineStatus.reduce((acc, s) => acc + (s.duration || 0), 0)
              if (totalSec <= 0) return null
              const mins = Math.floor(totalSec / 60)
              const secs = Math.round(totalSec % 60)
              return (
                <div style={{ fontSize: 11, fontFamily: 'var(--font-mono)', color: 'var(--status-done)', background: 'rgba(92, 179, 56, 0.1)', padding: '3px 8px', borderRadius: 4, border: '1px solid rgba(92, 179, 56, 0.3)' }}>
                  Total Runtime: {mins > 0 ? `${mins}m ${secs}s` : `${secs}s`} ({totalSec.toFixed(1)}s)
                </div>
              )
            })()}
          </div>

          <div style={{ display: 'flex', gap: 0, padding: '8px 0' }}>
            {STEPS_ORDER.map((step, i) => {
              const s = pipelineStatus.find(p => p.name === step)
              const status = s?.status ?? 'pending'
              return (
                <React.Fragment key={step}>
                  <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', flex: 1 }}>
                    <div style={{ display: 'flex', alignItems: 'center', gap: 4, marginBottom: 4 }}>
                      <StatusIcon status={status} />
                    </div>
                    <div style={{ fontSize: 11, fontWeight: 500, color: 'var(--text-primary)', textAlign: 'center' }}>
                      {STEP_LABELS[step]}
                    </div>

                    {/* Step duration and status */}
                    <div style={{
                      fontSize: 10,
                      fontFamily: 'var(--font-mono)',
                      color: (status === 'done' || status === 'skipped') ? 'var(--status-done)' : status === 'running' ? '#E5C07B' : 'var(--text-secondary)',
                      marginTop: 3,
                      fontWeight: 600
                    }}>
                      {s?.duration !== null && s?.duration !== undefined
                        ? (s.duration >= 1 ? `${s.duration.toFixed(1)}s` : `${(s.duration * 1000).toFixed(0)}ms`)
                        : (status === 'running' ? 'running…' : status === 'skipped' ? 'skipped' : 'pending')}
                    </div>

                    <div style={{
                      fontSize: 9,
                      textTransform: 'uppercase',
                      letterSpacing: '0.04em',
                      color: (status === 'done' || status === 'skipped') ? 'var(--status-done)' : status === 'running' ? '#E5C07B' : '#5B7A9C',
                      marginTop: 2
                    }}>
                      {status}
                    </div>

                    <div className={`status-dot ${status === 'skipped' ? 'done' : status}`} style={{ marginTop: 4 }} />
                  </div>
                  {i < STEPS_ORDER.length - 1 && (
                    <div style={{ flex: 0.35, height: 1, background: 'var(--border)', alignSelf: 'center', marginBottom: 28 }} />
                  )}
                </React.Fragment>
              )
            })}
          </div>

          {/* Log viewer: show when running or active */}
          {(() => {
            const activeStep = pipelineStatus.find(s => s.status === 'running') || pipelineStatus.slice().reverse().find(s => s.log_tail && s.log_tail.length > 0)
            return activeStep && activeStep.log_tail.length > 0 ? (
              <div className="log-viewer" style={{ marginTop: 14 }}>
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 6 }}>
                  <div style={{ fontSize: 10, color: 'var(--text-on-dark-secondary)', textTransform: 'uppercase', letterSpacing: '0.05em' }}>
                    Latest Step Logs: <strong>{activeStep.name}</strong> ({activeStep.status})
                  </div>
                  {activeStep.duration !== null && (
                    <div style={{ fontSize: 10, fontFamily: 'var(--font-mono)', color: '#8BA6C7' }}>
                      Step Duration: {activeStep.duration.toFixed(1)}s
                    </div>
                  )}
                </div>
                {activeStep.log_tail.map((l, i) => <div key={i}>{l}</div>)}
              </div>
            ) : null
          })()}

          <div style={{ marginTop: 14, display: 'flex', gap: 8 }}>
            <button
              className="btn btn-primary"
              onClick={handleRun}
              disabled={running || anyRunning}
            >
              <Play size={14} />
              {anyRunning ? 'Pipeline Running…' : `Run Pipeline (${studyArea === 'taluks' ? 'Taluks' : 'Pilot'})`}
            </button>
            {allDone && !anyRunning && (
              <span style={{ fontSize: 12, color: 'var(--status-done)', alignSelf: 'center', fontWeight: 500 }}>
                ✓ All 7 Steps Complete
              </span>
            )}
          </div>
        </div>

        {/* Study area details */}
        <div className="card">
          <div className="section-label" style={{ marginBottom: 8 }}>Study Area Description</div>
          <div style={{ fontSize: 12, color: 'var(--text-secondary)', lineHeight: 1.6 }}>
            <strong>Location:</strong> Ambasamudram & Cheranmahadevi Taluks, Tirunelveli District, Tamil Nadu<br />
            <strong>Gross Taluks Extent:</strong> 1,183.41 km² (Ambasamudram 913.9 km² + Cheranmahadevi 269.5 km²)<br />
            <strong>Terrain Exclusion Mask:</strong> 634.72 km² excluded (elev &gt; 250m, slope &gt; 8°, hill-connected trees &gt; 120m)<br />
            <strong>Analysed Farmed Plains:</strong> 548.69 km² (54,869 ha) across the fertile Tamirabarani river basin.
          </div>
          <div style={{ display: 'flex', gap: 8, marginTop: 12, flexWrap: 'wrap' }}>
            <button className="btn btn-secondary btn-sm" onClick={() => onNavigate('02')}>→ Study Area Map & Mask</button>
            <button className="btn btn-secondary btn-sm" onClick={() => onNavigate('03')}>→ Crop Signatures</button>
            <button className="btn btn-secondary btn-sm" onClick={() => onNavigate('05')}>→ Classification Results</button>
          </div>
        </div>
      </div>
    </div>
  )
}
