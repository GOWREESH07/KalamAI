import React, { useState, useEffect, useCallback, useRef } from 'react'
import { Map, Layers, BarChart2, Download, Activity, Sparkles } from 'lucide-react'
import { api, StepStatus, StudyAreaType } from './api'
import ScreenOverview from './screens/ScreenOverview'
import ScreenStudyArea from './screens/ScreenStudyArea'
import ScreenCropSignatures from './screens/ScreenCropSignatures'
import ScreenResults from './screens/ScreenResults'
import ScreenExport from './screens/ScreenExport'
import ScreenInsights from './screens/ScreenInsights'

const NAV_ITEMS = [
  { id: '01', label: 'Overview', icon: Activity },
  { id: '02', label: 'Study Area & Data', icon: Map },
  { id: '03', label: 'Crop Signatures', icon: BarChart2 },
  { id: '05', label: 'Classification', icon: Layers },
  { id: '06', label: 'Export', icon: Download },
  { id: '07', label: 'Insights', icon: Sparkles },
]

const STEP_TO_NAV: Record<string, string> = {
  acquire: '02',
  preprocess: '02',
  features: '03',
  segment: '05',
  classify: '05',
  export: '06',
  insights: '07',
}

export default function App() {
  const [activeScreen, setActiveScreen] = useState('01')
  const [studyArea, setStudyArea] = useState<StudyAreaType>('pilot')
  const [pipelineStatus, setPipelineStatus] = useState<StepStatus[]>([])
  const pollingRef = useRef<ReturnType<typeof setInterval> | null>(null)

  const fetchStatus = useCallback(async () => {
    try {
      const status = await api.getPipelineStatus(studyArea)
      setPipelineStatus(status)
    } catch {
      // backend not ready yet
    }
  }, [studyArea])

  useEffect(() => {
    fetchStatus()
    pollingRef.current = setInterval(fetchStatus, 2000)
    return () => {
      if (pollingRef.current) clearInterval(pollingRef.current)
    }
  }, [fetchStatus])

  const getStepStatus = (stepName: string): StepStatus | undefined =>
    pipelineStatus.find(s => s.name === stepName)

  const getNavDotStatus = (navId: string): string => {
    const relevantSteps = Object.entries(STEP_TO_NAV)
      .filter(([, nid]) => nid === navId)
      .map(([s]) => s)
    if (relevantSteps.length === 0) return 'pending'
    const statuses = relevantSteps.map(s => getStepStatus(s)?.status ?? 'pending')
    if (statuses.some(s => s === 'failed')) return 'failed'
    if (statuses.some(s => s === 'running')) return 'running'
    if (statuses.every(s => s === 'done')) return 'done'
    return 'pending'
  }

  const screenProps = {
    pipelineStatus,
    onStatusChange: fetchStatus,
    studyArea,
    setStudyArea,
  }

  return (
    <div className="app-shell">
      {/* Sidebar */}
      <aside className="sidebar">
        <div className="sidebar-header">
          <div className="sidebar-logo">
            <span>களம்</span>
            <span>🌾</span>
          </div>
        </div>

        {/* Study Area Switcher */}
        <div style={{ padding: '0 12px 10px 12px' }}>
          <div style={{ fontSize: 10, textTransform: 'uppercase', letterSpacing: '0.05em', color: '#6A89B0', marginBottom: 4, fontWeight: 600 }}>
            Study Area
          </div>
          <div
            style={{
              display: 'flex',
              background: '#0B172A',
              padding: 3,
              borderRadius: 6,
              border: '1px solid #1F3A5F',
            }}
          >
            <button
              onClick={() => setStudyArea('pilot')}
              style={{
                flex: 1,
                padding: '5px 4px',
                fontSize: 11,
                fontWeight: studyArea === 'pilot' ? 600 : 400,
                background: studyArea === 'pilot' ? '#1F3A5F' : 'transparent',
                color: studyArea === 'pilot' ? '#E8F1F5' : '#7B93B2',
                border: 'none',
                borderRadius: 4,
                cursor: 'pointer',
                transition: 'all 0.15s ease',
              }}
            >
              Pilot (33 km²)
            </button>
            <button
              onClick={() => setStudyArea('taluks')}
              style={{
                flex: 1,
                padding: '5px 4px',
                fontSize: 11,
                fontWeight: studyArea === 'taluks' ? 600 : 400,
                background: studyArea === 'taluks' ? '#1F3A5F' : 'transparent',
                color: studyArea === 'taluks' ? '#E8F1F5' : '#7B93B2',
                border: 'none',
                borderRadius: 4,
                cursor: 'pointer',
                transition: 'all 0.15s ease',
              }}
            >
              Full taluks
            </button>
          </div>
        </div>

        <nav className="sidebar-nav">
          {NAV_ITEMS.map(item => {
            const Icon = item.icon
            const dotStatus = getNavDotStatus(item.id)
            return (
              <div
                key={item.id}
                className={`nav-item${activeScreen === item.id ? ' active' : ''}`}
                onClick={() => setActiveScreen(item.id)}
              >
                <span className="nav-step-num">{item.id}</span>
                <Icon size={14} />
                <span className="nav-label">{item.label}</span>
                <span className={`status-dot ${dotStatus}`} />
              </div>
            )
          })}
        </nav>
        <div style={{ padding: '12px 16px', borderTop: '1px solid #1F3A5F' }}>
          <div style={{ fontSize: 10, color: '#4A6585', lineHeight: 1.5 }}>
            Contains modified<br/>Copernicus Sentinel<br/>data 2025–26
          </div>
        </div>
      </aside>

      {/* Main area */}
      <div className="main-content">
        {activeScreen === '01' && <ScreenOverview {...screenProps} onNavigate={setActiveScreen} />}
        {activeScreen === '02' && <ScreenStudyArea {...screenProps} />}
        {activeScreen === '03' && <ScreenCropSignatures {...screenProps} />}
        {activeScreen === '05' && <ScreenResults {...screenProps} />}
        {activeScreen === '06' && <ScreenExport {...screenProps} />}
        {activeScreen === '07' && <ScreenInsights studyArea={studyArea} />}
      </div>
    </div>
  )
}
