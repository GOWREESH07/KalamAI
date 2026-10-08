import React, { useState } from 'react'
import { api, StepStatus, StudyAreaType } from '../api'
import { Download, FileText, Image as ImageIcon } from 'lucide-react'

interface Props {
  pipelineStatus: StepStatus[]
  onStatusChange: () => void
  studyArea?: StudyAreaType
}

export default function ScreenExport({ pipelineStatus, studyArea = 'pilot' }: Props) {
  const [downloading, setDownloading] = useState<string | null>(null)

  const downloadFile = (type: string, filename: string) => {
    setDownloading(type)
    const url = api.exportFile(type, studyArea)
    const a = document.createElement('a')
    a.href = url
    a.download = filename
    document.body.appendChild(a)
    a.click()
    document.body.removeChild(a)
    setTimeout(() => setDownloading(null), 1000)
  }

  return (
    <div style={{ display: 'flex', height: '100%', overflow: 'hidden' }}>
      <div style={{ flex: 1, padding: '24px 32px', overflowY: 'auto' }}>
        <div style={{ marginBottom: 20 }}>
          <div className="page-title">Export Products</div>
          <div className="page-subtitle">Thematic maps, classified GeoTIFF raster, vector parcels & stats</div>
        </div>

        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(280px, 1fr))', gap: 16, marginBottom: 24 }}>
          <div className="card">
            <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 6 }}>
              <ImageIcon size={16} color="var(--accent)" />
              <div style={{ fontWeight: 600, fontSize: 13 }}>Thematic Map (PNG)</div>
            </div>
            <div style={{ fontSize: 12, color: 'var(--text-secondary)', marginBottom: 12 }}>
              Publication-quality cartographic layout with scale bar, north arrow, legend & credits.
            </div>
            <button className="btn btn-primary btn-sm" onClick={() => downloadFile('map_png', 'thematic_map.png')}>
              <Download size={13} /> Download PNG
            </button>
          </div>

          <div className="card">
            <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 6 }}>
              <FileText size={16} color="var(--accent)" />
              <div style={{ fontWeight: 600, fontSize: 13 }}>Thematic Map (PDF)</div>
            </div>
            <div style={{ fontSize: 12, color: 'var(--text-secondary)', marginBottom: 12 }}>
              Print-ready vector PDF formatted for reports and submission packages.
            </div>
            <button className="btn btn-primary btn-sm" onClick={() => downloadFile('map_pdf', 'thematic_map.pdf')}>
              <Download size={13} /> Download PDF
            </button>
          </div>

          <div className="card">
            <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 6 }}>
              <FileText size={16} color="var(--accent)" />
              <div style={{ fontWeight: 600, fontSize: 13 }}>Classified GeoTIFF</div>
            </div>
            <div style={{ fontSize: 12, color: 'var(--text-secondary)', marginBottom: 12 }}>
              2-band GeoTIFF in EPSG:32643: Band 1 = Class ID, Band 2 = Probability/Confidence.
            </div>
            <button className="btn btn-secondary btn-sm" onClick={() => downloadFile('tif', 'classified.tif')}>
              <Download size={13} /> Download TIF
            </button>
          </div>

          <div className="card">
            <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 6 }}>
              <FileText size={16} color="var(--accent)" />
              <div style={{ fontWeight: 600, fontSize: 13 }}>Parcels GeoPackage</div>
            </div>
            <div style={{ fontSize: 12, color: 'var(--text-secondary)', marginBottom: 12 }}>
              Segmented field parcels with attributed majority class, confidence, and area in ha.
            </div>
            <button className="btn btn-secondary btn-sm" onClick={() => downloadFile('gpkg', 'parcels.gpkg')}>
              <Download size={13} /> Download GPKG
            </button>
          </div>

          <div className="card">
            <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 6 }}>
              <FileText size={16} color="var(--accent)" />
              <div style={{ fontWeight: 600, fontSize: 13 }}>Area Statistics (CSV)</div>
            </div>
            <div style={{ fontSize: 12, color: 'var(--text-secondary)', marginBottom: 12 }}>
              Tabular breakdown of classified area across all 6 land cover categories.
            </div>
            <button className="btn btn-secondary btn-sm" onClick={() => downloadFile('csv', 'area_stats.csv')}>
              <Download size={13} /> Download CSV
            </button>
          </div>
        </div>

        {/* Map Preview */}
        <div className="card">
          <div className="section-label" style={{ marginBottom: 12 }}>Thematic Map Preview</div>
          <div style={{ background: '#ECEAE4', padding: 12, borderRadius: 4, textAlign: 'center' }}>
            <img
              src={`/api/export/map_png?study_area=${studyArea}`}
              alt="Thematic Map Preview"
              style={{ maxWidth: '100%', maxHeight: 480, objectFit: 'contain', border: '1px solid var(--border)' }}
              onError={(e: any) => {
                e.target.style.display = 'none'
              }}
            />
            <div style={{ fontSize: 11, color: 'var(--text-secondary)', marginTop: 8 }}>
              Generated via Matplotlib with cartographic chrome: lat/lon grid, scale bar, north arrow, Copernicus citation.
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}
