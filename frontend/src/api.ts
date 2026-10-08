// API client — all calls go to /api/* (proxied to localhost:8000)
const BASE = '/api'

async function request<T>(path: string, options?: RequestInit): Promise<T> {
  const res = await fetch(`${BASE}${path}`, {
    headers: { 'Content-Type': 'application/json' },
    ...options,
  })
  if (!res.ok) {
    const msg = await res.text().catch(() => 'unknown error')
    throw new Error(`${res.status} ${res.statusText}: ${msg}`)
  }
  return res.json() as Promise<T>
}

export type StudyAreaType = 'pilot' | 'taluks'

export interface StepStatus {
  name: string
  status: 'pending' | 'running' | 'done' | 'failed' | 'skipped'
  started_at: number | null
  finished_at: number | null
  duration: number | null
  log_tail: string[]
  error: string | null
}

export interface Metrics {
  overall_accuracy: number
  kappa: number
  metric_title?: string
  metric_explanation?: string
  is_seed_agreement?: boolean
  n_manual_val?: number
  confusion_matrix?: number[][]
  f1_per_class?: Record<string, number>
  feature_importance?: number[]
  n_train?: number
  n_test?: number
  class_names?: Record<string, string>
  counts_per_class?: Record<string, { seed: number; manual: number; total: number }>
  area_stats?: Record<string, number>
  analyzed_km2?: number
  total_taluk_km2?: number
  excluded_km2?: number
}

export interface AreaStat {
  class: string
  area_ha: number
  area_km2: number
}

export interface ZonalClassStat {
  name: string
  color: string
  ambasamudram_ha: number
  ambasamudram_pct: number
  cheranmahadevi_ha: number
  cheranmahadevi_pct: number
  total_ha: number
  total_pct: number
}

export interface ZonalStatsResponse {
  disclaimer: string
  excluded_km2: Record<string, number>
  total_taluk_km2: Record<string, number>
  analyzed_km2: Record<string, number>
  classes: ZonalClassStat[]
}

export interface MaskStatsResponse {
  total_taluk_km2: number
  excluded_km2: number
  analyzed_km2: number
  tree_below_120_kept_km2: number
  per_taluk: Record<string, { total_km2: number; excluded_km2: number; analyzed_km2: number }>
  resolution_m: number
}

export interface PipelineProgress {
  status: string
  current_step: string
  tiles_total: number
  tiles_done_features: number
  tiles_done_classify: number
  current_tile: string | null
  avg_tile_sec: number | null
  estimated_completion_sec: number | null
  estimated_finish_time: string | null
  exceeded_1230: boolean
  logs: string[]
}

export interface Profile {
  months: string[]
  ndvi: (number | null)[]
  ndwi: (number | null)[]
  vh: (number | null)[]
  is_interpolated?: boolean[]
  pixel: { row: number; col: number; x_utm: number; y_utm: number }
}

export interface ClassMeanStats {
  n: number
  color: string
  insufficient_samples: boolean
  ndvi_mean?: (number | null)[]
  ndvi_std?: (number | null)[]
  vh_mean?: (number | null)[]
  vh_std?: (number | null)[]
}

export interface ClassMeansResponse {
  months: string[]
  classes: Record<string, ClassMeanStats>
}

export interface DataAvailabilityItem {
  date: string
  cloud_free_pct: number
}

export interface InsightSummaryItem {
  id: string
  title: string
  subtitle: string
  key_finding: string
  status: 'ready' | 'pending' | 'failed' | 'partial'
}

export interface InsightDetail {
  id: string
  title: string
  subtitle: string
  key_finding: string
  extent_label?: string
  study_area?: string
  model_type?: string
  coverage_note?: string
  ay_label?: string
  total_cropland_ha?: number
  total_paddy_ha?: number
  total_area_ha?: number
  total_evaluated_ha?: number
  denominator_label?: string
  double_cropped_pct?: number
  single_cropped_pct?: number
  perennial_pct?: number
  kar_median_month?: string
  pishanam_median_month?: string
  kar_delayed_pct?: number
  pishanam_delayed_pct?: number
  total_parcels?: number
  critical_parcels?: number
  high_priority_parcels?: number
  village_count?: number
  villages?: any[]
  total_water_mcm?: number
  paddy_water_mcm?: number
  banana_water_mcm?: number
  paddy_water_share_pct?: number
  banana_water_share_pct?: number
  smallholder_share_pct?: number
  smallholder_count_pct?: number
  smallholder_area_pct?: number
  average_parcel_ha?: number
  median_parcel_ha?: number
  healthy_share_pct?: number
  stressed_share_pct?: number
  stressed_area_ha?: number
  stats?: Record<string, any>
  bins?: any[]
  chart_data?: any[]
  top50?: any[]
  legend?: { id: string | number; label: string; color: string }[]
  method: string
  limitations: string
  raster_url?: string
  png_url?: string
  csv_url?: string
  kml_url?: string
}

export const api = {
  getConfig: (studyArea: StudyAreaType = 'pilot') =>
    request<Record<string, any>>(`/config?study_area=${studyArea}`),
  getDataAvailability: (studyArea: StudyAreaType = 'pilot') =>
    request<DataAvailabilityItem[]>(`/data/availability?study_area=${studyArea}`),
  postAOI: (studyArea: StudyAreaType = 'pilot') =>
    request<{ area_km2: number; geometry: any; study_area?: string }>(`/aoi?study_area=${studyArea}`, { method: 'POST' }),
  getTalukBoundaries: () =>
    request<any>('/taluks/boundaries'),
  getTalukMask: () =>
    request<any>('/taluks/mask'),
  getTalukStats: () =>
    request<MaskStatsResponse>('/taluks/stats'),
  getPipelineStatus: (studyArea: StudyAreaType = 'pilot') =>
    request<StepStatus[]>(`/pipeline/status?study_area=${studyArea}`),
  getPipelineProgress: () =>
    request<PipelineProgress>('/pipeline/progress'),
  runPipeline: (from_step = 'acquire', studyArea: StudyAreaType = 'pilot') =>
    request<{ status: string }>(`/pipeline/run?from_step=${from_step}&study_area=${studyArea}`, { method: 'POST' }),
  getProfile: (lon: number, lat: number, studyArea: StudyAreaType = 'pilot') =>
    request<Profile>(`/profile?lon=${lon}&lat=${lat}&study_area=${studyArea}`),
  getClassMeans: (studyArea: StudyAreaType = 'pilot') =>
    request<ClassMeansResponse>(`/profiles/class_means?study_area=${studyArea}`),
  getDebugHistograms: (studyArea: StudyAreaType = 'pilot') =>
    request<Record<string, any>>(`/debug/histograms?study_area=${studyArea}`),
  getMetrics: (studyArea: StudyAreaType = 'pilot') =>
    request<Metrics>(`/results/metrics?study_area=${studyArea}`),
  getStats: (studyArea: StudyAreaType = 'pilot') =>
    request<AreaStat[]>(`/results/stats?study_area=${studyArea}`),
  getZonalStats: () =>
    request<ZonalStatsResponse>('/results/zonal_stats'),
  getParcelsGeojson: (studyArea: StudyAreaType = 'pilot', zoom?: number) =>
    zoom !== undefined
      ? `${BASE}/results/parcels.geojson?study_area=${studyArea}&zoom=${zoom}`
      : `${BASE}/results/parcels.geojson?study_area=${studyArea}`,
  getLabels: () =>
    request<{ type: string; features: any[] }>('/labels'),
  saveLabels: (fc: any) =>
    request<{ saved: number }>('/labels', { method: 'POST', body: JSON.stringify(fc) }),
  getLayerUrl: (layer: string, m?: number, studyArea: StudyAreaType = 'pilot') =>
    m !== undefined
      ? `${BASE}/layers/${layer}?m=${m}&study_area=${studyArea}`
      : `${BASE}/layers/${layer}?study_area=${studyArea}`,
  getLayerBounds: (layer: string, studyArea: StudyAreaType = 'pilot') =>
    request<{ bounds: [number, number, number, number] }>(`/layers/${layer}/bounds?study_area=${studyArea}`),
  getInsights: (studyArea: StudyAreaType = 'pilot') =>
    request<InsightSummaryItem[]>(`/insights?study_area=${studyArea}`),
  getInsightDetail: (insightId: string, studyArea: StudyAreaType = 'pilot') =>
    request<InsightDetail>(`/insights/${insightId}?study_area=${studyArea}`),
  computeAllInsights: (studyArea: StudyAreaType = 'pilot') =>
    request<{ status: string; insights: string[] }>(`/insights/compute_all?study_area=${studyArea}`, { method: 'POST' }),
  getInsightPngUrl: (insightId: string, studyArea: StudyAreaType = 'pilot') =>
    `${BASE}/insights/${insightId}/png?study_area=${studyArea}`,
  getInsightCsvUrl: (insightId: string, studyArea: StudyAreaType = 'pilot') =>
    `${BASE}/insights/${insightId}/csv?study_area=${studyArea}`,
  getInsightKmlUrl: (insightId: string, studyArea: StudyAreaType = 'pilot') =>
    `${BASE}/insights/${insightId}/kml?study_area=${studyArea}`,
  getAtlasPdfUrl: (studyArea: StudyAreaType = 'pilot') =>
    `${BASE}/insights/atlas/pdf?study_area=${studyArea}`,
  exportFile: (type: string, studyArea: StudyAreaType = 'pilot') =>
    `${BASE}/export/${type}?study_area=${studyArea}`,
  health: () =>
    request<{ status: string; demo_mode: boolean }>('/health'),
}

