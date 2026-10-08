export type BasemapType = 'esri-world' | 'esri-clarity' | 'osm'

export const BASEMAP_SOURCES: Record<string, any> = {
  'esri-world': {
    type: 'raster' as const,
    tiles: ['https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}'],
    tileSize: 256,
    maxzoom: 17,
    attribution: 'Tiles © Esri — Source: Esri, i-cubed, USDA, USGS, AEX, GeoEye, Getmapping, Aerogrid, IGN, IGP, UPR-EGP, and the GIS User Community',
  },
  'esri-clarity': {
    type: 'raster' as const,
    tiles: ['https://clarity.maptiles.arcgis.com/arcgis/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}'],
    tileSize: 256,
    maxzoom: 18,
    attribution: 'Imagery © Esri, Maxar, Earthstar Geographics',
  },
  'osm': {
    type: 'raster' as const,
    tiles: ['https://tile.openstreetmap.org/{z}/{x}/{y}.png'],
    tileSize: 256,
    maxzoom: 19,
    attribution: '© OpenStreetMap contributors',
  },
}

export function getBasemapLayers(activeBasemap: BasemapType = 'esri-world'): any[] {
  return [
    {
      id: 'basemap-esri-world',
      type: 'raster',
      source: 'esri-world',
      layout: { visibility: activeBasemap === 'esri-world' ? 'visible' : 'none' },
    },
    {
      id: 'basemap-esri-clarity',
      type: 'raster',
      source: 'esri-clarity',
      layout: { visibility: activeBasemap === 'esri-clarity' ? 'visible' : 'none' },
    },
    {
      id: 'basemap-osm',
      type: 'raster',
      source: 'osm',
      layout: { visibility: activeBasemap === 'osm' ? 'visible' : 'none' },
    },
  ]
}

export function updateBasemapVisibility(map: any, activeBasemap: BasemapType) {
  if (!map) return
  const basemaps: BasemapType[] = ['esri-world', 'esri-clarity', 'osm']
  basemaps.forEach(b => {
    const layerId = `basemap-${b}`
    if (map.getLayer(layerId)) {
      map.setLayoutProperty(layerId, 'visibility', activeBasemap === b ? 'visible' : 'none')
    }
  })
}
