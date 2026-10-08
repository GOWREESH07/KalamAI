import React from 'react'
import { BasemapType } from '../utils/basemaps'

interface Props {
  basemap: BasemapType
  onChange: (b: BasemapType) => void
  style?: React.CSSProperties
}

export default function BasemapSwitcher({ basemap, onChange, style }: Props) {
  return (
    <div
      style={{
        position: 'absolute',
        top: 10,
        right: 10,
        zIndex: 15,
        display: 'flex',
        background: 'rgba(10, 20, 36, 0.90)',
        backdropFilter: 'blur(6px)',
        padding: '3px 4px',
        borderRadius: 4,
        border: '1px solid #1F3A5F',
        gap: 3,
        boxShadow: '0 2px 8px rgba(0,0,0,0.3)',
        ...style,
      }}
    >
      {[
        { id: 'esri-world', label: 'Esri World Imagery' },
        { id: 'esri-clarity', label: 'Esri Clarity' },
        { id: 'osm', label: 'OSM' },
      ].map(b => (
        <button
          key={b.id}
          type="button"
          onClick={() => onChange(b.id as BasemapType)}
          style={{
            background: basemap === b.id ? '#1F3A5F' : 'transparent',
            color: basemap === b.id ? '#FFFFFF' : '#94A3B8',
            border: 'none',
            borderRadius: 3,
            padding: '4px 8px',
            fontSize: 10,
            fontWeight: basemap === b.id ? 600 : 400,
            cursor: 'pointer',
            transition: 'all 0.15s ease',
            whiteSpace: 'nowrap',
          }}
        >
          {b.label}
        </button>
      ))}
    </div>
  )
}
