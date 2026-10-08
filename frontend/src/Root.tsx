import React, { useState, useEffect } from 'react'
import App from './App'
import LandingPage from './LandingPage'

export default function Root() {
  const getPath = () => {
    const path = window.location.pathname
    if (path.startsWith('/app')) return '/app'
    return '/'
  }

  const [currentPath, setCurrentPath] = useState<string>(getPath)

  useEffect(() => {
    const handlePopState = () => {
      setCurrentPath(getPath())
    }
    window.addEventListener('popstate', handlePopState)
    return () => window.removeEventListener('popstate', handlePopState)
  }, [])

  const navigateTo = (path: string) => {
    try {
      window.history.pushState(null, '', path)
    } catch {
      // fallback
    }
    setCurrentPath(path)
    window.scrollTo(0, 0)
  }

  if (currentPath === '/app') {
    return <App />
  }

  return <LandingPage onNavigateToApp={() => navigateTo('/app')} />
}
