import React from 'react'
import { createRoot } from 'react-dom/client'

// Load one UI at a time so workstation styles cannot affect the classic UI.
const appModule = new URLSearchParams(window.location.search).get('ui') === 'classic'
  ? import('./classic/App.jsx')
  : import('./App.jsx')

appModule.then(({ default: App }) => {
  createRoot(document.getElementById('root')).render(
    <React.StrictMode>
      <App />
    </React.StrictMode>,
  )
})
