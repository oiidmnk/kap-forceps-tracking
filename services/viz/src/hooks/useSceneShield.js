import { useEffect } from 'react'

// Overlay controls inside a 3D view sit within the element CameraControls
// listens on. React handles events at the root — after the controls have
// already seen them — so the shield uses native listeners on the overlay
// itself: pointer and wheel input there never orbits, pans or zooms.
export function useSceneShield(ref) {
  useEffect(() => {
    const el = ref.current
    if (!el) return
    const stop = (e) => e.stopPropagation()
    el.addEventListener('pointerdown', stop)
    el.addEventListener('wheel', stop)
    return () => {
      el.removeEventListener('pointerdown', stop)
      el.removeEventListener('wheel', stop)
    }
  }, [ref])
}
