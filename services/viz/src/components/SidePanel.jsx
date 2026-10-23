import { forwardRef } from 'react'

const SidePanel = forwardRef(function SidePanel({ children }, ref) {
  return (
    <aside
      ref={ref}
      className="measurement-panel"
      aria-label="Retinal distance measurements"
    >
      {children}
    </aside>
  )
})
export default SidePanel
