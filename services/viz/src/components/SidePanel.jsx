export default function SidePanel({ children }) {
  return (
    <aside
      className="measurement-panel"
      aria-label="Retinal distance measurements"
    >
      {children}
    </aside>
  )
}
