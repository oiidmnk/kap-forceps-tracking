import { useStore } from '../feed/store.js'
import { instrumentMetrics } from '../geometry.js'

const JAW_CLOSED_MM = 0.12 // tip separation below which the jaws count as closed

// One compact row of instrument state: jaw opening, insertion depth from the
// trocar, and the shaft's angle to the retina where its axis would land.
export default function InstrumentStatus({ readout }) {
  const frame = useStore(readout, (s) => s.frame)
  const m = frame ? instrumentMetrics(frame) : null
  const cells = [
    {
      label: 'Jaws',
      value: m ? m.openingMm.toFixed(2) : '—',
      unit: 'mm',
      note: m ? (m.openingMm < JAW_CLOSED_MM ? 'closed' : 'open') : null,
    },
    { label: 'Insertion', value: m ? m.insertionMm.toFixed(1) : '—', unit: 'mm' },
    {
      label: 'Angle to retina',
      value: m?.retinaAngleDeg != null ? Math.round(m.retinaAngleDeg) : '—',
      unit: '°',
    },
  ]
  return (
    <section className="instrument-panel" aria-label="Instrument state">
      {cells.map((c) => (
        <div key={c.label}>
          <span>{c.label}</span>
          <strong>
            {c.value}
            <small>{c.unit}</small>
          </strong>
          {c.note && <em>{c.note}</em>}
        </div>
      ))}
    </section>
  )
}
