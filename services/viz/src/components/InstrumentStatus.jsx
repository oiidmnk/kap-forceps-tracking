import { useStore } from '../feed/store.js'
import { instrumentMetrics } from '../geometry.js'

import { JAW_CLOSED_MM } from '../config.js'

// One compact row of instrument state: jaw opening, insertion depth from the
// trocar, and the shaft's angle to the retina where its axis would land.
export default function InstrumentStatus({ readout }) {
  const { frame, held } = useStore(readout)
  const m = frame ? instrumentMetrics(frame) : null
  const cells = [
    {
      label: 'Jaws',
      value: m ? (m.openingMm < JAW_CLOSED_MM ? 'closed' : 'open') : '—',
      unit: '',
      // '—' keeps the row height when there is no measurement, so the panels below never move.
      note: m ? `${m.openingMm.toFixed(1)} mm` : '—',
    },
    { label: 'Insertion', value: m ? m.insertionMm.toFixed(0) : '—', unit: 'mm' },
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
          <strong className={held ? 'measurement-held' : undefined}>
            {c.value}
            <small>{c.unit}</small>
          </strong>
          {c.note && <em className={held ? 'measurement-held' : undefined}>{c.note}</em>}
        </div>
      ))}
    </section>
  )
}
