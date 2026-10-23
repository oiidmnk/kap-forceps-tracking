import { useSyncExternalStore } from 'react'

// Minimal external store: high-rate tracking data lives outside React state so
// only the components that read it re-render, not the whole app.
export function createStore(initial) {
  let state = initial
  const listeners = new Set()
  return {
    get: () => state,
    set(next) {
      if (Object.is(next, state)) return
      state = next
      for (const l of listeners) l()
    },
    subscribe(listener) {
      listeners.add(listener)
      return () => listeners.delete(listener)
    },
  }
}

const identity = (s) => s

// Subscribe to a store; `selector` must return a stable value (primitive or an
// object taken from the state) so unrelated updates don't re-render.
export function useStore(store, selector = identity) {
  return useSyncExternalStore(store.subscribe, () => selector(store.get()))
}
