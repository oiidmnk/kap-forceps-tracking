// Shared by all scene shortcuts. Read the mounted dialog directly so opening,
// closing and unmounting cannot leave a stale global flag behind.
export const navigationHelpOpen = () =>
  typeof document !== 'undefined' && !!document.querySelector('[data-navigation-help]')
