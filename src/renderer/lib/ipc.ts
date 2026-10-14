import type { ReelferryAPI } from '../../preload/index'

declare global {
  interface Window {
    reelferry: ReelferryAPI
  }
}

export function getApi(): ReelferryAPI {
  return window.reelferry
}
