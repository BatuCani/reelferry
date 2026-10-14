// Single source for product identity, shared by the main process (menu,
// window title) and the renderer.
export const APP_NAME = 'Reelferry'
export const APP_TAGLINE = 'Open-source AI video clipping'

export const REPO_URL = 'https://github.com/BatuCani/reelferry'
export const ISSUES_URL = `${REPO_URL}/issues`
export const LICENSE_NAME = 'Apache-2.0'
/** Reelferry is based on BridgeClip (MIT); Settings → About credits it. */
export const BRIDGECLIP_URL = 'https://github.com/bridge-mind/bridgeclip'

export const PROVIDER_LINKS = {
  openrouter: 'https://openrouter.ai/keys',
  zernio: 'https://zernio.com/dashboard/api-keys'
} as const

export const ZERNIO_LINKS = {
  signup: 'https://zernio.com/signup',
  pricing: 'https://zernio.com/pricing',
  billing: 'https://zernio.com/dashboard/billing'
} as const
