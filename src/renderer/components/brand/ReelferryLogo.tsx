import iconUrl from '../../../../resources/reelferry-icon.svg'
import { cn } from '../../lib/utils'

interface ReelferryLogoProps {
  /**
   * lockup: the app icon + "Reelferry" wordmark (for dark surfaces).
   * icon:   the app icon tile.
   * mark:   the icon alone, for tight spaces such as the collapsed sidebar.
   */
  variant?: 'lockup' | 'icon' | 'mark'
  /** Size by height (e.g. "h-6"); width follows the artwork. */
  className?: string
  alt?: string
}

/**
 * Reelferry's placeholder brand artwork. The icon source is
 * resources/reelferry-icon.svg; the wordmark is set in the app font.
 */
export function ReelferryLogo({ variant = 'lockup', className, alt = 'Reelferry' }: ReelferryLogoProps): React.JSX.Element {
  if (variant !== 'lockup') {
    return (
      <img
        src={iconUrl}
        alt={alt}
        draggable={false}
        className={cn('w-auto shrink-0 select-none', className)}
      />
    )
  }
  return (
    <svg
      viewBox="0 0 400 100"
      role="img"
      aria-label={alt}
      className={cn('w-auto shrink-0 select-none text-ink', className)}
    >
      <image href={iconUrl} x="-10" y="-10" width="120" height="120" />
      <text x="114" y="71" fill="currentColor" fontSize="64" fontWeight="600" letterSpacing="-1.5">Reelferry</text>
    </svg>
  )
}
