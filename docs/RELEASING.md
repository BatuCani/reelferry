# Releasing Reelferry

Reelferry has no published release yet.
Its source lives in `BatuCani/reelferry`.
No signing credentials belong in this repository, and no workflow in it signs or publishes packages.

## Status

| System | Architecture | Package | Verification |
| --- | --- | --- | --- |
| macOS | Apple silicon, Intel | DMG and updater ZIP | Needs a Developer ID, notarization, stapling and Gatekeeper check |
| Windows | x64 | NSIS EXE | Needs Authenticode signatures, a timestamp and an installed runtime smoke |
| Linux | x64 | AppImage and DEB | Needs an installed/extracted runtime smoke and a signed checksum manifest |

The packaging tooling inherited from BridgeClip builds all four platform/architecture targets.
Windows ARM64 and Linux ARM64 are not release targets.
Do not advertise an unsupported OS version based only on the build runner version.

## Updates

The updater (`src/main/auto-updater.ts`) reads `latest-mac.yml`, `latest.yml`, or `latest-linux.yml` from the release GitHub marks as latest in `BatuCani/reelferry`.
When that release doesn't ship the running platform, `src/main/update-provider.ts` falls back to the newest published release that has the platform's feed, so platforms can ship on separate schedules.
Prereleases are ignored.

Updates stay off in every packaged build while `OFFICIAL_RELEASES` in `src/main/update-provider.ts` is `null`.
Set it to the release signing identity only together with the first signed release.
Updates are also off for unpackaged builds, for macOS builds not signed by that team, for a macOS app running from its disk image or quarantine, and when `REELFERRY_DISABLE_AUTO_UPDATE=1`.
Test a real signed update from the first installed version to the next before claiming update acceptance; a packaging rehearsal does not prove an upgrade path.

Never overwrite a published version.
Fixes use a new version and tag.
Withdraw a bad public version and replace it with a higher version so installed updaters can recover.

## Runtime reproduction

macOS: `bash scripts/prepare-resources.sh arm64` on Apple silicon, or `x64` on Intel.
This builds LGPL FFmpeg and stages its caption libraries and licenses.

Windows/Linux x64: `python scripts/release/stage-runtime.py` from a clean checkout.
Windows requires Visual C++ build tools for the relocatable yt-dlp launcher.
Linux requires `patchelf`.
Python and FFmpeg downloads are pinned by SHA-256 in `scripts/release/runtime-lock.json`.
An HTTPS mirror may be selected with `REELFERRY_FFMPEG_MIRROR`, preserving the same digest checks.
Upstream FFmpeg daily assets expire, so a release pipeline needs its own mirror of the exact archives.

Run `npm ci`, application and engine tests, dependency audits, `npm run build`, then electron-builder for the native target.
Signed Windows builds use `scripts/release/windows-config.cjs`, which requires `SIGNING_PUBLISHER`, `SIGNING_ENDPOINT`, `SIGNING_ACCOUNT` and `SIGNING_PROFILE` for Azure signing and sets `forceCodeSigning`.
Unsigned developer packages must never be labeled official releases.

`scripts/release/verify-runtime.py <packaged-resources-directory>` copies resources to a path containing spaces and tests the shipped Python, FFmpeg, framing model, captions, speed, audio, and downloader without relying on the checkout.
`scripts/release/verify-windows.ps1` installs the NSIS artifact and checks its publisher.
`scripts/release/collect-artifacts.cjs` rejects missing platforms, mismatched versions, and altered artifacts, and writes a checksum manifest.

Windows/Linux FFmpeg uses the LGPL shared upstream build, with OpenH264 for CPU encoding.
Its LGPL version and dependency set differ from the minimal macOS build.
Publishing a package that bundles FFmpeg requires a reviewed corresponding-source distribution for its exact dependency set.
A link to upstream build recipes alone is not that distribution.
