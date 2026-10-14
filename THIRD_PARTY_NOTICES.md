# Third-party notices

Reelferry is licensed under the [Apache License 2.0](LICENSE).
It is based on [BridgeClip](https://github.com/bridge-mind/bridgeclip) v0.1.19, Copyright (c) 2026 BridgeMind LLC, licensed under the MIT license.
BridgeClip's license text and copyright notice are kept in [licenses/BridgeClip-MIT.txt](licenses/BridgeClip-MIT.txt) and apply to the portions of Reelferry that come from BridgeClip.
Neither license changes the licenses of bundled components.

| Component | Source and terms | Shipped notice |
| --- | --- | --- |
| BridgeClip v0.1.19 | [BridgeClip source](https://github.com/bridge-mind/bridgeclip), MIT, Copyright (c) 2026 BridgeMind LLC | `licenses/BridgeClip-MIT.txt`, packaged as `BridgeClip-MIT.txt` |
| Clipping engine and its assets | Engine source comes from BridgeClip and stays MIT; bundled fonts and YuNet retain their own licenses | `engine/LICENSE`, `engine/THIRD_PARTY_NOTICES.md`, and license files under `engine/assets/` |
| FFmpeg and ffprobe 8.1.3 (macOS) | [FFmpeg source](https://ffmpeg.org/download.html), LGPL 2.1 or later build with GPL, nonfree and version 3 components disabled | `engine-bin/FFMPEG-LICENSE`, `engine-bin/FFMPEG-SOURCE.txt`, `engine-bin/FFMPEG-SOURCE.tar.xz`; build options are in `scripts/build-ffmpeg-mac.sh` and the binary's `-version` output |
| FFmpeg and ffprobe 8.1.3 (Windows/Linux) | [BtbN shared LGPL build](https://github.com/BtbN/FFmpeg-Builds), LGPL v3 with GPL-only/nonfree components excluded; OpenH264 supplies CPU H.264 encoding | Exact archive digests and recipe commit in `scripts/release/runtime-lock.json`; packaged license, configuration and runtime provenance. Publication additionally requires a reviewed corresponding-source archive with dependency notices. |
| libass and its caption-shaping libraries | [libass source](https://github.com/libass/libass) and the upstream projects identified in the bundle; each retains its own license | `engine-bin/BUNDLED_LIBRARIES.txt` records exact installed versions and `engine-bin/THIRD_PARTY_LICENSES/` contains the notices. When Homebrew omits GLib's license, staging copies the LGPL 2.1 text from `scripts/licenses/`. |
| Python 3.12.14 | [Python](https://www.python.org/downloads/), Python Software Foundation license; packaged via [python-build-standalone](https://github.com/astral-sh/python-build-standalone) | `engine-venv/PYTHON-LICENSE` and distribution notices in `engine-venv/` |
| yt-dlp | [yt-dlp Python package](https://github.com/yt-dlp/yt-dlp), Unlicense for its own code, with separate terms for dependencies | Package metadata and license files in `engine-venv/`; `engine-bin/yt-dlp` is a shell launcher on macOS/Linux and a relocatable launcher from BridgeClip on Windows |
| Geist fonts | [Geist](https://github.com/vercel/geist-font), SIL Open Font License 1.1 | `Geist-OFL.txt` in the packaged app resources |
| Electron, React and other Node packages | Versions and package names in `package-lock.json`; each package retains its own license | Package metadata in the bundled app and the generated Node dependency inventory |

The renderer build also generates `RENDERER-THIRD-PARTY-LICENSES.txt` in the packaged resources, containing the installed license texts for the packages included in its JavaScript bundle. This is separate from the production Node inventory because renderer packages are installed as development dependencies. The build fails when a bundled renderer package has no license text.

Run details compare a run with OpusClip's published prices and processing times, naming the service in plain text. OpusClip is a trademark of its owner; Reelferry is not affiliated with it and bundles no OpusClip artwork.

The release packages the clipping engine committed in this repository. `scripts/prepare-resources.sh` and release CI stage the same pinned Python archive and hash-verified FFmpeg source. The final package still needs an asset and dependency license review, a signed/notarized install check, and a review of the generated inventories before publication.
