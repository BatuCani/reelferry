# Reelferry app icon

The app icon is a neutral placeholder: a white play symbol in a ring on a dark rounded tile.

## Source and exports

- **Source:** `resources/reelferry-icon.svg`, a hand-written vector using fills only, so every SVG renderer draws it the same.
- **Raster master:** `resources/reelferry-icon.png`, the source rendered at 1024 × 1024 with a transparent outer canvas.
- **App assets:** `build/icon.png` (1024px), `build/icon.icns` (macOS), and `build/icon.ico` (Windows: 16, 24, 32, 48, 64, 128, 256px).
- **Renderer:** the app imports the SVG source directly.

To change the artwork, edit the SVG, render the raster master, then export the app assets on macOS (Node, sips and iconutil required):

```sh
magick -background none -density 300 resources/reelferry-icon.svg -resize 1024x1024 PNG32:resources/reelferry-icon.png
npm run icons
```

Do not edit generated assets individually.
