# Industrial Data Matrix approach notes

## Current focus images

- `cropped.png`
- `yenitest.png`

## What the previous work shows

- Generic preprocessing plus `zxingcpp` / `pylibdmtx` is not enough for these dot-peen marks.
- For `cropped.png`, the geometry can be recovered: the successful path found a 20x20 grid and decoded only after a local 2-bit correction around uncertain cells.
- The failure mode is no longer mainly "where is the Data Matrix"; it is "which grid cells are truly occupied by dot-peen marks".
- Re-running many threshold/render/decode variants is too slow and image-specific for an industrial reader.

## More promising pipeline

1. Build dot likelihood maps.
   - Use bright-background dot-peen features rather than plain binarization.
   - Current useful channels are Hessian/blobness, FFT band-pass, local dark response, and inverted gray response.

2. Fit a stable lattice/frame.
   - Use the fixed factory camera/lighting assumption.
   - Treat the symbol as a near-regular lattice with small perspective/shear, not as arbitrary contours.

3. Search grid offset and module count.
   - Do not hard-code only one 20x20 hypothesis.
   - Score candidates with Data Matrix structure: solid L finder edges and alternating timing edges.

4. Convert cells to bits with per-cell confidence.
   - Store a score per cell, not just a binary threshold.
   - Use local row/column normalization and uncertainty margins.

5. Decode only the best few candidates.
   - Render candidate bit matrices with quiet zone.
   - Run direct decode first.
   - If needed, run a bounded local search only over the lowest-margin cells.

## Findings from `step11_industrial_grid_probe.py`

- This script is a diagnostic probe, not the final decoder.
- Current run:
  - `cropped.png`: best diagnostic grid is 20 modules, structural score `7.107`, direct decode empty.
  - `yenitest.png`: best diagnostic grid is 20 modules, structural score `6.935`, direct decode empty.
- `cropped.png` is still known to decode through the previous local correction path (`q48`, 2 cell flips).
- `yenitest.png` exposes the main weakness in the current implementation: the first frame fit finds useful dot axes, but the selected grid window and cell confidence model are not strong enough to decode.
- The next optimization should focus on grid-offset refinement and cell confidence before adding more decoder brute force.

## Working method from `step12_range_grid_decode.py`

- Both focus images decode with the same cell-reading method once the 20x20 grid frame is calibrated.
- Processing chain:
  1. Fix or calibrate a 20x20 grid frame.
  2. Build a `7x7` local range response: `dilate(gray) - erode(gray)`.
  3. For every grid cell, sample a `13x13` patch around the expected module center.
  4. Use the patch `p90` value as the dot-presence score.
  5. Threshold the 400 scores by quantile near `0.48..0.50`.
  6. Render the resulting bit matrix with quiet zone and decode with `zxingcpp`.
- Current verified results:
  - `cropped.png`: decodes at `q=0.48`, `q=0.49`, `q=0.50`.
  - `yenitest.png`: decodes at `q=0.47`, `q=0.48`, `q=0.49`, `q=0.50`.
- Decoded text for both current focus images:
  - `#8B3886177B ###=0302604331701S`

## Revised next step

- Turn the hard-coded frame values in `step12_range_grid_decode.py` into an automatic or one-time calibrated frame finder.
- Keep the range-grid cell classifier as the main bit extraction method; it is much more stable than direct Hough occupancy or generic thresholding.

## Useful sources checked

- MDPI 2025 DataMatrix recognition paper: emphasizes adaptive sampling grids and gray-trend binarization after coarse positioning, which matches our current failure mode.
- Dot-peen industry notes: dot-peen Data Matrix marks are permanent but low contrast; correct illumination and DPM-specific processing matter more than generic thresholding.
- Dot-peen deep-learning paper: confirms low contrast and partial degradation as common industrial issues, but a learned detector is more relevant for ROI localization than for our fixed-camera cell sampling problem.
