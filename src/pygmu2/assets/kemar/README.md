# KEMAR HRTF WAV files

This directory holds MIT KEMAR compact HRTF impulse responses as stereo WAV files (44.1 kHz).

## Setup

1. **Download** the compact KEMAR dataset:
   - https://sound.media.mit.edu/resources/KEMAR/
   - Get `compact.zip` (or `KEMAR.zip`).

2. **Unzip** directly into this directory (`pygmu2/assets/kemar`, i.e. `src/pygmu2/assets/kemar` in the repo). The archive already contains `.wav` files—no conversion is required.

   If your archive has only `.dat` files, use the conversion script instead:
   ```bash
   python scripts/convert_kemar_to_wav.py --input /path/to/extracted_dat --output src/pygmu2/assets/kemar
   ```

**Naming convention:** `H{elevation}e{azimuth}a.wav`
   - **Elevation:** Optional sign (`-` or none) followed by a two-digit number (degrees), e.g. `-40`, `00`, `90`.
   - **Azimuth:** Three-digit number (degrees), e.g. `000`, `045`, `180`.
   - Examples: `H00e045a.wav` = 0° elevation, 45° azimuth; `H-40e090a.wav` = -40° elevation, 90° azimuth; `H90e180a.wav` = 90° elevation, 180° azimuth.

**Coordinate interpretation (MIT KEMAR convention):**
   - **Straight ahead:** Azimuth **0°** (file suffix `e000a`) = sound source in front of the listener.
   - **Azimuth:** Horizontal angle around the listener. 0° = front, 90° = right ear side, 180° = directly behind. The dataset uses 0°–180° (one hemisphere; the other side is symmetric).
   - **Elevation:** Vertical angle from the horizontal plane. 0° = ear level (horizontal). Positive = above the listener (e.g. +90° = directly above), negative = below (e.g. -40° = the lowest elevation in this set).

**45° to the left:** The dataset only contains 0°–180° (right hemisphere). For **45° to the left**, use the **same file as 45° to the right**: `H0e045a.wav` (0° elevation, 45° azimuth). When rendering, **swap the left and right IR channels** so that the “right-ear” IR is applied to the listener’s left output and vice versa. (Left/right symmetry is assumed.)

**Rendering (binaural):** Convolve a mono source with the left and right impulse responses from the chosen WAV: one convolution for the left output channel, one for the right. Use the IRs as-is for right-side angles; for left-side angles (negative azimuth in a -180°…+180° convention), use the symmetric positive azimuth file and swap L/R as above. In pygmu2, `SpatialPE(method=SpatialHRTF(azimuth=..., elevation=...))` is not yet implemented; you can achieve the same by loading the KEMAR WAV, optionally swapping channels for left-side positions, and using `ConvolvePE` to convolve the mono source with each IR channel to produce stereo.

## Citation

KEMAR data: Copyright 1994 MIT Media Laboratory. Free use with citation:

- Gardner, W. G., and Martin, K. D., "HRTF measurements of a KEMAR dummy-head microphone", MIT Media Lab Perceptual Computing Technical Report #280, 1994.
