# Hand Trail Gesture Tracker

Track hand movements in 3D space and export as plottable SVG files.

## Quick Start

### 1. Record & Export
```bash
python main.py --rec 'recordings/your_recording.ssrec' --calibration your_calibration
```

### 2. Plot
```bash
./plot_latest.sh
```

Then manually plot each file in Inkscape: Extensions > iDraw > AxiDraw Control > Plot

## Files

- **`main.py`** - Main application with visualization
- **`calibration.py`** - Interactive calibration tool for defining floor polygon
- **`plot_latest.sh`** - Opens latest SVG files for plotting
- **`detectors/trail.py`** - Hand tracking and SVG export logic
- **`PLOTTING_README.md`** - Detailed plotting workflow and troubleshooting

## Directories

- **`recordings/`** - ZED camera recordings (.ssrec files)
- **`calibrations/`** - Saved calibration polygons (.json files)
- **`trails/`** - Exported SVG files organized by timestamp

## Output Structure

```
trails/trail_YYYYMMDD_HHMMSS/
├── all_trails_*.svg          # Combined preview (all people, all hands)
├── person_0_left_*.svg       # Individual hand files for plotting
├── person_0_right_*.svg      #   (one per hand per person)
└── ...
```

## Key Features

- ✅ Fixed scale (800mm × 1200mm) for gesture comparison across sessions
- ✅ A5 page size (210mm × 148mm landscape) optimized for plotting
- ✅ Proximity-based skeleton re-identification (handles temporary tracking loss)
- ✅ Color-coded trails per person
- ✅ 20-second recording duration (configurable)

## Notes

- Only individual hand files (`person_X_left/right_*.svg`) should be plotted
- The `all_trails_*.svg` is for preview/documentation only
- Auto-plotting is disabled due to macOS CLI limitations
