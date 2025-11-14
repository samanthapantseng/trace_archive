# Hand Trail Plotting Workflow

## The Problem

Inkscape's CLI with `--batch-process --actions` crashes on macOS due to fork/exec issues when trying to automate the iDraw extension. This is a known macOS limitation with GUI applications.

## The Solution: Semi-Automated Workflow

### Step 1: Record Hand Movements and Export SVG Files

```bash
python main.py --rec 'path/to/recording.ssrec' --calibration firstChair_test
```

This generates:

-   `trails/trail_YYYYMMDD_HHMMSS/all_trails_YYYYMMDD_HHMMSS.svg` (combined visualization, NOT plotted)
-   `trails/trail_YYYYMMDD_HHMMSS/person_X_left_YYYYMMDD_HHMMSS.svg` (individual hand files)
-   `trails/trail_YYYYMMDD_HHMMSS/person_X_right_YYYYMMDD_HHMMSS.svg` (individual hand files)

### Step 2: Plot the Individual Hand Files

**Run the plotting script:**

```bash
./plot_latest.sh
```

This will:

1. Find the most recent `trails/trail_*` folder
2. Open each individual hand SVG file in Inkscape one at a time
3. Wait for you to manually trigger the plot
4. Prompt for Enter to open the next file

**To plot a specific file manually:**

```bash
open -a Inkscape trails/trail_YYYYMMDD_HHMMSS/person_0_left_YYYYMMDD_HHMMSS.svg
```

### Step 3: Manual Plotting in Inkscape

For each file that opens:

1. Wait for Inkscape to fully load the SVG
2. Go to menu: **Extensions > iDraw > AxiDraw Control**
3. Click **"Plot"** button
4. Wait for plotting to complete
5. Return to terminal and press **Enter** for next file

## File Structure

```
trails/
└── trail_20241114_180244/
    ├── all_trails_20241114_180244.svg        # Combined view (not plotted)
    ├── person_0_left_20241114_180244.svg     # Plot this
    ├── person_0_right_20241114_180244.svg    # Plot this
    ├── person_1_left_20241114_180244.svg     # Plot this (if multiple people)
    └── person_1_right_20241114_180244.svg    # Plot this (if multiple people)
```

## Why Manual Plotting?

**Attempts at full automation all failed due to:**

-   ❌ Direct subprocess.run() from Python → Segmentation fault
-   ❌ subprocess.Popen() with detachment → Segmentation fault
-   ❌ Shell script with clean environment → Segmentation fault
-   ❌ Background processes → Segmentation fault

**Root cause:** The iDraw extension itself tries to spawn child processes, which macOS restricts when called via CLI in batch mode.

## Why Manual Plotting?

**Attempts at full automation all failed due to:**
- ❌ Direct subprocess.run() from Python → Segmentation fault
- ❌ subprocess.Popen() with detachment → Segmentation fault
- ❌ Shell script with clean environment → Segmentation fault
- ❌ Background processes → Segmentation fault

**Root cause:** The iDraw extension itself tries to spawn child processes, which macOS restricts when called via CLI in batch mode.

**Only reliable method:** ✅ Open GUI, manually trigger extension

## Configuration

Edit `detectors/trail.py`:

-   Line 30: `self._duration_seconds = 20` → Change recording duration
-   Line 37: `self._plotter_enabled = False` → Keep disabled (auto-plotting crashes)
-   Lines 314-315: `max_gesture_height_mm` and `max_gesture_width_mm` → Adjust scale

## Troubleshooting

**Q: Inkscape crashes when I click Plot**
A: This is different from the CLI crash. Check your iDraw extension and plotter connection.

**Q: SVG files are empty/blank**
A: Check that hands were visible during recording. Review calibration polygon.

**Q: Scale is too small/large**
A: Adjust `max_gesture_height_mm` and `max_gesture_width_mm` in trail.py lines 314-315.

**Q: Want to re-enable auto-plotting to try again**
A: Don't. It will crash. The semi-automated workflow is the reliable solution.
