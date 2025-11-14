#!/bin/bash
# Open all individual hand SVGs from the most recent trail folder in Inkscape
# Files will open one at a time - manually plot each before proceeding
# Usage: ./plot_latest.sh

TRAILS_DIR="/Users/samanthapan/Documents/senseSpace/senseSpace/students/trace_archive/gesture/trails"

# Find most recent trail folder
LATEST_DIR=$(ls -td "$TRAILS_DIR"/trail_* 2>/dev/null | head -1)

if [ -z "$LATEST_DIR" ]; then
    echo "No trail folders found in $TRAILS_DIR"
    exit 1
fi

echo "================================================"
echo "Opening files from: $(basename "$LATEST_DIR")"
echo "================================================"
echo ""

# Collect files to plot (exclude all_trails)
FILES=()
for svg_file in "$LATEST_DIR"/*.svg; do
    if [[ "$svg_file" != *"all_trails"* ]]; then
        FILES+=("$svg_file")
    fi
done

FILE_COUNT=${#FILES[@]}

if [ "$FILE_COUNT" -eq 0 ]; then
    echo "No individual hand SVG files found"
    exit 1
fi

echo "Found $FILE_COUNT files to plot"
echo ""
echo "WORKFLOW:"
echo "1. Each file will open in Inkscape"
echo "2. Go to: Extensions > iDraw > AxiDraw Control > Plot"
echo "3. After plotting, press Enter for next file"
echo ""
echo "================================================"
echo ""

# Open each file one at a time
COUNT=0
for svg_file in "${FILES[@]}"; do
    COUNT=$((COUNT + 1))
    echo "[$COUNT/$FILE_COUNT] Opening: $(basename "$svg_file")"
    
    # Open in Inkscape
    open -a Inkscape "$svg_file"
    
    # Wait for user to plot and confirm
    if [ $COUNT -lt $FILE_COUNT ]; then
        echo ""
        read -p "Press Enter when ready for next file... "
        echo ""
    fi
done

echo ""
echo "================================================"
echo "All $FILE_COUNT files have been opened"
echo "================================================"
