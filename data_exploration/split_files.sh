#!/bin/bash
set -x

split_video() {
    echo "Splitting file..."
    filename="$1"  # Input filename
    length="$2"    # Segment length in seconds

    # Replace backslashes with forward slashes
    filename="${filename//\\//}"

    # Check if the input file exists
    if [ ! -f "$filename" ]; then
        echo "Error: File '$filename' not found."
        return 1
    fi
    echo "Splitting file $filename"

#    # Extract the filename without extension and create the output directory
#    output_dir="PIS_ID_000_SPLIT/${filename%.*}_split"
#    mkdir -p "$output_dir"

    # Run the ffmpeg command to split the video
    ffmpeg -i "$filename" -c:v libx264 -crf 22 -map 0 -segment_time "$length" -g 2 \
    -sc_threshold 0 -force_key_frames "expr:gte(t,n_forced*2)" -reset_timestamps 1 \
    -f segment "$filename%4d.mp4"
}
split_video "$@"