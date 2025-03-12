#!/bin/bash
# Source directories
SRC_10='../sptio-temporal-dataset/10'
SRC_11='../sptio-temporal-dataset/11'
SRC_12='../sptio-temporal-dataset/12'

# Destination directories
DIR_10='./10'
DIR_11='./11'
DIR_12='./12'

# Abnormal patterns lookup files
ABNORMAL_10='../sptio-temporal-dataset/AbnormalTracks_10.txt'
ABNORMAL_11='../sptio-temporal-dataset/AbnormalTracks_11.txt'
ABNORMAL_12='../sptio-temporal-dataset/AbnormalTracks_12.txt'

# Function to process abnormal track files and create single-line comma-separated output
process_abnormal_tracks() {
    local input_file=$1
    local output_file=$2

    if [[ ! -f "$input_file" || ! -r "$input_file" ]]; then
        echo "Cannot read input file $input_file"
        return 1
    fi

    local output_dir=$(dirname "$output_file")
    if [[ ! -w "$output_dir" ]]; then
        echo "Cannot write to output directory $output_dir"
        return 1
    fi

    if ! cat "$input_file" | sed 's/\.csv$//' | tr '_' ',' | tr '\n' ',' | sed 's/,$//' >"$output_file"; then
        echo "Failed to process $input_file"
        return 1
    fi

    echo "Created $output_file successfully"
}

# Function to segregate files
segregate_files() {
    local src_dir=$1
    local dest_dir=$2
    local abnormal_list=$3

    # Create normal and abnormal subdirectories
    mkdir -p "$dest_dir/normal" "$dest_dir/abnormal"

    if [[ ! -d "$src_dir" ]]; then
        echo "Source directory $src_dir not found"
        return 1
    fi

    # Copy all files from source to destination root first
    if ! cp "$src_dir"/*.csv "$dest_dir/" 2>/dev/null; then
        echo "No CSV files found in $src_dir or copy failed"
        return 1
    fi

    # Read abnormal numbers into an array
    IFS=',' read -r -a abnormal_nums <"$abnormal_list"

    # Process each file in destination
    for file in "$dest_dir"/*.csv; do
        if [[ ! -f "$file" ]]; then
            echo "No CSV files to process in $dest_dir"
            break
        fi

        local filename=$(basename "$file" .csv)
        local numbers=$(echo "$filename" | tr '_' ' ')
        local is_abnormal=false

        # Check if file contains any abnormal number
        for num in $numbers; do
            for abnormal in "${abnormal_nums[@]}"; do
                if [[ "$num" == "$abnormal" ]]; then
                    is_abnormal=true
                    break 2
                fi
            done
        done

        # Move to exactly one location based on is_abnormal flag
        if [[ "$is_abnormal" == true ]]; then
            mv "$file" "$dest_dir/abnormal/" && echo "Moved $filename.csv to $dest_dir/abnormal"
        else
            mv "$file" "$dest_dir/normal/" && echo "Moved $filename.csv to $dest_dir/normal"
        fi
    done
}

# Create main directories and process abnormal tracks
mkdir -p "$DIR_10" "$DIR_11" "$DIR_12"

# Process abnormal tracks with error checking
for dir in "$DIR_10" "$DIR_11" "$DIR_12"; do
    case "$dir" in
    "$DIR_10")
        process_abnormal_tracks "$ABNORMAL_10" "$dir/process_list.txt" || exit 1
        ;;
    "$DIR_11")
        process_abnormal_tracks "$ABNORMAL_11" "$dir/process_list.txt" || exit 1
        ;;
    "$DIR_12")
        process_abnormal_tracks "$ABNORMAL_12" "$dir/process_list.txt" || exit 1
        ;;
    esac
done

# Segregate files for each dataset
segregate_files "$SRC_10" "$DIR_10" "$DIR_10/process_list.txt"
segregate_files "$SRC_11" "$DIR_11" "$DIR_11/process_list.txt"
segregate_files "$SRC_12" "$DIR_12" "$DIR_12/process_list.txt"
