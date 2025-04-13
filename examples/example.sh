#!/bin/bash
# Example shell script to demonstrate SelectN CLI usage

# Set up paths
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
SOURCE_DIR="$PROJECT_ROOT/src/selectn"
OUTPUT_DIR="$PROJECT_ROOT/output"

# Create output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"

echo "SelectN CLI Example"
echo "==================="
echo "Input directory: $SOURCE_DIR"
echo "Output directory: $OUTPUT_DIR"
echo

# Run SelectN with basic settings
echo "Running SelectN to select 3 representative Python files..."
echo

# Run the CLI through Python module (since we haven't installed the package)
python -m src.selectn.cli.cli \
    --input-dir "$SOURCE_DIR" \
    --extensions .py \
    --recursive \
    --n-samples 3 \
    --sampling-method hybrid \
    --feature-method tfidf \
    --dimension-reduction svd \
    --n-components 10 \
    --output-dir "$OUTPUT_DIR" \
    --visualize

echo
echo "Example completed! Check the output directory for results."
echo "Run 'ls -la $OUTPUT_DIR' to see generated files."