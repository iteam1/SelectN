#!/usr/bin/env python3
"""
Command-line interface for generating test data.
This module provides a CLI for generating random text files for testing selectN.
"""

import argparse
import os
import sys
from selectn.utils.test_data_generator import generate_test_files

def main():
    """Main CLI entry point for test data generation."""
    parser = argparse.ArgumentParser(description="Generate random text files for testing selectN")
    parser.add_argument("--output-dir", "-o", type=str, default="./test_files",
                        help="Directory to save generated files (default: ./test_files)")
    parser.add_argument("--num-files", "-n", type=int, default=2000,
                        help="Number of files to generate (default: 2000)")
    parser.add_argument("--extension", "-e", type=str, default=".txt",
                        help="File extension for generated files (default: .txt)")
    
    args = parser.parse_args()
    
    # Define progress callback
    def progress_callback(current, total):
        print(f"Generated {current}/{total} files")
    
    try:
        # Generate test files
        files = generate_test_files(
            args.output_dir, 
            args.num_files, 
            args.extension,
            callback=progress_callback
        )
        
        print(f"Successfully generated {args.num_files} files in {args.output_dir}")
        return 0
    except Exception as e:
        print(f"Error generating test files: {e}", file=sys.stderr)
        return 1

if __name__ == "__main__":
    sys.exit(main())
