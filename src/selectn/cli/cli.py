"""
Command-line interface for selectN.

This module provides a CLI for using the selectN system to select
representative samples from a collection of documents.
"""
import os
import argparse
import logging
from typing import List, Dict, Any, Optional, Union

from selectn.documents.document import DocumentCollection
from selectn.core.feature_extraction import (
    TfidfFeatureExtractor, NLPFeatureExtractor, 
    SyntacticFeatureExtractor, HybridFeatureExtractor
)
from selectn.core.sampler import (
    RandomSampler, ClusteringSampler, 
    DiversitySampler, HybridSampler
)
from selectn.core.selector import Selector
from selectn.utils.visualization import generate_visualization_suite

# Version information
__version__ = "0.2.0"

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def setup_argparse() -> argparse.ArgumentParser:
    """
    Set up argument parser for the CLI.
    
    Returns:
        Configured argument parser.
    """
    parser = argparse.ArgumentParser(
        description="selectN: Select representative samples from document collections"
    )
    
    # Version information
    parser.add_argument(
        '--version', action='version', version=f'selectN v{__version__}',
        help='Show version information and exit'
    )
    
    # Input options
    input_group = parser.add_argument_group("Input Options")
    input_group.add_argument(
        '--input-dir', '-i', type=str, required=True,
        help='Input directory containing documents to process'
    )
    input_group.add_argument(
        '--extensions', '-e', type=str, nargs='+', default=None,
        help='File extensions to include (e.g., .py .java .g4)'
    )
    input_group.add_argument(
        '--recursive', '-r', action='store_true',
        help='Recursively search directories for files'
    )
    input_group.add_argument(
        '--doc-type', '-t', type=str, choices=['code', 'antlr', 'config', 'log'],
        help='Document type (overrides automatic detection)'
    )
    
    # Feature extraction options
    feature_group = parser.add_argument_group("Feature Extraction Options")
    feature_group.add_argument(
        '--feature-method', type=str, 
        choices=['tfidf', 'nlp', 'syntactic', 'hybrid'], 
        default='hybrid',
        help='Feature extraction method'
    )
    feature_group.add_argument(
        '--max-features', type=int, default=None,
        help='Maximum number of features (for applicable methods)'
    )
    feature_group.add_argument(
        '--dimension-reduction', type=str, choices=['pca', 'svd', 'none'],
        default='none',
        help='Dimension reduction method'
    )
    feature_group.add_argument(
        '--n-components', type=int, default=100,
        help='Number of components for dimension reduction'
    )
    
    # Sampling options
    sampling_group = parser.add_argument_group("Sampling Options")
    sampling_group.add_argument(
        '--n-samples', '-n', type=int, required=True,
        help='Number of samples to select'
    )
    sampling_group.add_argument(
        '--sampling-method', type=str, 
        choices=['random', 'clustering', 'diversity', 'hybrid'], 
        default='hybrid',
        help='Sampling method'
    )
    sampling_group.add_argument(
        '--diversity-weight', type=float, default=0.5,
        help='Weight for diversity in hybrid sampling (0-1)'
    )
    
    # Output options
    output_group = parser.add_argument_group("Output Options")
    output_group.add_argument(
        '--output-dir', '-o', type=str, default='./output',
        help='Output directory for results and visualizations'
    )
    output_group.add_argument(
        '--visualize', '-v', action='store_true',
        help='Generate visualizations'
    )
    output_group.add_argument(
        '--max-viz-files', type=int, default=1000,
        help='Maximum number of files to include in visualizations (default: 1000)'
    )
    
    return parser


def create_feature_extractor(args) -> Union[TfidfFeatureExtractor, NLPFeatureExtractor, 
                                           SyntacticFeatureExtractor, HybridFeatureExtractor]:
    """
    Create a feature extractor based on command-line arguments.
    
    Args:
        args: Command-line arguments.
        
    Returns:
        Configured feature extractor.
    """
    dim_reduction = None if args.dimension_reduction == 'none' else args.dimension_reduction
    
    if args.feature_method == 'tfidf':
        return TfidfFeatureExtractor(
            max_features=args.max_features,
            dimension_reduction=dim_reduction,
            n_components=args.n_components
        )
    elif args.feature_method == 'nlp':
        try:
            return NLPFeatureExtractor()
        except ImportError as e:
            logger.warning(f"Error creating NLP feature extractor: {e}")
            logger.warning("Falling back to TF-IDF feature extractor")
            return TfidfFeatureExtractor(
                max_features=args.max_features,
                dimension_reduction=dim_reduction,
                n_components=args.n_components
            )
    elif args.feature_method == 'syntactic':
        return SyntacticFeatureExtractor(
            max_features=args.max_features
        )
    elif args.feature_method == 'hybrid':
        extractors = []
        weights = []
        
        # Add TF-IDF extractor
        extractors.append(TfidfFeatureExtractor(
            max_features=args.max_features,
            dimension_reduction=dim_reduction,
            n_components=args.n_components
        ))
        weights.append(0.5)
        
        # Try to add NLP extractor
        try:
            extractors.append(NLPFeatureExtractor())
            weights.append(0.2)
        except ImportError:
            logger.warning("NLP feature extractor not available")
        
        # Add syntactic extractor
        extractors.append(SyntacticFeatureExtractor(
            max_features=args.max_features
        ))
        weights.append(0.3)
        
        # Normalize weights
        if sum(weights) != 1.0:
            weights = [w / sum(weights) for w in weights]
        
        return HybridFeatureExtractor(extractors, weights)
    
    # Default to TF-IDF
    return TfidfFeatureExtractor(
        max_features=args.max_features,
        dimension_reduction=dim_reduction,
        n_components=args.n_components
    )


def create_sampler(args) -> Union[RandomSampler, ClusteringSampler, DiversitySampler, HybridSampler]:
    """
    Create a sampler based on command-line arguments.
    
    Args:
        args: Command-line arguments.
        
    Returns:
        Configured sampler.
    """
    if args.sampling_method == 'random':
        return RandomSampler()
    elif args.sampling_method == 'clustering':
        return ClusteringSampler()
    elif args.sampling_method == 'diversity':
        return DiversitySampler()
    elif args.sampling_method == 'hybrid':
        return HybridSampler()
    
    # Default to hybrid
    return HybridSampler()


def main():
    """Main CLI entry point."""
    # Parse arguments
    parser = setup_argparse()
    args = parser.parse_args()
    
    # Create feature extractor
    logger.info(f"Creating feature extractor: {args.feature_method}")
    feature_extractor = create_feature_extractor(args)
    
    # Create sampler
    logger.info(f"Creating sampler: {args.sampling_method}")
    sampler = create_sampler(args)
    
    # Create selector
    logger.info("Creating selector")
    selector = Selector(
        feature_extractor=feature_extractor,
        sampler=sampler,
        output_dir=args.output_dir
    )
    
    # Load documents
    logger.info(f"Loading documents from {args.input_dir}")
    document_collection = DocumentCollection()
    document_collection.add_from_directory(
        directory_path=args.input_dir,
        extensions=args.extensions,
        recursive=args.recursive,
        doc_type=args.doc_type
    )
    logger.info(f"Loaded {len(document_collection)} documents")
    
    if len(document_collection) == 0:
        logger.error("No documents found. Please check input directory and extensions.")
        return
    
    if args.n_samples > len(document_collection):
        logger.error(f"Cannot select {args.n_samples} samples from {len(document_collection)} documents")
        return
    
    # Process documents
    logger.info("Processing documents and extracting features")
    selector.process_documents(document_collection)
    
    # Select samples
    logger.info(f"Selecting {args.n_samples} samples")
    selected_documents = selector.select_samples(
        n_samples=args.n_samples,
        diversity_weight=args.diversity_weight
    )
    
    # Save results
    logger.info("Saving results")
    output_path = selector.save_results(selected_documents)
    logger.info(f"Saved results to {output_path}")
    
    # Generate visualizations if requested
    if args.visualize:
        logger.info("Generating visualizations")
        # Check if we need to limit the number of files for visualization
        if len(document_collection) > args.max_viz_files:
            logger.warning(f"Limiting visualization to {args.max_viz_files} files (out of {len(document_collection)})")
            viz_data = selector.get_visualization_data(max_files=args.max_viz_files)
        else:
            viz_data = selector.get_visualization_data()
            
        viz_files = generate_visualization_suite(
            viz_data=viz_data,
            output_dir=args.output_dir
        )
        logger.info(f"Saved visualizations to {args.output_dir}")
        for viz_type, path in viz_files.items():
            logger.info(f"  - {viz_type}: {path}")
    
    logger.info("Done!")


if __name__ == "__main__":
    main()