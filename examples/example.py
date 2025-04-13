#!/usr/bin/env python
"""
Basic example of using selectN to select representative code samples.
"""
import os
import sys
import logging

# (since we have installed the package)
from selectn.documents.document import DocumentCollection
from selectn.core.feature_extraction import TfidfFeatureExtractor
from selectn.core.sampler import HybridSampler
from selectn.core.selector import Selector
from selectn.utils.visualization import generate_visualization_suite

# Set up logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    """Run a basic example of selectN."""
    # Path to selectN source code as an example corpus
    input_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../src/selectn'))
    output_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../output'))
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Number of samples to select
    n_samples = 3
    
    logger.info(f"Using input directory: {input_dir}")
    logger.info(f"Using output directory: {output_dir}")
    
    # Create document collection
    logger.info("Creating document collection")
    collection = DocumentCollection()
    collection.add_from_directory(input_dir, extensions=['.py'], recursive=True)
    logger.info(f"Loaded {len(collection)} documents")
    
    # Create feature extractor and sampler
    logger.info("Creating feature extractor and sampler")
    feature_extractor = TfidfFeatureExtractor(dimension_reduction='svd', n_components=10)
    sampler = HybridSampler()
    
    # Create selector
    logger.info("Creating selector")
    selector = Selector(feature_extractor, sampler, output_dir=output_dir)
    
    # Process documents
    logger.info("Processing documents")
    selector.process_documents(collection)
    
    # Select samples
    logger.info(f"Selecting {n_samples} representative samples")
    selected_documents = selector.select_samples(n_samples=n_samples, diversity_weight=0.7)
    
    # Display selected documents
    logger.info("Selected documents:")
    for i, doc in enumerate(selected_documents):
        filename = doc.metadata.get('file_path', f"Document {i}")
        logger.info(f"  - {filename}")
    
    # Save results
    logger.info("Saving results")
    output_path = selector.save_results(selected_documents)
    logger.info(f"Saved results to {output_path}")
    
    # Generate visualizations
    logger.info("Generating visualizations")
    viz_data = selector.get_visualization_data()
    viz_files = generate_visualization_suite(viz_data, output_dir, prefix="test_")
    
    logger.info("Visualization files:")
    for viz_type, path in viz_files.items():
        logger.info(f"  - {viz_type}: {path}")
    
    logger.info("Example completed successfully!")
    
if __name__ == "__main__":
    main()