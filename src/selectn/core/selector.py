"""
Main interface for the selectN system.

This module provides the main Selector class that orchestrates the process
of selecting representative samples from a collection of documents.
"""
import os
import json
import numpy as np
from typing import List, Dict, Any, Optional, Union, Tuple
from sklearn.manifold import TSNE
from sklearn.preprocessing import normalize
from sklearn.metrics.pairwise import cosine_distances
from selectn.core.feature_extraction import FeatureExtractorBase
from selectn.core.sampler import SamplerBase, ClusteringSampler, DiversitySampler, HybridSampler
from selectn.documents.document import Document, DocumentCollection
import random
from sklearn.cluster import KMeans


class Selector:
    """
    Main class for selecting representative samples from a document collection.
    """
    
    def __init__(self, feature_extractor: FeatureExtractorBase, 
                 sampler: SamplerBase = None, 
                 output_dir: Optional[str] = "./output"):
        """
        Initialize a selector with a feature extractor and sampler.
        
        Args:
            feature_extractor: The feature extractor to use.
            sampler: The sampler to use (defaults to HybridSampler if None).
            output_dir: Directory to save output files.
        """
        self.feature_extractor = feature_extractor
        self.sampler = sampler or HybridSampler()
        self.output_dir = output_dir
        self.vector_representations = None
        self.selected_indices = None
        self.document_collection = None
    
    def process_documents(self, document_collection: DocumentCollection) -> np.ndarray:
        """
        Process documents and extract feature vectors.
        
        Args:
            document_collection: The document collection to process.
            
        Returns:
            A numpy array of feature vectors.
        """
        self.document_collection = document_collection
        
        # Get preprocessed document contents
        preprocessed_contents = document_collection.get_preprocessed_contents()
        
        # Extract features
        self.vector_representations = self.feature_extractor.fit_transform(preprocessed_contents)
        
        return self.vector_representations
    
    def select_samples(self, n_samples: int, **kwargs) -> List[Document]:
        """
        Select n_samples representative samples from the document collection.
        
        Args:
            n_samples: The number of samples to select.
            **kwargs: Additional arguments to pass to the sampler.
            
        Returns:
            A list of selected Document objects.
        """
        if self.vector_representations is None or self.document_collection is None:
            raise ValueError("Documents must be processed before selecting samples.")
        
        if n_samples > len(self.document_collection):
            raise ValueError(f"Cannot select {n_samples} samples from {len(self.document_collection)} documents.")
        
        # Use the sampler to select samples
        self.selected_indices = self.sampler.sample(
            self.vector_representations, n_samples, **kwargs
        )
        
        # Get the selected documents
        selected_documents = [self.document_collection[idx] for idx in self.selected_indices]
        
        return selected_documents
    
    def save_results(self, selected_documents: List[Document], 
                     file_name: str = "selected_samples.json") -> str:
        """
        Save the selected samples to a file.
        
        Args:
            selected_documents: The selected document objects.
            file_name: The name of the output file.
            
        Returns:
            The path to the saved file.
        """
        # Create output directory if it doesn't exist
        os.makedirs(self.output_dir, exist_ok=True)
        
        output_path = os.path.join(self.output_dir, file_name)
        
        # Prepare results dictionary
        results = {
            "n_samples": len(selected_documents),
            "total_documents": len(self.document_collection) if self.document_collection else 0,
            "samples": []
        }
        
        # Add each selected document
        for doc in selected_documents:
            sample_info = {
                "content": doc.content,
                "metadata": doc.metadata
            }
            results["samples"].append(sample_info)
        
        # Write to file
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2)
        
        return output_path
    
    def get_document_similarities(self) -> np.ndarray:
        """
        Calculate pairwise similarities between documents.
        
        Returns:
            A similarity matrix as numpy array.
        """
        if self.vector_representations is None:
            raise ValueError("Documents must be processed first.")
        
        # Normalize vectors
        normalized_vectors = normalize(self.vector_representations)
        
        # Calculate cosine similarity
        similarities = np.dot(normalized_vectors, normalized_vectors.T)
        
        return similarities
    
    def get_cluster_assignments(self, n_clusters: int) -> np.ndarray:
        """
        Get cluster assignments for documents.
        
        Args:
            n_clusters: Number of clusters to form.
            
        Returns:
            An array of cluster labels for each document.
        """
        if self.vector_representations is None:
            raise ValueError("Documents must be processed first.")
        
        kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(self.vector_representations)
        
        return kmeans.labels_
    
    def get_diversity_scores(self) -> np.ndarray:
        """
        Calculate diversity scores for documents.
        
        Returns:
            An array of diversity scores for each document.
        """
        if self.vector_representations is None:
            raise ValueError("Documents must be processed first.")
        
        from sklearn.metrics import pairwise_distances
        
        # Calculate pairwise distances
        distances = pairwise_distances(self.vector_representations)
        
        # For each document, calculate the average distance to other documents
        diversity_scores = np.mean(distances, axis=1)
        
        return diversity_scores
    
    def get_visualization_data(self, max_files=None) -> Dict[str, Any]:
        """
        Get data for visualization.
        
        Args:
            max_files: Maximum number of files to include in visualization.
                       If None, all files are included.
        
        Returns:
            A dictionary with visualization data.
        """
        if self.vector_representations is None or self.document_collection is None:
            raise ValueError("Documents must be processed first.")
        
        # Determine which files to include in visualization
        n_samples = len(self.document_collection)
        if max_files is not None and max_files < n_samples:
            # If we have selected indices, make sure they're included
            if self.selected_indices:
                # Start with selected indices
                include_indices = set(self.selected_indices)
                
                # Add random indices until we reach max_files
                remaining = max_files - len(include_indices)
                if remaining > 0:
                    # Get all indices excluding the already selected ones
                    all_indices = set(range(n_samples)) - include_indices
                    # Add random indices from the remaining set
                    include_indices.update(random.sample(list(all_indices), min(remaining, len(all_indices))))
                
                # Convert to sorted list
                include_indices = sorted(include_indices)
            else:
                # Just take a random sample
                include_indices = sorted(random.sample(range(n_samples), max_files))
            
            # Subset the vector representations
            vector_subset = self.vector_representations[include_indices]
            
            # Create index mapping from original to subset
            index_mapping = {old_idx: new_idx for new_idx, old_idx in enumerate(include_indices)}
            
            # Map selected indices to new positions
            if self.selected_indices:
                selected_indices_mapped = [index_mapping[idx] for idx in self.selected_indices]
            else:
                selected_indices_mapped = []
        else:
            # Use all data
            vector_subset = self.vector_representations
            include_indices = list(range(n_samples))
            selected_indices_mapped = self.selected_indices or []
        
        # Reduce dimensions for visualization
        # Use a smaller perplexity for small datasets
        subset_size = len(include_indices)
        perplexity = min(30, max(5, subset_size // 3))  # Scale perplexity based on sample size
        tsne = TSNE(n_components=2, random_state=0, perplexity=perplexity)
        vectors_2d = tsne.fit_transform(vector_subset)
        
        # Get cluster assignments if we have enough documents
        n_clusters = min(5, subset_size)
        if subset_size == n_samples:
            # Using all data
            cluster_labels = self.get_cluster_assignments(n_clusters)
        else:
            # Need to compute clusters for the subset
            kmeans = KMeans(n_clusters=n_clusters, random_state=0)
            cluster_labels = kmeans.fit_predict(vector_subset)
        
        # Get diversity scores
        if subset_size == n_samples:
            # Using all data
            diversity_scores = self.get_diversity_scores()
        else:
            # Compute diversity scores for the subset
            pairwise_distances = cosine_distances(vector_subset)
            diversity_scores = np.mean(pairwise_distances, axis=1)
        
        # Prepare visualization data
        viz_data = {
            "points": [],
            "selected_indices": selected_indices_mapped
        }
        
        for i, orig_idx in enumerate(include_indices):
            point_data = {
                "x": float(vectors_2d[i, 0]),
                "y": float(vectors_2d[i, 1]),
                "cluster": int(cluster_labels[i]),
                "diversity": float(diversity_scores[i]),
                "selected": i in selected_indices_mapped,
                "metadata": self.document_collection[orig_idx].metadata
            }
            viz_data["points"].append(point_data)
        
        return viz_data
    
    def save_visualization_data(self, viz_data: Optional[Dict[str, Any]] = None,
                               file_name: str = "visualization_data.json") -> str:
        """
        Save visualization data to a file.
        
        Args:
            viz_data: Optional visualization data (if None, will be generated).
            file_name: The name of the output file.
            
        Returns:
            The path to the saved file.
        """
        if viz_data is None:
            viz_data = self.get_visualization_data()
        
        # Create output directory if it doesn't exist
        os.makedirs(self.output_dir, exist_ok=True)
        
        output_path = os.path.join(self.output_dir, file_name)
        
        # Write to file
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(viz_data, f, indent=2)
        
        return output_path