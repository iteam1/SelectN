"""
Core sampling functionality for SelectN.

This module provides abstract base classes and implementations for sampling
from collections of documents based on various strategies.
"""
import abc
from typing import List, Any, Dict, Optional, Tuple, Set
import numpy as np
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from sklearn.metrics import pairwise_distances


class SamplerBase(abc.ABC):
    """Base class for all sampling strategies."""

    @abc.abstractmethod
    def sample(self, vector_representations: np.ndarray, n_samples: int, **kwargs) -> List[int]:
        """
        Sample n_samples from the given vector representations.
        
        Args:
            vector_representations: The vector representations of the documents.
            n_samples: The number of samples to select.
            **kwargs: Additional arguments specific to the sampling strategy.
            
        Returns:
            A list of indices representing the selected samples.
        """
        pass


class RandomSampler(SamplerBase):
    """Random sampling strategy."""
    
    def sample(self, vector_representations: np.ndarray, n_samples: int, **kwargs) -> List[int]:
        """
        Randomly sample n_samples from the given vector representations.
        
        Args:
            vector_representations: The vector representations of the documents.
            n_samples: The number of samples to select.
            
        Returns:
            A list of indices representing the selected samples.
        """
        if n_samples > len(vector_representations):
            raise ValueError(f"Cannot sample {n_samples} from {len(vector_representations)} documents.")
            
        return list(np.random.choice(len(vector_representations), n_samples, replace=False))


class ClusteringSampler(SamplerBase):
    """Clustering-based sampling strategy."""
    
    def __init__(self, method: str = "kmeans"):
        """
        Initialize the clustering sampler.
        
        Args:
            method: The clustering method to use. Currently supports 'kmeans'.
        """
        self.method = method
        
    def sample(self, vector_representations: np.ndarray, n_samples: int, **kwargs) -> List[int]:
        """
        Sample n_samples from the given vector representations using clustering.
        
        Args:
            vector_representations: The vector representations of the documents.
            n_samples: The number of samples to select.
            
        Returns:
            A list of indices representing the selected samples.
        """
        if n_samples > len(vector_representations):
            raise ValueError(f"Cannot sample {n_samples} from {len(vector_representations)} documents.")
            
        if self.method == "kmeans":
            # Apply KMeans clustering
            kmeans = KMeans(n_clusters=n_samples, random_state=0).fit(vector_representations)
            
            # For each cluster, find the closest point to the centroid
            selected_indices = []
            for i in range(n_samples):
                # Get all points in this cluster
                cluster_indices = np.where(kmeans.labels_ == i)[0]
                
                # Calculate distance to centroid
                distances = np.linalg.norm(
                    vector_representations[cluster_indices] - kmeans.cluster_centers_[i], 
                    axis=1
                )
                
                # Find the closest point
                closest_point_idx = cluster_indices[np.argmin(distances)]
                selected_indices.append(closest_point_idx)
                
            return selected_indices
        else:
            raise ValueError(f"Unsupported clustering method: {self.method}")


class DiversitySampler(SamplerBase):
    """Diversity-based sampling strategy using maximum dissimilarity."""
    
    def sample(self, vector_representations: np.ndarray, n_samples: int, **kwargs) -> List[int]:
        """
        Sample n_samples from the given vector representations to maximize diversity.
        
        Args:
            vector_representations: The vector representations of the documents.
            n_samples: The number of samples to select.
            
        Returns:
            A list of indices representing the selected samples.
        """
        if n_samples > len(vector_representations):
            raise ValueError(f"Cannot sample {n_samples} from {len(vector_representations)} documents.")
            
        # Start with a random sample
        selected_indices = [np.random.randint(0, len(vector_representations))]
        
        # Calculate pairwise distances
        distances = pairwise_distances(vector_representations)
        
        # Select remaining samples
        while len(selected_indices) < n_samples:
            # Calculate minimum distance from each point to any selected point
            min_distances = np.min(distances[selected_indices][:, np.arange(len(vector_representations))], axis=0)
            
            # Exclude already selected points
            mask = np.ones(len(vector_representations), dtype=bool)
            mask[selected_indices] = False
            min_distances[~mask] = -1
            
            # Select the point with the maximum minimum distance
            next_point = np.argmax(min_distances)
            selected_indices.append(next_point)
            
        return selected_indices


class HybridSampler(SamplerBase):
    """Hybrid sampling strategy combining clustering and diversity."""
    
    def sample(self, vector_representations: np.ndarray, n_samples: int, 
               diversity_weight: float = 0.5, **kwargs) -> List[int]:
        """
        Sample n_samples using a hybrid approach of clustering and diversity.
        
        Args:
            vector_representations: The vector representations of the documents.
            n_samples: The number of samples to select.
            diversity_weight: Weight given to diversity vs. clustering (0-1).
            
        Returns:
            A list of indices representing the selected samples.
        """
        if n_samples > len(vector_representations):
            raise ValueError(f"Cannot sample {n_samples} from {len(vector_representations)} documents.")
        
        # Allocate samples to each strategy
        n_cluster_samples = int(n_samples * (1 - diversity_weight))
        n_diversity_samples = n_samples - n_cluster_samples
        
        # Ensure at least one sample for each strategy if both weights are non-zero
        if diversity_weight > 0 and diversity_weight < 1:
            n_cluster_samples = max(1, n_cluster_samples)
            n_diversity_samples = max(1, n_diversity_samples)
            # Readjust to ensure total is n_samples
            if n_cluster_samples + n_diversity_samples > n_samples:
                n_diversity_samples = n_samples - n_cluster_samples
        
        selected_indices = set()
        
        # Get samples from clustering if needed
        if n_cluster_samples > 0:
            cluster_sampler = ClusteringSampler()
            cluster_indices = cluster_sampler.sample(vector_representations, n_cluster_samples)
            selected_indices.update(cluster_indices)
        
        # Get additional samples for diversity if needed
        if n_diversity_samples > 0:
            # If we have already selected some points, we need to account for them
            if selected_indices:
                # Calculate distances to already selected points
                distances = pairwise_distances(vector_representations)
                
                # Remaining indices to choose from
                remaining_indices = list(set(range(len(vector_representations))) - selected_indices)
                
                # For each remaining index, calculate minimum distance to any selected point
                min_distances = []
                for idx in remaining_indices:
                    min_dist = min(distances[idx, selected_idx] for selected_idx in selected_indices)
                    min_distances.append((idx, min_dist))
                
                # Sort by distance (largest first) and select the top n_diversity_samples
                sorted_distances = sorted(min_distances, key=lambda x: x[1], reverse=True)
                diversity_indices = [idx for idx, _ in sorted_distances[:n_diversity_samples]]
                selected_indices.update(diversity_indices)
            else:
                # If no points selected yet, just use the DiversitySampler
                diversity_sampler = DiversitySampler()
                diversity_indices = diversity_sampler.sample(vector_representations, n_diversity_samples)
                selected_indices.update(diversity_indices)
        
        return list(selected_indices)