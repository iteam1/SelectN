"""
Feature extraction functionality for SelectN.

This module provides abstract base classes and implementations for extracting
features from different types of documents.
"""
import abc
import spacy
import numpy as np
from typing import List, Dict, Any, Optional, Union
from sklearn.preprocessing import normalize
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer


class FeatureExtractorBase(abc.ABC):
    """Base class for all feature extractors."""

    @abc.abstractmethod
    def fit(self, documents: List[str], **kwargs) -> None:
        """
        Fit the feature extractor to the given documents.
        
        Args:
            documents: The documents to fit on.
            **kwargs: Additional arguments specific to the extractor.
        """
        pass
    
    @abc.abstractmethod
    def transform(self, documents: List[str], **kwargs) -> np.ndarray:
        """
        Transform the given documents into feature vectors.
        
        Args:
            documents: The documents to transform.
            **kwargs: Additional arguments specific to the extractor.
            
        Returns:
            A numpy array of feature vectors.
        """
        pass
    
    def fit_transform(self, documents: List[str], **kwargs) -> np.ndarray:
        """
        Fit and transform the given documents.
        
        Args:
            documents: The documents to fit and transform.
            **kwargs: Additional arguments specific to the extractor.
            
        Returns:
            A numpy array of feature vectors.
        """
        self.fit(documents, **kwargs)
        return self.transform(documents, **kwargs)


class TfidfFeatureExtractor(FeatureExtractorBase):
    """Feature extractor using TF-IDF."""
    
    def __init__(self, max_features: Optional[int] = None, 
                 dimension_reduction: Optional[str] = None,
                 n_components: int = 100):
        """
        Initialize the TF-IDF feature extractor.
        
        Args:
            max_features: Maximum number of features to extract.
            dimension_reduction: Method for dimension reduction. Options: 'pca', 'svd', None.
            n_components: Number of components for dimension reduction.
        """
        self.max_features = max_features
        self.dimension_reduction = dimension_reduction
        self.n_components = n_components
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            token_pattern=r'(?u)\b\w+\b|[^\w\s]'  # Includes punctuation as tokens
        )
        self.reducer = None
        
    def fit(self, documents: List[str], **kwargs) -> None:
        """
        Fit the TF-IDF vectorizer and dimension reducer to the documents.
        
        Args:
            documents: The documents to fit on.
        """
        X = self.vectorizer.fit_transform(documents)
        
        if self.dimension_reduction == 'pca':
            self.reducer = PCA(n_components=min(self.n_components, X.shape[1]))
            self.reducer.fit(X.toarray())
        elif self.dimension_reduction == 'svd':
            self.reducer = TruncatedSVD(n_components=min(self.n_components, X.shape[1]))
            self.reducer.fit(X)
    
    def transform(self, documents: List[str], **kwargs) -> np.ndarray:
        """
        Transform the documents using TF-IDF and optional dimension reduction.
        
        Args:
            documents: The documents to transform.
            
        Returns:
            A numpy array of feature vectors.
        """
        X = self.vectorizer.transform(documents)
        
        if self.reducer is not None:
            if self.dimension_reduction == 'pca':
                return self.reducer.transform(X.toarray())
            else:
                return self.reducer.transform(X)
        
        return X.toarray() if hasattr(X, 'toarray') else X


class NLPFeatureExtractor(FeatureExtractorBase):
    """Feature extractor using spaCy NLP."""
    
    def __init__(self, model: str = "en_core_web_sm"):
        """
        Initialize the NLP feature extractor.
        
        Args:
            model: The spaCy model to use.
        """
        try:
            self.nlp = spacy.load(model)
        except OSError:
            raise ImportError(f"Spacy model '{model}' not found. Please install it using: "
                             f"python -m spacy download {model}")
    
    def fit(self, documents: List[str], **kwargs) -> None:
        """
        No fitting needed for spaCy models.
        
        Args:
            documents: The documents to fit on.
        """
        pass
    
    def transform(self, documents: List[str], **kwargs) -> np.ndarray:
        """
        Transform the documents using spaCy embeddings.
        
        Args:
            documents: The documents to transform.
            
        Returns:
            A numpy array of feature vectors.
        """
        vectors = []
        for doc in self.nlp.pipe(documents, batch_size=32):
            # Use doc vector if available, otherwise average token vectors
            if doc.has_vector:
                vectors.append(doc.vector)
            else:
                # Average the token vectors, ignoring tokens without vectors
                token_vectors = [token.vector for token in doc if token.has_vector]
                if token_vectors:
                    vectors.append(np.mean(token_vectors, axis=0))
                else:
                    # Fallback to zeros if no token has a vector
                    vectors.append(np.zeros(self.nlp.vocab.vectors.shape[1]))
        
        return np.array(vectors)


class SyntacticFeatureExtractor(FeatureExtractorBase):
    """
    Feature extractor for syntactic structures in code.
    Focuses on extracting features related to ANTLR grammar development.
    """
    
    def __init__(self, ngram_range=(1, 3), max_features: Optional[int] = None):
        """
        Initialize the syntactic feature extractor.
        
        Args:
            ngram_range: The range of n-grams to extract.
            max_features: Maximum number of features to extract.
        """
        self.ngram_range = ngram_range
        self.max_features = max_features
        self.vectorizer = CountVectorizer(
            ngram_range=ngram_range,
            max_features=max_features,
            token_pattern=r'(?u)\b\w+\b|[^\w\s]'  # Includes punctuation as tokens
        )
    
    def fit(self, documents: List[str], **kwargs) -> None:
        """
        Fit the vectorizer to the documents.
        
        Args:
            documents: The documents to fit on.
        """
        # Preprocess documents to highlight syntactic structures
        processed_docs = self._preprocess_for_syntax(documents)
        self.vectorizer.fit(processed_docs)
    
    def transform(self, documents: List[str], **kwargs) -> np.ndarray:
        """
        Transform the documents into syntactic feature vectors.
        
        Args:
            documents: The documents to transform.
            
        Returns:
            A numpy array of feature vectors.
        """
        processed_docs = self._preprocess_for_syntax(documents)
        X = self.vectorizer.transform(processed_docs)
        return X.toarray()
    
    def _preprocess_for_syntax(self, documents: List[str]) -> List[str]:
        """
        Preprocess documents to highlight syntactic structures.
        
        Args:
            documents: The documents to preprocess.
            
        Returns:
            List of preprocessed documents.
        """
        processed = []
        for doc in documents:
            # Add special markers for syntax elements common in code and grammars
            doc = doc.replace('{', ' OPEN_BRACE ')
            doc = doc.replace('}', ' CLOSE_BRACE ')
            doc = doc.replace('[', ' OPEN_BRACKET ')
            doc = doc.replace(']', ' CLOSE_BRACKET ')
            doc = doc.replace('(', ' OPEN_PAREN ')
            doc = doc.replace(')', ' CLOSE_PAREN ')
            doc = doc.replace(';', ' SEMICOLON ')
            doc = doc.replace(':', ' COLON ')
            doc = doc.replace('.', ' DOT ')
            doc = doc.replace('=', ' EQUALS ')
            processed.append(doc)
        return processed


class HybridFeatureExtractor(FeatureExtractorBase):
    """
    Hybrid feature extractor combining multiple types of features.
    """
    
    def __init__(self, extractors: List[FeatureExtractorBase], weights: Optional[List[float]] = None):
        """
        Initialize the hybrid feature extractor.
        
        Args:
            extractors: List of feature extractors to combine.
            weights: Optional weights for each extractor (must sum to 1).
        """
        self.extractors = extractors
        
        if weights is None:
            # Equal weights by default
            self.weights = [1.0 / len(extractors)] * len(extractors)
        else:
            if len(weights) != len(extractors):
                raise ValueError("Number of weights must match number of extractors")
            if abs(sum(weights) - 1.0) > 1e-10:
                raise ValueError("Weights must sum to 1")
            self.weights = weights
    
    def fit(self, documents: List[str], **kwargs) -> None:
        """
        Fit all extractors to the documents.
        
        Args:
            documents: The documents to fit on.
        """
        for extractor in self.extractors:
            extractor.fit(documents, **kwargs)
    
    def transform(self, documents: List[str], **kwargs) -> np.ndarray:
        """
        Transform the documents using all extractors and combine the results.
        
        Args:
            documents: The documents to transform.
            
        Returns:
            A numpy array of combined feature vectors.
        """
        # Get features from each extractor
        all_features = []
        for i, extractor in enumerate(self.extractors):
            features = extractor.transform(documents, **kwargs)
            
            # Normalize features to have unit L2 norm
            features_normalized = normalize(features)
            
            # Apply weight
            features_weighted = features_normalized * self.weights[i]
            
            all_features.append(features_weighted)
        
        # Concatenate all features
        # Note: This assumes all feature matrices have the same number of rows (documents)
        combined_features = np.hstack(all_features)
        
        return combined_features