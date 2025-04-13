"""
Document handling functionality for selectN.

This module provides abstract base classes and implementations for different
types of documents that can be processed by selectN.
"""
import os
import re
import abc
from typing import List, Dict, Any, Optional, Union, Tuple, Callable


class Document(abc.ABC):
    """Base class for all document types."""

    def __init__(self, content: str, metadata: Optional[Dict[str, Any]] = None):
        """
        Initialize a document.
        
        Args:
            content: The raw content of the document.
            metadata: Optional metadata associated with the document.
        """
        self.content = content
        self.metadata = metadata or {}

    @abc.abstractmethod
    def preprocess(self) -> str:
        """
        Preprocess the document content for feature extraction.
        
        Returns:
            Preprocessed document content.
        """
        pass

    def __str__(self) -> str:
        """String representation of the document."""
        return f"{self.__class__.__name__}({len(self.content)} chars)"
    
    def __repr__(self) -> str:
        """Detailed string representation of the document."""
        return f"{self.__class__.__name__}(size={len(self.content)}, metadata={self.metadata})"


class CodeDocument(Document):
    """Document class for code files."""
    
    def preprocess(self) -> str:
        """
        Preprocess code content for feature extraction.
        
        Returns:
            Preprocessed code content.
        """
        # Basic preprocessing for code documents
        # This is a simple example - real implementation would be more sophisticated        
        # Simple preprocessing: normalize whitespace and preserve structure
        
        lines = self.content.split('\n')
        processed_lines = []
        
        for line in lines:

            # Skip empty lines
            if not line.strip():
                continue

            processed_lines.append(line.strip())
        
        return '\n'.join(processed_lines)
    

class ConfigDocument(Document):
    """Document class for configuration files."""
    
    def preprocess(self) -> str:
        """
        Preprocess configuration file content for feature extraction.
        
        Returns:
            Preprocessed configuration content.
        """
        # Basic preprocessing for config documents
        # This is just an example and would be extended based on config file type
        
        # Simple preprocessing: normalize whitespace and preserve structure
        lines = self.content.split('\n')
        processed_lines = []
        
        for line in lines:
            # Skip empty lines and comments
            if not line.strip() or line.strip().startswith('#'):
                continue
                
            processed_lines.append(line.strip())
        
        return '\n'.join(processed_lines)


class LogDocument(Document):
    """Document class for log files."""
    
    def preprocess(self) -> str:
        """
        Preprocess log file content for feature extraction.
        
        Returns:
            Preprocessed log content.
        """
        # For logs, we might want to focus on log patterns, not specific values
        # This is a simple example; real implementation would be more sophisticated
        
        lines = self.content.split('\n')
        processed_lines = []
        
        for line in lines:
            # Skip empty lines
            if not line.strip():
                continue
                
            # Replace numeric values with a placeholder
            line = re.sub(r'\b\d+\b', 'NUM', line)
            
            # Replace UUIDs, IPs, etc. with placeholders
            line = re.sub(r'\b[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\b', 'UUID', line, flags=re.IGNORECASE)
            line = re.sub(r'\b(?:\d{1,3}\.){3}\d{1,3}\b', 'IP_ADDR', line)
            
            processed_lines.append(line.strip())
        
        return '\n'.join(processed_lines)


class DocumentFactory:
    """Factory class for creating appropriate document objects."""
    
    @staticmethod
    def create_document(content: str,
                        file_path: Optional[str] = None, 
                        doc_type: Optional[str] = None, 
                        metadata: Optional[Dict[str, Any]] = None) -> Document:
        """
        Create a document of the appropriate type based on file path or explicit type.
        
        Args:
            content: The raw content of the document.
            file_path: Optional path to the file, used to infer document type.
            doc_type: Optional explicit document type.
            metadata: Optional metadata for the document.
            
        Returns:
            A Document object of the appropriate type.
        """
        # Use explicit type if provided
        if doc_type:
            doc_type = doc_type.lower()

        # Otherwise infer from file extension
        elif file_path:
            _, ext = os.path.splitext(file_path)
            ext = ext.lower()
            
            if ext in ['.py', '.java', '.cpp', '.c', '.h', '.js', '.ts', '.go', '.rs', '.php', '.rb']:
                doc_type = 'code'
            elif ext in ['.json', '.yaml', '.yml', '.toml', '.ini', '.conf', '.xml']:
                doc_type = 'config'
            elif ext in ['.log']:
                doc_type = 'log'
            else:
                # Default to code document
                doc_type = 'code'
        else:
            # Default to code document
            doc_type = 'code'
        
        # Set metadata
        if metadata is None:
            metadata = {}
        
        if file_path:
            metadata['file_path'] = file_path
            metadata['file_name'] = os.path.basename(file_path)
        
        # Create document of appropriate type
        if doc_type == 'config':
            return ConfigDocument(content, metadata)
        elif doc_type == 'log':
            return LogDocument(content, metadata)
        else:
            # Default to code document
            return CodeDocument(content, metadata)
        

class DocumentCollection:
    """A collection of documents to be processed by selectN."""
    
    def __init__(self):
        """Initialize an empty document collection."""
        self.documents: List[Document] = []
        
    def add_document(self, document: Document) -> None:
        """
        Add a document to the collection.
        
        Args:
            document: The document to add.
        """
        self.documents.append(document)
    
    def add_documents(self, documents: List[Document]) -> None:
        """
        Add multiple documents to the collection.
        
        Args:
            documents: The documents to add.
        """
        self.documents.extend(documents)
    
    def add_from_file(self,
                      file_path: str,
                      doc_type: Optional[str] = None,
                      metadata: Optional[Dict[str, Any]] = None) -> None:
        """
        Add a document from a file.
        
        Args:
            file_path: Path to the file.
            doc_type: Optional explicit document type.
            metadata: Optional metadata for the document.
        """
        with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
            content = f.read()
        
        document = DocumentFactory.create_document(
            content=content,
            file_path=file_path,
            doc_type=doc_type,
            metadata=metadata
        )
        self.add_document(document)
    
    def add_from_directory(self,
                           directory_path: str,
                           extensions: Optional[List[str]] = None,
                           recursive: bool = True,
                           doc_type: Optional[str] = None) -> None:
        """
        Add documents from all files in a directory.
        
        Args:
            directory_path: Path to the directory.
            extensions: Optional list of file extensions to include.
            recursive: Whether to search recursively.
            doc_type: Optional explicit document type for all files.
        """
        for root, _, files in os.walk(directory_path):
            for file in files:
                file_path = os.path.join(root, file)
                
                # Check extension if specified
                if extensions:
                    _, ext = os.path.splitext(file)
                    if ext.lower() not in extensions:
                        continue
                
                try:
                    self.add_from_file(file_path, doc_type)
                except Exception as e:
                    print(f"Error loading {file_path}: {e}")
            
            # If not recursive, break after first iteration
            if not recursive:
                break
    
    def get_contents(self) -> List[str]:
        """
        Get the raw contents of all documents.
        
        Returns:
            List of document contents.
        """
        return [doc.content for doc in self.documents]
    
    def get_preprocessed_contents(self) -> List[str]:
        """
        Get the preprocessed contents of all documents.
        
        Returns:
            List of preprocessed document contents.
        """
        return [doc.preprocess() for doc in self.documents]
    
    def __len__(self) -> int:
        """Get the number of documents in the collection."""
        return len(self.documents)
    
    def __getitem__(self, idx: int) -> Document:
        """Get a document by index."""
        return self.documents[idx]