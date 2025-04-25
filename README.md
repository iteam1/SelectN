# selectN

Using AI and NLP techniques to automatically select N representative code samples from large corpora.

## Features

- **Multiple Feature Extraction Methods**:
  - TF-IDF based features
  - NLP-based features using spaCy
  - Syntactic structure analysis
  - Hybrid approaches combining multiple methods

- **Diverse Sampling Strategies**:
  - Random sampling (baseline)
  - Clustering-based sampling
  - Diversity-maximizing sampling
  - Hybrid sampling combining multiple strategies

- **Visualization Tools**:
  - 2D embeddings visualization
  - Document similarity heatmaps
  - Diversity score analysis
  - Interactive HTML visualizations

- **Modular Architecture**:
  - Extensible for new document types
  - Pluggable feature extraction methods
  - Customizable sampling strategies

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/selectN.git
cd selectN

# Install dependencies
pip install -r requirements.txt

# Install spaCy language model (optional, for NLP features)
python -m spacy download en_core_web_sm

# Install the package in development mode
pip install -e .
```

## Usage

### Command Line Interface

`selectN` provides a command-line interface for easy use:

```bash
# Basic usage
selectn --input-dir /path/to/code/files --n-samples 10 --output-dir ./output

# Specifying file extensions
selectn --input-dir /path/to/code/files --extensions .java .py .g4 --n-samples 10

# Using a specific sampling method
selectn --input-dir /path/to/code/files --n-samples 10 --sampling-method diversity

# Generate visualizations
selectn --input-dir /path/to/code/files --n-samples 10 --visualize

# Limit visualization files for large datasets
selectn --input-dir /path/to/code/files --n-samples 10 --visualize --max-viz-files 500

# Show version information
selectn --version

# Show help information
selectn --help
```

### Test Data Generation

selectN includes a tool for generating test data to experiment with the system:

```bash
# Generate 1000 test text files
selectn-testgen --output-dir ./test_files --num-files 1000

# Specify a different file extension
selectn-testgen --output-dir ./test_files --num-files 500 --extension .log

# Show help information
selectn-testgen --help
```

### Python API

You can also use selectN as a library in your Python code:

```python
from selectn.documents.document import DocumentCollection
from selectn.core.feature_extraction import HybridFeatureExtractor
from selectn.core.sampler import HybridSampler
from selectn.core.selector import Selector

# Create a document collection
collection = DocumentCollection()
collection.add_from_directory("/path/to/code/files", extensions=[".java", ".py"])

# Create feature extractor and sampler
feature_extractor = HybridFeatureExtractor()
sampler = HybridSampler()

# Create selector and process documents
selector = Selector(feature_extractor, sampler)
selector.process_documents(collection)

# Select samples
selected_documents = selector.select_samples(n_samples=10)

# Generate visualizations
viz_data = selector.get_visualization_data()
selector.save_visualization_data(viz_data)
```

## Architecture

`selectN` is built with a modular architecture:

### Core Components

- **Feature Extraction**: Transforms documents into vector representations
- **Sampling**: Selects the most representative samples based on various strategies
- **Selector**: Orchestrates the selection process

### Document Handling

- **Document**: Base class for all document types
- **DocumentCollection**: Manages collections of documents
- **DocumentFactory**: Creates appropriate document objects based on file type

### Visualization

- Static visualizations with matplotlib and seaborn
- Interactive visualizations using D3.js

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
