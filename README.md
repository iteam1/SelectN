# 🔍 selectN

*Using AI and NLP techniques to automatically select **N representative code samples** from large corpora.*

---

## ✨ Features

**Multiple Feature Extraction Methods**  
- TF-IDF based features  
- NLP-based features using spaCy  
- Syntactic structure analysis  
- Hybrid approaches combining multiple methods

**Diverse Sampling Strategies**  
- Random sampling (baseline)  
- Clustering-based sampling  
- Diversity-maximizing sampling  
- Hybrid sampling combining multiple strategies  
- Outlier detection for finding anomalous samples

**Visualization Tools**  
- 2D embeddings visualization  
- Document similarity heatmaps  
- Diversity score analysis  
- Interactive HTML visualizations

**Modular Architecture**  
- Extensible for new document types  
- Pluggable feature extraction methods  
- Customizable sampling strategies

---

## ⚙️ Installation

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

---

## 🧪 Usage

### 💻 Command Line Interface

Run `selectN` directly from the terminal:

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

# Select outliers using Isolation Forest (default)
selectn --input-dir /path/to/code/files --n-samples 10 --sampling-method outlier

# Select outliers using Local Outlier Factor
selectn --input-dir /path/to/code/files --n-samples 10 --sampling-method outlier --outlier-method lof

# Select outliers using distance-based method
selectn --input-dir /path/to/code/files --n-samples 10 --sampling-method outlier --outlier-method distance --contamination 0.05

# Show version
selectn --version

# Show help
selectn --help
```

---

### 🧪 Test Data Generation

Generate synthetic test datasets for experimentation:

```bash
# Generate 1000 test text files
selectn-testgen --output-dir ./test_files --num-files 1000

# Use a different file extension
selectn-testgen --output-dir ./test_files --num-files 500 --extension .log

# Show help
selectn-testgen --help
```

---

### 🐍 Python API

Use `selectN` as a library in your Python projects:

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

---

## 🧩 Architecture Overview

`selectN` is designed with modularity and extensibility in mind.

### 🔧 Core Components
- **Feature Extraction**: Transforms documents into vector embeddings
- **Sampling**: Selects representative samples using various strategies
- **Selector**: Orchestrates the overall document selection pipeline

### 📄 Document Handling
- **Document**: Base class for document abstractions  
- **DocumentCollection**: Manages groups of documents  
- **DocumentFactory**: Dynamically creates document types based on file extension

### 📊 Visualization
- Static plots: `matplotlib`, `seaborn`  
- Interactive HTML: `D3.js`

---

## 🤝 Contributing

We welcome contributions from the community! Please feel free to fork the repo and submit a Pull Request 🚀

---

## 📄 License

This project is licensed under the MIT License. See the [LICENSE](./LICENSE) file for details.