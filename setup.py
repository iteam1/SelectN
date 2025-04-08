"""
Setup script for the SelectN package.
"""
from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="selectn",
    version="0.1.0",
    author="SelectN Team",
    author_email="info@selectn.dev",
    description="AI-powered representative code sample selection for ANTLR grammar development",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/SelectN",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.7",
    install_requires=[
        "numpy>=1.19.0",
        "scikit-learn>=0.24.0",
        "matplotlib>=3.3.0",
        "seaborn>=0.11.0",
        "spacy>=3.0.0",
    ],
    extras_require={
        "dev": [
            "pytest>=6.0.0",
            "black>=20.8b1",
            "pylint>=2.6.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "selectn=selectn.cli.cli:main",
        ],
    },
)
