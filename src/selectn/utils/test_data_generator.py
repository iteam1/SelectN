"""
Test data generator for selectN.

This module provides functionality to generate random text files
for testing the selectN package.
"""

import os
import random
import string
from typing import List, Optional

# List of topics to generate content about
TOPICS = [
    "machine learning", "artificial intelligence", "data science", 
    "computer vision", "natural language processing", "deep learning",
    "neural networks", "reinforcement learning", "supervised learning",
    "unsupervised learning", "clustering", "classification", "regression",
    "feature extraction", "dimensionality reduction", "transfer learning",
    "generative models", "computer science", "algorithms", "data structures",
    "software engineering", "web development", "mobile development",
    "cloud computing", "distributed systems", "databases", "networking",
    "security", "cryptography", "operating systems", "compilers",
    "programming languages", "python", "java", "javascript", "c++", "go"
]

# List of sentence templates
SENTENCE_TEMPLATES = [
    "The field of {topic} has seen significant growth in recent years.",
    "{topic} is revolutionizing how we approach problem-solving.",
    "Recent advances in {topic} have opened new possibilities.",
    "Researchers in {topic} are making groundbreaking discoveries.",
    "The application of {topic} in industry has led to efficiency gains.",
    "Understanding {topic} requires a solid foundation in mathematics.",
    "The future of {topic} looks promising with new developments.",
    "Companies are investing heavily in {topic} technologies.",
    "The intersection of {topic} and other fields creates new opportunities.",
    "Learning about {topic} can enhance your career prospects.",
    "The challenges in {topic} include scalability and performance.",
    "The history of {topic} dates back to early computing pioneers.",
    "Practical applications of {topic} can be found in everyday technology.",
    "The theoretical foundations of {topic} are based on mathematical principles.",
    "Experts in {topic} are in high demand in the job market."
]

def generate_random_word(min_length: int = 3, max_length: int = 10) -> str:
    """Generate a random word with length between min_length and max_length."""
    length = random.randint(min_length, max_length)
    return ''.join(random.choice(string.ascii_lowercase) for _ in range(length))

def generate_random_sentence(min_words: int = 5, max_words: int = 15) -> str:
    """Generate a random sentence with random words."""
    num_words = random.randint(min_words, max_words)
    words = [generate_random_word() for _ in range(num_words)]
    sentence = ' '.join(words)
    return sentence.capitalize() + '.'

def generate_topic_sentence() -> str:
    """Generate a sentence about a random topic using templates."""
    topic = random.choice(TOPICS)
    template = random.choice(SENTENCE_TEMPLATES)
    return template.format(topic=topic)

def generate_paragraph(min_sentences: int = 3, max_sentences: int = 8) -> str:
    """Generate a paragraph with a mix of random and topic-based sentences."""
    num_sentences = random.randint(min_sentences, max_sentences)
    sentences = []
    
    for _ in range(num_sentences):
        # 70% chance of topic sentence, 30% chance of random sentence
        if random.random() < 0.7:
            sentences.append(generate_topic_sentence())
        else:
            sentences.append(generate_random_sentence())
    
    return ' '.join(sentences)

def generate_document(min_paragraphs: int = 2, max_paragraphs: int = 10) -> str:
    """Generate a document with multiple paragraphs."""
    num_paragraphs = random.randint(min_paragraphs, max_paragraphs)
    paragraphs = [generate_paragraph() for _ in range(num_paragraphs)]
    return '\n\n'.join(paragraphs)

def generate_test_files(
    output_dir: str, 
    num_files: int, 
    file_extension: str = '.txt',
    callback: Optional[callable] = None
) -> List[str]:
    """
    Generate the specified number of test files with random content.
    
    Args:
        output_dir: Directory to save generated files
        num_files: Number of files to generate
        file_extension: File extension for generated files
        callback: Optional callback function to report progress
        
    Returns:
        List of paths to generated files
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    generated_files = []
    
    for i in range(1, num_files + 1):
        filename = f"document_{i:04d}{file_extension}"
        filepath = os.path.join(output_dir, filename)
        
        # Generate random document content
        content = generate_document()
        
        # Write content to file
        with open(filepath, 'w') as f:
            f.write(content)
        
        generated_files.append(filepath)
        
        # Report progress every 100 files
        if i % 100 == 0 and callback:
            callback(i, num_files)
    
    return generated_files
