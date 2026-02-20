# Genrify 🎵   
Transformer-Based Lyrics Genre Classification

## Overview
Genrify is a machine learning system that predicts a song’s genre using only lyrical content.  
The project investigates how contextual language models can capture genre-specific linguistic patterns across large music datasets without relying on audio features.

The focus of this work is building and evaluating scalable NLP pipelines for multi-class genre prediction using both classical ML and transformer-based approaches.

---

## Technical Approach

The system implements parallel modeling pipelines to evaluate performance across different representation methods.

### Baseline Modeling
- TF-IDF vectorization for high-dimensional lyric representation  
- Logistic regression classifier for multi-class genre prediction  
- Baseline performance benchmarking for comparison against deep models  

### Transformer-Based Modeling
- DistilBERT encoder for contextual lyric embeddings  
- Fine-tuned classification head for genre prediction  
- PyTorch-based training and evaluation pipeline  
- Comparative evaluation against classical ML baseline  

---

## Dataset
- Spotify songs dataset containing lyrics and genre labels  
- Balanced subset across major genre classes  
- Artist-aware train/validation/test split to reduce leakage  
- Text normalization and tokenization pipeline for NLP training  

---

## Tech Stack
Python, PyTorch, HuggingFace Transformers, scikit-learn, Pandas, NumPy

---

## Repository Structure
```
Genrify/
 ├── src/                # training, preprocessing, evaluation modules
 ├── notebooks/          # exploratory analysis and model experiments
 ├── experiments/        # evaluation outputs and comparisons
 ├── requirements.txt
 └── README.md
```


---

## Author
Arushna Balaganesh  
Computer Engineering @ University of Toronto
