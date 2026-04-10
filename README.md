# Genrify 🎵
Transformer-Based Lyrics Genre Classification

## Overview

Genrify predicts a song's genre from lyrics alone — no audio features. Fine-tuning DistilBERT on 50,000 songs across 10 genres, the model achieves 5x above random chance, outperforming a TF-IDF baseline and matching prior NLP-only benchmarks in the literature.

Genre is fundamentally multi-modal; rhythm, tempo, and instrumentation carry more defining weight than text. This project establishes how far a lyrics-only transformer approach can push that ceiling — and where it breaks down.

---

## Approach

- **DistilBERT** (66.5M parameters) fine-tuned for sequence classification on lyrics tokenized up to 512 tokens
- Custom 3-layer classification head (768→512→256→10) with ReLU activations and dropout regularization
- Two-phase training: frozen encoder warmup → full end-to-end fine-tuning with AdamW and linear warmup scheduler
- Benchmarked against a TF-IDF + logistic regression baseline to isolate the contribution of contextual embeddings

---

## Data Pipeline

- Sourced from [550K Spotify Songs](https://www.kaggle.com/datasets/serkantysz/550k-spotify-songs-audio-lyrics-and-genres) — heavily imbalanced in the raw form (197K Rock vs 12K Classical); balanced to 5,000 songs/genre across 10 classes
- **Artist-aware** train/val/test split to prevent artist-level leakage — songs by the same artist are never split across sets
- Text normalization, short-lyric filtering, and whitespace cleaning across 34,471 training samples

**Genres:** Blues, Classical, Country, Electronic, Folk, Hip-Hop, Jazz, Pop, R&B, Rock

---

## Findings

- Hip-Hop achieved the strongest per-class performance (precision 0.75, recall 0.81), driven by its distinctive vocabulary — consistent with prior work in lyric-based classification
- Blues/Rock and Country/Folk were the most commonly confused pairs, reflecting shared lyrical themes rather than model failure
- Contextual embeddings from DistilBERT meaningfully outperformed bag-of-words representations, confirming that word order and context carry genre signal beyond lexical features alone

---

## Limitations

Lyrics capture thematic and linguistic patterns but miss the audio signals that most strongly define genre. Extending the pipeline with MFCCs or spectrogram features would be the highest-impact next step.

---

## Tech Stack
Python, PyTorch, HuggingFace Transformers, scikit-learn, Pandas, NumPy

---

## Author
Arushna Balaganesh  
Computer Engineering @ University of Toronto
