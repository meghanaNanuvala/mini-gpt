# Mini GPT-Style Language Model

This repository contains an end-to-end implementation of a **mini GPT-style language model** built from scratch. It covers dataset preprocessing, tokenizer training, model design, and evaluation, showcasing the fundamentals of autoregressive text generation using Transformers.


## 📌 Project Overview

- Implemented a **causal language model** (CLM) based on the Transformer decoder architecture.  
- Trained a **custom Byte Pair Encoding (BPE) tokenizer** with a vocabulary size of 20,000.  
- Dataset: **100,000 cleaned English Wikipedia articles** (≈861K training sequences).  
- Frameworks: **TensorFlow/Keras, Keras-NLP, Hugging Face Tokenizers**.  
- Training environment: **Google Colab (for prototyping) + Kaggle GPUs (for full training)**.  
- Achieved **validation perplexity ~100** after 2 epochs (~13M parameters model).  


## 🚀 Features

- **Data preprocessing**: removes wiki markup, cleans text, splits into paragraphs.  
- **Tokenizer training**: trains BPE tokenizer from scratch (20k vocab, BOS/PAD/UNK tokens).  
- **Dataset preparation**: converts text to fixed-length token sequences for CLM training.  
- **Model architecture**:  
  - 4 Transformer decoder layers  
  - 256-dim embeddings, 4 attention heads, 1024 feedforward dim  
  - ~13.45M parameters  
- **Evaluation**:  
  - Quantitative: loss & perplexity  
  - Qualitative: generated text samples  
- **Multi-GPU support** via `tf.distribute.MirroredStrategy`.  


## 📂 Repository Structure


---

## 📊 Results

- **Loss (Epoch 2):** 4.61  
- **Perplexity:** 100.76  
- **Generated Examples:**
  - *Prompt:* "The future of AI"  
    *Output:* repetitive but syntactically fluent text.  
  - *Prompt:* "Albert Einstein was"  
    *Output:* semantically plausible sentences with named entities.  

---

## 🔍 Key Learnings

- Demonstrated the **fundamentals of building GPT-like models** from scratch.  
- Showed trade-offs between **model size vs. compute resources**.  
- Highlighted importance of **tokenization, dataset quality, and decoding strategies**.  

---

## 🛠️ Setup & Usage

1. Clone this repo  
   ```bash
   git clone https://github.com/<your-username>/mini-gpt-meghana.git
   cd mini-gpt-meghana


