# Assignment 3 – Topic Modeling and Sentiment Analysis in Financial Text

**Submitted by:** Haider Rizvi
**Roll Number:** I21-0547
**Course:** Deep Learning for Perception (CS4045)

---

## Project Overview

This project implements and evaluates multiple sentiment analysis systems on financial news sentences from the Financial Phrase Bank v1.0 dataset. The pipeline includes:

1. **Data Preprocessing & EDA**
2. **LDA Topic Modeling**
3. **Sentiment Analysis**

   * FinBERT (pre-trained & fine-tuned)
   * Local LLM (OPT-1.3B, zero-shot)
   * RAG + LLM (FAISS-based retrieval)
4. **Evaluation Metrics & Analysis**
5. **Fine-tuning of FinBERT** (to achieve ≥90% accuracy)

---

## Folder Structure

```
.
├── notebooks/
│   ├── 01_data_preprocessing_eda.ipynb
│   ├── 02_lda_topic_modeling.ipynb
│   └── 03_sentiment_analysis.ipynb
├── model_checkpoints/
│   ├── finbert_finetuned/   # Fine-tuned FinBERT weights
│   └── rag_checkpoints/     # Sentence embeddings, FAISS index, results CSV
├── report.pdf               # Formal report (4–8 pages)
└── README.md
```

---

## Setup & Installation

1. **Python Version:** 3.10+
2. **Required Libraries:**

```bash
pip install torch transformers datasets scikit-learn sentence-transformers faiss-cpu tqdm pandas matplotlib seaborn
```

> If using GPU, replace `faiss-cpu` with `faiss-gpu`.

3. **Optional:** Jupyter Notebook or Google Colab for running notebooks interactively.

---

## Usage Instructions

1. **Data Preprocessing & EDA**

   * Open `01_data_preprocessing_eda.ipynb`
   * Run all cells to preprocess data, analyze class distribution, and visualize sentence lengths.

2. **Topic Modeling (LDA)**

   * Open `02_lda_topic_modeling.ipynb`
   * Run all cells to perform LDA topic modeling, generate top keywords, and assign topics to sentences.

3. **Sentiment Analysis**

   * Open `03_sentiment_analysis.ipynb`
   * Ensure `finbert_finetuned/` and `rag_checkpoints/` are in `model_checkpoints/`
   * Run cells for FinBERT, Local LLM, and RAG pipelines.
   * Metrics and confusion matrices will be computed automatically.

---

## Reproducibility

* All random seeds are fixed for deterministic results.
* GPU is recommended for FinBERT fine-tuning (batch size = 16, 3 epochs, mixed precision).
* RAG uses `all-MiniLM-L6-v2` embeddings and FAISS for similarity retrieval.

---

## Results

* **FinBERT (fine-tuned):** 92.19% accuracy
* **Local LLM (OPT-1.3B, zero-shot):** 62.15% accuracy
* **RAG + LLM:** 62.15% accuracy (same as local LLM)

> Refer to `report.pdf` for full evaluation metrics, confusion matrices, and analysis.

---

## Notes

* Only FinBERT was fine-tuned as per the Fine-Tuning Rule (accuracy < 90% before fine-tuning).
* RAG is implemented manually with FAISS retrieval, not using any external RAG pipeline.
* Ensure all dependencies are installed before running notebooks.

---

## Author

Haider Rizvi (I21-0547)
Applied Machine Learning, FAST NUCES Islamabad
