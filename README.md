# Financial Sentiment Analysis & Topic Modeling

A comparative NLP project exploring financial sentiment classification and unsupervised topic modeling using the **Financial PhraseBank v1.0** dataset.

**Author:** Haider Rizvi  
**Course:** Deep Learning for Perception (CS4045)  
**Institution:** FAST-NUCES Islamabad

## Project Overview

This project investigates different approaches to understanding financial news text. It combines unsupervised topic modeling with three sentiment analysis approaches and a fine-tuning experiment.

The project includes:

1. **Data Preprocessing & Exploratory Data Analysis**
   - Text cleaning, tokenization, stopword removal, and lemmatization.
   - Analysis of sentiment class distribution and sentence lengths.

2. **Topic Modeling with LDA**
   - Experimentation with 3, 5, 7, and 10 topics.
   - Topic selection using the C_V coherence metric.
   - Assignment and interpretation of dominant topics.

3. **Comparative Sentiment Analysis**
   - Pretrained FinBERT financial sentiment classifier.
   - Zero-shot sentiment classification using OPT-1.3B.
   - Retrieval-Augmented Generation (RAG) using Sentence Transformers, FAISS, and OPT-1.3B.

4. **Model Evaluation**
   - Accuracy, precision, recall, and F1-score.
   - Confusion matrices and class-level performance analysis.
   - Comparison of model behavior, including majority-class prediction.

5. **FinBERT Fine-Tuning**
   - Fine-tuning the pretrained FinBERT model on labeled financial sentences.
   - Evaluation using a stratified training-validation split.
   - Achieved 92.19% validation accuracy.

## Repository Structure

The repository currently contains the following structure:

```text
.
├── dataset/
│   └── FinancialPhraseBank-v1.0/
│       ├── License.txt
│       ├── README.txt
│       ├── Sentences_50Agree.txt
│       ├── Sentences_66Agree.txt
│       ├── Sentences_75Agree.txt
│       └── Sentences_AllAgree.txt
│
├── finbert_finetuned/
│   ├── config.json
│   ├── model.safetensors
│   └── tokenizer files
│
├── finbert_finetuned - Copy/
│
├── rag_checkpoints/
│   ├── corpus_embeddings.npy
│   └── faiss_index.bin
│
├── notebook.ipynb
├── requirements.txt
├── README.md
│
├── all_sentiment_results.csv
├── finbert_results.csv
├── finbert_finetuned_results.csv
├── rag_financial_sentiment_results.csv
│
└── report_i210547_haiderRizvi_deepa3.pdf
```

The `finance_nlp/` directory is a local Python virtual environment and is not required to reproduce the project. It should not be committed to GitHub.

## Technologies Used

- **Programming:** Python
- **Deep Learning:** PyTorch, Hugging Face Transformers
- **NLP:** FinBERT, OPT-1.3B, NLTK
- **Topic Modeling:** Gensim, LDA
- **Semantic Search:** Sentence Transformers (`all-MiniLM-L6-v2`), FAISS
- **Data Processing:** Pandas, NumPy, Scikit-learn
- **Visualization:** Matplotlib, Seaborn
- **Development:** Jupyter Notebook

## Setup & Installation

### 1. Clone the repository

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
cd <YOUR_REPOSITORY_NAME>
```

### 2. Create a virtual environment

```bash
python -m venv .venv
```

Activate it:

**Windows:**
```powershell
.venv\Scripts\activate
```

**Linux/macOS:**
```bash
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

GPU acceleration is recommended for transformer inference and fine-tuning. The notebook's fine-tuning configuration uses mixed precision (`fp16=True`), which requires compatible hardware.

### 4. Run the notebook

Open `notebook.ipynb` using Jupyter Notebook, JupyterLab, or Google Colab.

Run the notebook cells in sequence. The notebook covers preprocessing, topic modeling, sentiment classification, RAG retrieval, model evaluation, and FinBERT fine-tuning.

The dataset is expected at:

```text
dataset/FinancialPhraseBank-v1.0/Sentences_75Agree.txt
```

## Methodology

### 1. Data Preprocessing and EDA

The Financial PhraseBank dataset contains financial news sentences annotated with positive, negative, or neutral sentiment.

The preprocessing pipeline includes:
- Lowercasing.
- Removing punctuation, numbers, and non-alphabetic characters.
- Tokenization.
- English stopword removal.
- Word lemmatization.

Exploratory analysis examines class distribution and sentence-length patterns.

### 2. LDA Topic Modeling

Latent Dirichlet Allocation (LDA) was used to discover recurring themes in the financial text.

Four topic configurations were evaluated: 3, 5, 7, and 10 topics. The five-topic configuration achieved the highest C_V coherence score of **0.4239**.

The resulting topics were interpreted using their most representative keywords, covering themes such as company news, financial performance, market activity, profitability, and product or service announcements.

### 3. Sentiment Classification

Three approaches were compared:

| Approach | Model / Method |
|---|---|
| Pretrained classifier | FinBERT (`yiyanghkust/finbert-tone`) |
| Zero-shot classification | OPT-1.3B (`facebook/opt-1.3b`) |
| Retrieval-augmented classification | Sentence Transformers + FAISS + OPT-1.3B |

The RAG implementation retrieves five semantically similar financial sentences and includes their known sentiment labels as contextual examples in the OPT prompt.

### 4. FinBERT Fine-Tuning

The pretrained FinBERT model initially achieved 87.72% accuracy. Since this was below the assignment's 90% threshold, the model was fine-tuned.

The fine-tuning configuration included:

- 80/20 stratified training-validation split.
- Three training epochs.
- Batch size of 16.
- Learning rate of `2e-5`.
- Maximum sequence length of 128 tokens.
- Mixed-precision training (`fp16=True`).

The fine-tuned model achieved **92.19% validation accuracy**.

## Results

| Approach | Accuracy |
|---|---:|
| Pretrained FinBERT | 87.72% |
| Zero-shot OPT-1.3B | 62.15% |
| RAG + OPT-1.3B | 62.15% |
| Fine-tuned FinBERT | **92.19%** |

### Key Findings

- Pretrained FinBERT provided a strong initial baseline for financial sentiment classification.
- Zero-shot OPT-1.3B predicted every sentence as neutral, resulting in 62.15% accuracy due to the dataset's majority-neutral class distribution.
- The implemented RAG approach did not improve upon the zero-shot OPT-1.3B result.
- Fine-tuning FinBERT improved its validation performance to 92.19%, exceeding the assignment's required accuracy threshold.

Detailed classification reports and confusion matrices are included in the notebook and accompanying report.

## Experimental Limitations

- The OPT-1.3B model exhibited majority-class prediction behavior, highlighting limitations of the zero-shot prompting approach for this task.
- RAG did not improve classification performance in the current experiment. Additionally, the FAISS index was constructed using the full dataset, including examples used during evaluation. Consequently, the RAG results should not be interpreted as an independent held-out generalization estimate.
- The fine-tuned FinBERT result represents validation performance rather than performance on a separate, untouched test set.
- Random seeds were configured for reproducibility, although complete determinism is not guaranteed across hardware and execution environments.

## Project Artifacts

- `notebook.ipynb`: Complete implementation and experiments.
- `finbert_finetuned/`: Fine-tuned model weights and tokenizer configuration.
- `rag_checkpoints/`: Sentence embeddings and FAISS index.
- `*_results.csv`: Saved experiment outputs.
- `report_i210547_haiderRizvi_deepa3.pdf`: Formal assignment report.

## License & Dataset

The project uses the Financial PhraseBank v1.0 dataset. Refer to the dataset's included `License.txt` and `README.txt` for its original usage and licensing conditions.
