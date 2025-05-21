# Biomedical Abbreviation Detection – NLP Sequence Classification

This repository contains an NLP prototype for detecting **abbreviations (AC)** and their corresponding **long forms (LF)** in biomedical text using **BIO-tagged sequence classification**. The model is designed to label tokens from scientific articles with one of four categories:  
- `B-AC`: Beginning of abbreviation  
- `B-LF`: Beginning of long form  
- `I-LF`: Inside long form  
- `B-O`: Outside (other)  

The dataset is sourced from the [PLOD-CW dataset](https://huggingface.co/datasets/surrey-nlp/PLOD-CW), which contains ~50,000 labelled tokens derived from PLOS biomedical journal articles.

---

## 🔬 Objective

To build and evaluate a sequence classification model that accurately tags biomedical tokens with abbreviation-related BIO labels using a variety of NLP techniques, embedding strategies, and model architectures.

---

## 🚀 Key Components

### 1. Preprocessing Strategies
- **Tokenization-only**: Preserves biomedical structure and terminology
- **Full pipeline**: Includes lemmatization, stopword removal, normalization, and missing data handling
- Observed that simpler preprocessing (tokenization-only) outperformed aggressive cleaning

### 2. Vectorization Methods
- **Word2Vec**: Trained on the dataset for local semantic representation
- **GloVe**: Pre-trained (100d, Wiki-Gigaword) for global co-occurrence context

### 3. Model Architectures
- **Bidirectional LSTM**: Captures sequence and contextual dependencies
- **k-Nearest Neighbors (kNN)**: Uses word embeddings to classify tokens by semantic similarity

### 4. Hyperparameter Tuning
- Explored `k = 5` vs `k = 100` in kNN
- `k = 100` provided better generalisation and F1 score

---

## 📊 Evaluation Highlights

- Models evaluated using **F1-score**, with special focus on minority tags (`B-AC`, `B-LF`, `I-LF`)
- LSTM showed strong performance on frequent tags (`B-O`), while kNN was more balanced across rare tags
- Class imbalance was a significant challenge — future improvements could include loss weighting or data augmentation

---

## 📂 Contents

| File | Description |
|------|-------------|
| `Biomedical_text_processing.ipynb` | Main notebook with all experiments and evaluations |

---

## 💡 Future Directions

- Integrate **transformer-based models** (e.g., BERT) with CRF layers for better tag dependencies
- Apply **focal loss** or **weighted class loss** to improve minority tag learning
- Wrap the best-performing model in a lightweight **API or CLI tool** for real-world usage

---

## 🛠️ Tools & Libraries

- Python 3.10+
- TensorFlow / Keras
- Gensim
- Scikit-learn
- NLTK
- Seaborn / Matplotlib


