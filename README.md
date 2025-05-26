# Brand Sentiment Analysis using BERT

> *"Twitter’s Take on Your Brand: Sweet Treat or Bitter Tweet?"*

---

## 🔍 Overview

This project focuses on classifying tweets about various brands into three sentiment categories:
- **Sweet Treat** (Positive)
- **Bitter Tweet** (Negative)
- **Plain Post** (Neutral)

I fine-tuned a pre-trained BERT model on a dataset of brand-related tweets, addressed class imbalance with weighted loss and augmentation, and evaluated performance using comprehensive metrics and used  Weights & Biases (WandB) for tracking.

---

## 📁 Dataset

- Based on a dataset of **brand-related tweets** labeled as positive, negative, or neutral.
- Original dataset: [Dataset](https://www.kaggle.com/datasets/tusharpaul2001/brand-sentiment-analysis-dataset/data).
- Tweets include mentions of popular brands like Apple, Google, Android, etc.

---

## ⚙️ Features & Techniques

### 🧪 Initial Experiments

- **Traditional ML Pipeline**
  - Applied preprocessing: lowercasing, HTML tag removal, retweet and mention removal.
  - Extracted features: number of words, characters, nouns, adjectives, and negative words.
  - Trained ML models: **Logistic Regression** and **Random Forest**.
  - Accuracy stayed below **73%** despite hyperparameter tuning.
  - **F1-scores for minority classes were low**, even with balancing techniques like oversampling and SMOTE.
  - KDE plots showed **no significant correlation** between extracted features and sentiment labels.
  - Removing these features did **not impact performance**, so they were dropped.

---

### 🔄 Shift to BERT-based Modeling

- Switched to **BERT-base-uncased** from Hugging Face Transformers.
- Initially retained previous preprocessing and added **class weights**, resulting in slight F1-score improvement.
- Found that **preprocessing can degrade BERT performance**—removing it led to a **3–4% boost in accuracy and F1-scores**.

---

### 🧠 Final Model Strategy

- **Text Augmentation**  
  Applied NLP techniques using `nlpaug`:
  - Synonym Replacement
  - Random Swap
  - Contextual Word Insertion  
  Augmentation targeted underrepresented classes.

- **Class Imbalance Handling**  
  Used **custom class weights** in the loss function to handle class imbalance.

- **Performance Boost**  
  Combining **augmentation** + **class weights** led to **drastic improvement** in both accuracy and per-class F1-scores.

- **Evaluation**
  - Achieved **~87% accuracy** and **88.6% F1 score** on the test set.
  - Used **macro and weighted F1-scores** for deeper insights.
  - Training and validation tracked with **Weights & Biases (WandB)**.

---

## 📁 Folder Overview

| Folder           | Description                               |
|------------------|-------------------------------------------|
| `bert_model/`     | Contains the fine tuned BERT model       |
| `bert_sentiment/` | Handles data loading, training, evaluation |
| `logs/`           | Logs from training/evaluation             |
| `wandb/`          | Auto-generated W&B logs and metadata      |
| `app.py`          | Implements a Streamlit-based user interface that loads and interacts with the fine-tuned BERT model saved in bert_model     |

---

## 📈 Results

### Test Classification Report

| Class         | Precision | Recall | F1-score |
|---------------|-----------|--------|----------|
| Plain Post    | 0.90      | 0.79   | 0.84     |
| Sweet Treat   | 0.83      | 0.91   | 0.87     |
| Bitter Tweet  | 0.92      | 0.97   | 0.95     |
| **Accuracy**  |           |        | **0.87** |

---

## 🛠️ Setup & Usage

### Requirements

- Python 3.8+
- PyTorch
- Transformers
- Datasets (HuggingFace)
- scikit-learn
- nlpaug
- wandb

### Installation

```bash
pip install -r requirements.txt
