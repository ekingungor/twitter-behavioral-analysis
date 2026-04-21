# Twitter Behavioral Analysis

A research project on the behavioral analysis of Turkish Twitter/X users, focused on **troll account detection**. The pipeline combines textual, visual, and profile-level features to classify accounts as **Troll General** or **Not Troll**.

---

## Overview

The project represents the same user set across multiple feature spaces and compares these representations using both supervised and unsupervised learning:

- **User-level document construction**: every account's tweets are aggregated into three separate documents — originals, retweets, and replies.
- **Turkish-specific text preprocessing**: URL stripping, emoji translation, Turkish stop-word removal (NLTK + custom list), punctuation and diacritic normalization.
- **Multiple representations**: BERT embeddings, TF-IDF vectors, LDA topic distributions, reply sentiment scores, profile picture / banner visual features, and extra account metadata are evaluated in parallel.
- **Broad model comparison**: classification and clustering algorithms are benchmarked side-by-side on a shared pipeline.

---

## Dataset

The pipeline is driven by two sources:

| File | Content |
|---|---|
| `l_list_tweets.csv` / `l_list_accounts.csv` | "L-list" accounts and their tweets |
| `random_users_tweets.csv` / `random_users_accounts.csv` | Randomly sampled users |
| `all_users_annotations.csv` | Manual labels (`Troll General` / `Not Troll`) — `Bot` and `Troll Boğaziçi` labels are merged into `Troll General` |
| `users_extra_features.json` | Per-account numerical side features |
| `pp_features.json` / `banner_features.json` | Profile picture and banner visual embeddings |
| `turkish-stop_words.txt` | Custom Turkish stop-word list |

Labeled data is randomly split into train / validation / test at an **80 / 10 / 10** ratio and persisted as `train_labels_encoded.json`, `val_labels_encoded.json`, and `test_labels_encoded.json`.

---

## Pipeline

### 1. Text Preprocessing

Three preprocessing functions with varying intensity:

- `pre_process_light` — URL, emoji, and punctuation cleanup only
- `pre_process_heavy` — full cleanup + RT/mention removal + stop-word removal + lowercasing + `i̇` → `i` normalization
- `pre_process_emojized_heavy` — heavy cleanup but keeps the textual form of emojis

### 2. BERT Embeddings

Model: [`dbmdz/distilbert-base-turkish-cased`](https://huggingface.co/dbmdz/distilbert-base-turkish-cased)

Each user's document is split into 512-token chunks; pooled outputs are averaged to produce a 768-dim user vector. The embeddings of the three document types (originals / retweets / replies) are concatenated into a single **2304-dim** user vector.

MLP trained on top of these vectors:

```
Input(2304) → Dense(512) → Dense(512) → Dropout(0.3)
            → Dense(256) → Dropout(0.3) → Dropout(0.2)
            → Dense(32)  → Dense(16)   → Dense(1, sigmoid)
```

### 3. Topic Modelling

A Gensim LDA model (10 topics, 10 passes) is trained on all user documents. To dampen political bias, terms such as `oy, yok, chp, türkiye, türk, kılıçdaroğlu, erdoğan, kemal, allah, seçim, pkk` are stripped beforehand.

### 4. Reply Sentiment Analysis

Model: [`savasy/bert-base-turkish-sentiment-cased`](https://huggingface.co/savasy/bert-base-turkish-sentiment-cased)

Run over reply-type tweets only. Each user is assigned a single sentiment score in the [-1, +1] range. Troll vs. normal distributions are compared using histograms and descriptive statistics.

### 5. Visual Features

Pre-computed profile picture and banner embeddings are concatenated; missing images are filled with zero vectors. PCA is applied for 3D visualization.

### 6. TF-IDF + Classical ML

A TF-IDF matrix is extracted from user documents using `TfidfVectorizer(norm='l2')`. Models trained and compared on the same representation:

**Supervised**
- Naive Bayes
- KNN
- SVM
- Random Forests
- Linear Discriminant Analysis
- Quadratic Discriminant Analysis
- Gradient Boosting
- Neural Networks (MLP)
- Voting Ensemble (SVM + RF + MLP, hard voting)

**Unsupervised (clustering)**
- DBSCAN
- Gaussian Mixture Model
- Agglomerative Clustering

Confusion matrices, precision, recall, F1, and Adjusted Rand Index are reported for each model.

---

## Project Structure

```
twitter-behavioral-analysis/
├── BERT_ekin.ipynb       # Full pipeline — original Colab notebook
├── bert_ekin.py          # .py export of the notebook
└── README.md
```

The notebook is organized into the following sections:

1. Environment Preparation
2. Data Preparation (File Reading, Data Creation, Text Pre-Processing)
3. Encoded Annotations
4. BERT (Pipeline, Embedding Vectors, Concatenated Vectors, NN Model)
5. Topic Modelling
6. Sentiment Analysis For Replies
7. Image Features
8. TF-IDF (Vectorization, Clustering, Supervised Algorithms)

---

## Dependencies

- Python 3.9+
- `transformers`, `torch`, `tensorflow`
- `scikit-learn`, `gensim`, `nltk`
- `pandas`, `numpy`, `matplotlib`, `seaborn`
- `emoji`, `joblib`, `tqdm`

NLTK resources: `stopwords`, `punkt`, `wordnet`.

---

## Running

The notebook is authored for Google Colab and expects CSV/JSON files under `/content/drive/MyDrive`. To run locally, point `HOME_PATH` at your own data directory:

```python
HOME_PATH = "/path/to/your/data"
```

---

## Notes

- The label distribution is imbalanced; F1 is preferred over accuracy for model comparison.
- For sentiment comparison, 100 samples are drawn from each of the troll and normal groups and paired.
- Visual features are optional in the pipeline; if image data is missing, the pipeline continues with TF-IDF and BERT vectors alone.
