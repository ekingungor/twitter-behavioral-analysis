# Twitter Behavioral Analysis

Türkçe Twitter/X kullanıcılarının davranışsal analizi ve troll hesap tespiti üzerine kapsamlı bir araştırma projesidir. Hesapları ikili olarak (**Troll General** / **Not Troll**) sınıflandırmak için metin, görsel ve profil bazlı öznitelikleri bir arada kullanır.

---

## Genel Bakış

Proje, aynı kullanıcı setini birden fazla öznitelik uzayında temsil eder ve bu temsilleri hem denetimli hem de denetimsiz öğrenme algoritmalarıyla karşılaştırır:

- **Kullanıcı seviyesinde döküman oluşturma**: Her hesabın tüm tweet'leri; orijinal, retweet ve cevap olarak ayrıştırılarak üç ayrı döküman halinde toplanır.
- **Türkçeye özel metin ön işleme**: URL temizliği, emoji çevirisi, Türkçe stop word çıkarımı (NLTK + özel liste), noktalama/aksan normalizasyonu.
- **Çoklu temsil**: BERT gömmeleri, TF-IDF vektörleri, LDA topic dağılımları, reply sentiment skorları, profil fotoğrafı/banner görsel öznitelikleri ve ek hesap metadatası aynı anda değerlendirilir.
- **Bol modelli karşılaştırma**: Tek bir pipeline üzerinde sınıflandırma ve kümeleme algoritmalarının paralel karşılaştırması.

---

## Veri Seti

Pipeline iki kaynaktan beslenir:

| Dosya | İçerik |
|---|---|
| `l_list_tweets.csv` / `l_list_accounts.csv` | "L-list" hesapları ve tweet'leri |
| `random_users_tweets.csv` / `random_users_accounts.csv` | Rastgele örneklenmiş kullanıcılar |
| `all_users_annotations.csv` | Manuel etiketler (`Troll General` / `Not Troll`) — `Bot` ve `Troll Boğaziçi` etiketleri `Troll General` altında birleştirilir |
| `users_extra_features.json` | Hesap bazlı ek sayısal öznitelikler |
| `pp_features.json` / `banner_features.json` | Profil fotoğrafı ve banner görsel gömmeleri |
| `turkish-stop_words.txt` | Türkçeye özel stop word listesi |

Etiketli veri %80 / %10 / %10 oranlarıyla train / validation / test olarak rastgele bölünür ve `train_labels_encoded.json`, `val_labels_encoded.json`, `test_labels_encoded.json` olarak diske yazılır.

---

## Pipeline

### 1. Metin Ön İşleme

Üç farklı yoğunlukta ön işleme fonksiyonu bulunur:

- `pre_process_light` — sadece URL, emoji ve noktalama temizliği
- `pre_process_heavy` — tam temizlik + RT/mention çıkarımı + stop word çıkarımı + küçük harfe çevirme + `i̇` → `i` normalizasyonu
- `pre_process_emojized_heavy` — ağır temizliğe ek olarak emojilerin metin karşılığının korunması

### 2. BERT Gömmeleri

Model: [`dbmdz/distilbert-base-turkish-cased`](https://huggingface.co/dbmdz/distilbert-base-turkish-cased)

Her kullanıcının dökümanı 512 token'lık parçalara bölünür, her parça için pooled output alınır ve ortalaması kullanıcıyı temsil eden 768-boyutlu vektörü verir. Üç farklı dökümanın (orijinal / retweet / reply) gömmeleri birleştirilip **2304 boyutlu** tek bir kullanıcı vektörü oluşturulur.

Bu vektörler üzerinde eğitilen MLP:

```
Input(2304) → Dense(512) → Dense(512) → Dropout(0.3)
            → Dense(256) → Dropout(0.3) → Dropout(0.2)
            → Dense(32)  → Dense(16)   → Dense(1, sigmoid)
```

### 3. Topic Modelling

Gensim LDA (10 topic, 10 pass) tüm kullanıcı dökümanları üzerinde eğitilir. Politik bağlamı baskılamak için `oy, yok, chp, türkiye, türk, kılıçdaroğlu, erdoğan, kemal, allah, seçim, pkk` gibi terimler önceden çıkarılır.

### 4. Reply Sentiment Analizi

Model: [`savasy/bert-base-turkish-sentiment-cased`](https://huggingface.co/savasy/bert-base-turkish-sentiment-cased)

Sadece reply tipi tweet'ler üzerinde çalıştırılır. Her kullanıcıya [-1, +1] aralığında tek bir sentiment skoru atanır. Troll ve normal kullanıcıların dağılımları histogram ve betimleyici istatistiklerle karşılaştırılır.

### 5. Görsel Öznitelikler

Önceden hesaplanmış profil fotoğrafı ve banner gömmeleri birleştirilir; eksik görsel için sıfır-vektörle doldurma yapılır. PCA ile 3 boyuta indirgenip görselleştirilir.

### 6. TF-IDF + Klasik ML

`TfidfVectorizer(norm='l2')` ile kullanıcı dökümanlarından TF-IDF matrisi çıkarılır. Aynı temsil üzerinde eğitilen ve karşılaştırılan modeller:

**Denetimli**
- Naive Bayes
- KNN
- SVM
- Random Forests
- Linear Discriminant Analysis
- Quadratic Discriminant Analysis
- Gradient Boosting
- Neural Networks (MLP)
- Voting Ensemble (SVM + RF + MLP, hard voting)

**Denetimsiz (kümeleme)**
- DBSCAN
- Gaussian Mixture Model
- Agglomerative Clustering

Her model için confusion matrix, precision, recall, F1 ve Adjusted Rand Index raporlanır.

---

## Proje Yapısı

```
twitter-behavioral-analysis/
├── BERT_ekin.ipynb       # Tüm pipeline — orijinal Colab notebook
├── bert_ekin.py          # Notebook'un .py dışa aktarımı
└── README.md
```

Notebook aşağıdaki bölümlere ayrılmıştır:

1. Environment Preparation
2. Data Preparation (File Reading, Data Creation, Text Pre-Processing)
3. Encoded Annotations
4. BERT (Pipeline, Embedding Vectors, Concatenated Vectors, NN Model)
5. Topic Modelling
6. Sentiment Analysis For Replies
7. Image Features
8. TF-IDF (Vectorization, Clustering, Supervised Algorithms)

---

## Bağımlılıklar

- Python 3.9+
- `transformers`, `torch`, `tensorflow`
- `scikit-learn`, `gensim`, `nltk`
- `pandas`, `numpy`, `matplotlib`, `seaborn`
- `emoji`, `joblib`, `tqdm`

NLTK kaynakları: `stopwords`, `punkt`, `wordnet`.

---

## Çalıştırma

Notebook Google Colab'da çalışacak şekilde yazılmıştır ve `/content/drive/MyDrive` altındaki CSV/JSON dosyalarına bağımlıdır. Yerelde çalıştırmak için `HOME_PATH` değişkenini kendi veri dizininizle değiştirmeniz yeterlidir.

```python
HOME_PATH = "/path/to/your/data"
```

---

## Notlar

- Etiket dağılımı dengesizdir; performans karşılaştırmalarında accuracy yerine F1 tercih edilmiştir.
- Sentiment dağılımı karşılaştırması için troll ve normal kümelerinden 100'er örnek alınıp eşlenir.
- Görsel öznitelikler pipeline'a opsiyoneldir; veri eksikse pipeline TF-IDF ve BERT vektörleriyle çalışmaya devam eder.
