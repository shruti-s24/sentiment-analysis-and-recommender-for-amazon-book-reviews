---

# 📚 Book Recommendation System (Hadoop + Spark + Flask UI)

**Book Recommendation System** built using:

* **Apache Hadoop + HDFS** for distributed storage
* **Apache Spark** for data processing + model training
* **ALS Collaborative Filtering** for personalized recommendations
* **TF-IDF Content-Based Filtering** for similarity search
* **Google Books API** to enrich missing descriptions
* **Flask Web Interface** for a clean UI

---

## 📦 Dataset Used

We use the Amazon Books Review dataset from Kaggle:

🔗 [https://www.kaggle.com/datasets/mohamedbakhet/amazon-books-reviews](https://www.kaggle.com/datasets/mohamedbakhet/amazon-books-reviews)

Download and extract:

```bash
mkdir -p ~/datasets/books
unzip amazon-books-reviews.zip -d ~/datasets/books
```

Upload to HDFS:

```bash
hdfs dfs -mkdir -p /data/amazon_book_reviews
hdfs dfs -put ~/datasets/books/*.csv /data/amazon_book_reviews
```

---

## 🗂 Required CSVs in HDFS

| File                    | Purpose          |
| ----------------------- | ---------------- |
| Books_rating.csv        | User ratings     |
| books_data.csv          | Book metadata    |
| amazon_books_merged.csv | Combined dataset |

Ensure they exist:

```bash
hdfs dfs -ls /data/amazon_book_reviews
```

---

## 🔐 Google Books API Setup (Required for Description Enrichment)

1. Go to Google Cloud Console
   [https://console.cloud.google.com/apis/credentials](https://console.cloud.google.com/apis/credentials)

2. Create an **API Key**

3. Enable:
   ✅ **Books API**

4. Save the key permanently to your shell profile:

```bash
echo 'export GOOGLE_BOOKS_API="<YOUR_API_KEY_HERE>"' >> ~/.bashrc
source ~/.bashrc
```

Confirm:

```bash
echo $GOOGLE_BOOKS_API
```

This environment variable is automatically used by `enrich_desc.py` and `content-filter.py`.

---

## 🖥️ Python Environment Setup

```bash
python3 -m venv pyenv
source pyenv/bin/activate

pip install --upgrade pip
pip install pyspark nltk pandas numpy scikit-learn tqdm vaderSentiment flask
```

---

## 🧱 Hadoop + HDFS Setup (WSL Users)

Follow this guide:

🔗 [https://dev.to/samujjwaal/hadoop-installation-on-windows-10-using-wsl-2ck1](https://dev.to/samujjwaal/hadoop-installation-on-windows-10-using-wsl-2ck1)

Start Hadoop:

```bash
start-dfs.sh
start-yarn.sh
```

Check:

```bash
hdfs dfs -ls /
```

---

## 🔀 Data Processing Pipeline

### 1️⃣ Merge

```bash
spark-submit merge.py
```

### 2️⃣ Preprocess

```bash
spark-submit preprocessv1.py
```

### 3️⃣ Sentiment Scoring

```bash
spark-submit amazon_books_sentiment.py
```

### 4️⃣ Final Cleanup

```bash
spark-submit finalpreprocess.py
```

---

## 🤝 Collaborative Filtering

```bash
spark-submit colab-filter.py
```

---

## 📖 Content-Based Filtering

### (Run once to enrich descriptions using Google Books API)

```bash
spark-submit enrich_desc.py
```

### Run recommender:

```bash
spark-submit content-filter.py "Harry Potter"
```

---

## 🌐 Web UI

```bash
python3 app.py
```

Open in browser:

```
http://localhost:5000
```

---

## 📂 Project Structure

```
├── app.py                        # Flask UI
├── colab-filter.py               # Collaborative filtering model
├── content-filter.py             # Content-based recommendations
├── enrich_desc.py                # Google Books description enrichment
├── preprocessv1.py
├── amazon_books_sentiment.py
├── finalpreprocess.py
├── merge.py
├── templates/                    # UI HTML
├── logs/                         # Saved logs
└── results/                      # Stored responses
```

---

## Author

Shruti S

If you found this useful, ⭐ star the repo!
