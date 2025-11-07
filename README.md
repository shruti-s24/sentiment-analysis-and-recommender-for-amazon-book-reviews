# 📚 Book Recommendation System (Hadoop + Spark + Flask UI)

A complete end-to-end **Book Recommender System** built using:

* **HDFS + Hadoop** for distributed storage
* **Apache Spark** for data processing
* **ALS Collaborative Filtering** for personalized recommendations
* **TF-IDF Content-Based Filtering** for similar-book lookup
* **Flask Web Interface** for easy user interaction

---

## 🚀 Features

| Feature                        | Description                                         |
| ------------------------------ | --------------------------------------------------- |
| **Collaborative Filtering**    | Recommends books based on similar users (ALS model) |
| **Content-Based Filtering**    | Recommends similar books based on descriptions      |
| **Sentiment-Enhanced Ratings** | Book ratings boosted by text sentiment              |
| **Interactive Web UI**         | Clean and modern HTML frontend using Flask          |
| **HDFS + Spark**               | Distributed scalable big-data processing            |

---

## 🖥️ System Requirements

* Windows / Linux / Mac
* Python **3.8+**
* Java **8+**
* **Hadoop** + **HDFS**
* **Apache Spark** 3.x

---

## Dataset used
https://www.kaggle.com/datasets/mohamedbakhet/amazon-books-reviews

---

## ⚙️ Hadoop Setup (Windows WSL)

Follow this guide to install Hadoop on WSL:

[https://dev.to/samujjwaal/hadoop-installation-on-windows-10-using-wsl-2ck1](https://dev.to/samujjwaal/hadoop-installation-on-windows-10-using-wsl-2ck1)

---

## 🗄️ HDFS Setup

```bash
hdfs dfs -mkdir /data
hdfs dfs -mkdir /data/amazon_book_reviews

hdfs dfs -put /mnt/c/Users/shrut/Downloads/Books_rating.csv /data/amazon_book_reviews
hdfs dfs -put /mnt/c/Users/shrut/Downloads/books_data.csv /data/amazon_book_reviews
hdfs dfs -put /mnt/c/Users/shrut/Downloads/amazon_books_merged.csv /data/amazon_book_reviews
```

---

## 🐍 Python Environment Setup

```bash
python3 -m venv pyenv
source pyenv/bin/activate

pip install --upgrade pip
pip install pyspark nltk pandas numpy scikit-learn tqdm vaderSentiment
```

---

## 🔀 Data Pipeline

### 1️⃣ Merge Datasets

```bash
nano merge.py
spark-submit merge.py
```

### 2️⃣ Preprocess

```bash
nano preprocessv1.py
spark-submit preprocessv1.py
```

### 3️⃣ Sentiment Analysis

```bash
nano amazon_books_sentiment.py
spark-submit amazon_books_sentiment.py
```

### 4️⃣ Final Preprocessing

```bash
nano finalpreprocess.py
spark-submit finalpreprocess.py
```

---

## 🤝 Collaborative Filtering

```bash
nano colab-filter.py
spark-submit colab-filter.py
```

---

## 📖 Content-Based Filtering

### (Run once to fill missing descriptions)

```bash
nano enrich_desc.py
spark-submit enrich_desc.py
```

### Run Recommender

```bash
nano content-filter.py
spark-submit content-filter.py "BOOK_NAME"
```

---

## 🌐 Web UI

Start the interface:

```bash
python3 app.py
```

Then open:

```
http://localhost:5000
```

---

## 📂 Project Structure

```
├── app.py                       # Flask Web UI
├── colab-filter.py              # Collaborative Filtering Model
├── content-filter.py            # Content-Based Model
├── enrich_desc.py               # One-time description enrichment
├── preprocessv1.py              # Initial cleaning
├── amazon_books_sentiment.py    # Sentiment scoring
├── finalpreprocess.py           # Final dataset formatting
├── merge.py                     # Dataset merging
├── templates/                   # HTML UI files
└── logs/ & results/             # Saved outputs
```

---

## 🎯 Demo Output Examples

| Mode          | Example Result                                                                    |
| ------------- | --------------------------------------------------------------------------------- |
| Collaborative | “Users similar to you loved *The Hobbit*, *Eragon*, *Percy Jackson*...”           |
| Content-Based | “Books similar to *Harry Potter* → *Percy Jackson*, *The Magicians*, *Narnia*...” |

---

## ❤️ Author

Shruti

If you found this useful, ⭐ star the repo!

---
