---

# 📚 Book Recommendation System (Hadoop + Spark + Flask UI)

A scalable Book Recommendation System using:

* **HDFS + Hadoop** for distributed storage
* **Spark** for preprocessing and model training
* **ALS Collaborative Filtering** for personalization
* **TF-IDF Content Recommendation** for similarity search
* **Google Books API** for enriching missing book descriptions
* **Flask Web Interface** for user interaction

---

## 1️⃣ Install Hadoop & HDFS (WSL Users)

Follow this guide to install Hadoop:
[https://dev.to/samujjwaal/hadoop-installation-on-windows-10-using-wsl-2ck1](https://dev.to/samujjwaal/hadoop-installation-on-windows-10-using-wsl-2ck1)

Start Hadoop services:

```bash
start-dfs.sh
start-yarn.sh
```

Verify:

```bash
hdfs dfs -ls /
```

---

## 2️⃣ Download Dataset

Dataset used:
[https://www.kaggle.com/datasets/mohamedbakhet/amazon-books-reviews](https://www.kaggle.com/datasets/mohamedbakhet/amazon-books-reviews)

Extract and upload to HDFS:
In my case the csv files were saved in Downloads, use the filepath where you have extracted the dataset.

```bash
hdfs dfs -mkdir /data
hdfs dfs -mkdir /data/amazon_book_reviews

hdfs dfs -put /mnt/c/Users/shrut/Downloads/Books_rating.csv /data/amazon_book_reviews
hdfs dfs -put /mnt/c/Users/shrut/Downloads/books_data.csv /data/amazon_book_reviews
```

Check:

```bash
hdfs dfs -ls /data/amazon_book_reviews
```

---

## 3️⃣ Python Environment Setup

```bash
python3 -m venv pyenv
source pyenv/bin/activate

pip install --upgrade pip
pip install pyspark nltk pandas numpy scikit-learn tqdm vaderSentiment flask
```

---

## 4️⃣ Google Books API Setup (required for description enrichment)

1. Go to: [https://console.cloud.google.com/apis/credentials](https://console.cloud.google.com/apis/credentials)
2. Create **API Key**
3. Enable **Books API**
4. Save key in your environment:

```bash
echo 'export GOOGLE_BOOKS_API="<YOUR_API_KEY_HERE>"' >> ~/.bashrc
source ~/.bashrc
```

Confirm:

```bash
echo $GOOGLE_BOOKS_API
```

---

## 5️⃣ Data Processing Pipeline (Run in this exact order)

### Step 1: Merge Datasets

```bash
nano merge.py
spark-submit merge.py
```

### Step 2: Preprocess

```bash
nano preprocessv1.py
spark-submit preprocessv1.py
```

### Step 3: Sentiment Analysis

```bash
nano amazon_books_sentiment.py
spark-submit amazon_books_sentiment.py
```

### Step 4: Final Preprocessing for Recommendation Models

```bash
nano finalpreprocess.py
spark-submit finalpreprocess.py
```

---

## 6️⃣ Content Description Enrichment (Run Only Once)

Adds missing descriptions using Google Books API:

```bash
nano enrich_desc.py
spark-submit enrich_desc.py
```

---

## 7️⃣ Run the Recommenders

### Collaborative Filtering (User-Based)

```bash
nano colab-filter.py
spark-submit colab-filter.py
```

### Content-Based Filtering (Book Similarity)

```bash
nano content-filter.py
spark-submit content-filter.py "BOOK_NAME"
```

---

## 8️⃣ Launch Web Interface

```bash
python3 app.py
```

Navigate to:

```
http://localhost:5000
```

---
## Author
Shruti
