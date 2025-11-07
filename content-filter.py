import os
import sys
import math
import json
from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    col,
    lower,
    regexp_replace,
    trim,
    length,
    concat_ws,
    when,
)
from pyspark.sql.types import FloatType
from pyspark.ml.feature import Tokenizer, StopWordsRemover, HashingTF, IDF

# ===================== LOGGING BUFFER =====================
log_lines = []


def log(s=""):
    print(s)
    log_lines.append(s)


def write_log(path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(log_lines))


# ===================== SPARK =====================
spark = (
    SparkSession.builder.appName("ContentBasedBookRecommender")
    .config("spark.executor.memory", "3g")
    .config("spark.driver.memory", "3g")
    .config("spark.sql.shuffle.partitions", "20")
    .getOrCreate()
)

log("=" * 80)
log("📚 CONTENT-BASED BOOK RECOMMENDER (TF-IDF + Cosine Similarity)")
log("=" * 80)

# ===================== INPUT =====================
input_path = "hdfs:///data/amazon_book_reviews/books_enriched"

target_title = sys.argv[1] if len(sys.argv) > 1 else None
if not target_title:
    log('\n❌ Usage: spark-submit content-filter.py "Book Title Here"')
    spark.stop()
    sys.exit(1)

log(f"\n🔍 Searching for: {target_title}")

clean_title = (
    target_title.lower()
    .replace(" ", "_")
    .replace("/", "_")
    .replace("\\", "_")
    .replace(":", "_")
)

result_file = f"results/book_{clean_title}.json"
log_file = f"logs/book_{clean_title}.txt"

# ===================== LOAD DATA =====================
df = spark.read.csv(input_path, header=True, inferSchema=True)

df = df.withColumnRenamed("final_description", "description")
books = df.select("book_title", "description").distinct().dropna(subset=["book_title"])

books = books.withColumn(
    "description",
    when(
        (col("description").isNull()) | (length(col("description")) < 20),
        col("book_title"),
    ).otherwise(col("description")),
)

# ===================== FIND TARGET BOOK =====================
target_row = books.filter(lower(col("book_title")) == target_title.lower()).collect()

if not target_row:
    log("\n❌ Book not found in enriched dataset.")
    write_log(log_file)
    print(f"LOG_FILE={log_file}")
    spark.stop()
    sys.exit(1)

log("✅ Book found. Computing similarity...")

# ===================== TEXT PROCESSING =====================
books = books.withColumn(
    "combined_text", concat_ws(" ", col("description"), col("book_title"))
)
books = books.withColumn("text_clean", lower(col("combined_text")))
books = books.withColumn(
    "text_clean", regexp_replace(col("text_clean"), r"[^a-z0-9\s]", " ")
)
books = books.withColumn("text_clean", regexp_replace(col("text_clean"), r"\s+", " "))
books = books.withColumn("text_clean", trim(col("text_clean")))

tokenizer = Tokenizer(inputCol="text_clean", outputCol="words")
words = tokenizer.transform(books)

remover = StopWordsRemover(inputCol="words", outputCol="filtered_words")
filtered = remover.transform(words)

hashing_tf = HashingTF(
    inputCol="filtered_words", outputCol="raw_features", numFeatures=2**18
)
featurized = hashing_tf.transform(filtered)

idf = IDF(inputCol="raw_features", outputCol="tfidf_features")
idf_model = idf.fit(featurized)
tfidf_data = idf_model.transform(featurized).cache()

# ===================== TARGET VECTOR =====================
target_vec = (
    tfidf_data.filter(lower(col("book_title")) == target_title.lower())
    .select("tfidf_features")
    .collect()[0][0]
)


def cosine(v1, v2):
    dot = float(v1.dot(v2))
    norm1 = math.sqrt(v1.dot(v1))
    norm2 = math.sqrt(v2.dot(v2))
    return dot / (norm1 * norm2) if norm1 and norm2 else 0.0


cosine_udf = spark.udf.register(
    "cosine_udf", lambda x: float(cosine(x, target_vec)), FloatType()
)

similar = (
    tfidf_data.withColumn("similarity", cosine_udf(col("tfidf_features")))
    .filter(lower(col("book_title")) != target_title.lower())
    .orderBy(col("similarity").desc())
    .limit(20)
)

results = similar.collect()

# ===================== PRINT & SAVE =====================
log("\n=============================================================")
log("📚 TOP SIMILAR BOOKS")
log("=============================================================")

recommendations = []
for i, row in enumerate(results, 1):
    log(f"{i:2}. {row['similarity']:.3f}  {row['book_title'][:55]}")
    recommendations.append(
        {
            "rank": i,
            "title": row["book_title"],
            "similarity_score": float(row["similarity"]),
        }
    )

os.makedirs("results", exist_ok=True)
with open(result_file, "w", encoding="utf-8") as f:
    json.dump(
        {
            "query": target_title,
            "matched_title": target_row[0]["book_title"],
            "recommendations": recommendations,
        },
        f,
        ensure_ascii=False,
        indent=2,
    )

write_log(log_file)

print(f"LOG_FILE={log_file}")  # UI reads this
print(f"RESULT_FILE={result_file}")  # UI reads this

log("\n✅ CONTENT RECOMMENDATION COMPLETE!")
spark.stop()
