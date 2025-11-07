from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    col,
    lower,
    regexp_replace,
    trim,
    count,
    avg,
    desc,
    length,
    when,
)
from pyspark.sql.types import StructType, StructField, StringType
import requests
from tqdm import tqdm
import os

# ==========================================================
# 🚀 SPARK
# ==========================================================
spark = (
    SparkSession.builder.appName("EnrichBookDescriptions")
    .config("spark.executor.memory", "4g")
    .config("spark.driver.memory", "4g")
    .getOrCreate()
)

print("\n" + "=" * 80)
print("📚 ENRICHING BOOK DESCRIPTIONS (Google Books API)")
print("=" * 80 + "\n")

# ==========================================================
# 🔑 API KEY
# ==========================================================
API_KEY = os.getenv("GOOGLE_BOOKS_API_KEY")
if not API_KEY:
    print(
        "❌ ERROR: Missing API key!\nRun:\n   export GOOGLE_BOOKS_API_KEY='YOUR_KEY_HERE'\n"
    )
    spark.stop()
    exit(1)

# ==========================================================
# 📥 LOAD CLEANED DATA
# ==========================================================
input_path = "hdfs:///data/amazon_book_reviews/cleaned_data"
print(f"📥 Loading data from: {input_path}")

ratings_df = spark.read.csv(input_path, header=True, inferSchema=True)

# Keep necessary metadata
books_df = (
    ratings_df.select("book_title", "description")
    .dropna(subset=["book_title"])
    .distinct()
)

# ==========================================================
# 🎯 PRIORITIZE BY POPULARITY
# ==========================================================
print("\n🔍 Selecting top books by review count...")

# ⚠️ Change this number based on API quota: 300–500 recommended
BOOK_LIMIT = 400  # <-- change here to 300–500 as you want

popularity = (
    ratings_df.groupBy("book_title")
    .agg(count("*").alias("review_count"), avg("rating").alias("avg_rating"))
    .orderBy(desc("review_count"), desc("avg_rating"))
    .limit(BOOK_LIMIT)
)

# ✅ Use INNER JOIN so we do NOT expand back to 45k books
books = popularity.join(books_df, "book_title", "inner")

# Clean titles
books = books.withColumn(
    "clean_title", lower(regexp_replace(col("book_title"), r"[^a-zA-Z0-9\s]", ""))
)
books = books.withColumn(
    "clean_title", trim(regexp_replace(col("clean_title"), r"\s+", " "))
)

# ✅ Limit BEFORE collecting to Python list
books_list = books.select("clean_title", "description").limit(BOOK_LIMIT).collect()


# ==========================================================
# 🌐 GOOGLE BOOKS LOOKUP
# ==========================================================
def fetch_desc(title):
    url1 = (
        f"https://www.googleapis.com/books/v1/volumes?q=intitle:{title}&key={API_KEY}"
    )
    try:
        data = requests.get(url1, timeout=4).json()
        desc = data["items"][0]["volumeInfo"].get("description", None)
        if desc and len(desc) > 60:
            return desc
    except:
        pass

    url2 = f"https://www.googleapis.com/books/v1/volumes?q={title}&key={API_KEY}"
    try:
        data = requests.get(url2, timeout=4).json()
        desc = data["items"][0]["volumeInfo"].get("description", None)
        if desc and len(desc) > 60:
            return desc
    except:
        pass

    return None


print("\n🔍 Fetching missing descriptions...\n")

enriched = []
for row in tqdm(books_list, total=len(books_list)):
    title = row["clean_title"]
    existing = row["description"]

    # keep if already good
    if existing and len(str(existing)) > 80:
        enriched.append((title, existing))
        continue

    new_desc = fetch_desc(title)
    enriched.append((title, new_desc))

# ==========================================================
# ✅ CREATE DATAFRAME SAFELY
# ==========================================================
schema = StructType(
    [
        StructField("clean_title", StringType(), True),
        StructField("new_description", StringType(), True),
    ]
)

enriched_df = spark.createDataFrame(enriched, schema=schema)

# ==========================================================
# 🔗 MERGE WITH ORIGINAL
# ==========================================================
print("\n🔗 Applying enriched descriptions...")

final_df = (
    books.join(enriched_df, "clean_title", "left")
    .withColumn(
        "final_description",
        when(col("new_description").isNotNull(), col("new_description")).otherwise(
            col("description")
        ),
    )
    .select("book_title", "final_description")
    .dropDuplicates(["book_title"])
)

print(f"✅ Final enriched rows: {final_df.count():,}")

# ==========================================================
# 💾 SAVE RESULT
# ==========================================================
output_path = "hdfs:///data/amazon_book_reviews/books_enriched"
print(f"\n💾 Saving enriched descriptions to: {output_path}")

final_df.write.mode("overwrite").csv(output_path, header=True)

print("\n✅ ENRICHMENT COMPLETE!")
print("=" * 80)

spark.stop()
