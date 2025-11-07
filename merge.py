from pyspark.sql import SparkSession
from pyspark.sql.functions import col, trim, lower, regexp_replace

# Initialize Spark
spark = (
    SparkSession.builder.appName("AmazonBookDataMerge")
    .config("spark.sql.ansi.enabled", "false")
    .getOrCreate()
)

print("=" * 80)
print("📚 AMAZON BOOKS DATA MERGE")
print("=" * 80)

# ======================================================
# 1️⃣ LOAD REVIEWS FILE
# ======================================================
reviews_path = "hdfs:///data/amazon_book_reviews/Books_rating.csv"  # Adjust path
print(f"\n📥 Loading reviews from: {reviews_path}")

reviews = spark.read.csv(reviews_path, header=True, inferSchema=True)
print(f"✅ Loaded {reviews.count()} reviews")
print("\n📋 Reviews schema:")
reviews.printSchema()
print("\n🔍 Sample reviews:")
reviews.show(3, truncate=50)

# Normalize review column names
reviews = (
    reviews.withColumnRenamed("User_id", "user_id")
    .withColumnRenamed("Title", "book_title")
    .withColumnRenamed("review/score", "review_score")
    .withColumnRenamed("review/text", "review_text")
    .withColumnRenamed("review/summary", "review_summary")
    .withColumnRenamed("review/time", "review_time")
    .withColumnRenamed("review/helpfulness", "review_helpfulness")
    .withColumnRenamed("profileName", "profile_name")
    .withColumnRenamed("ProfileName", "profile_name")
)
# Clean book titles for matching
reviews = reviews.withColumn(
    "book_title_clean", lower(trim(regexp_replace(col("book_title"), r"[^\w\s]", "")))
)

# ======================================================
# 2️⃣ LOAD BOOK DETAILS FILE
# ======================================================
books_path = "hdfs:///data/amazon_book_reviews/books_data.csv"  # Adjust path
print(f"\n📥 Loading book details from: {books_path}")

books = spark.read.csv(books_path, header=True, inferSchema=True)
print(f"✅ Loaded {books.count()} books")
print("\n📋 Books schema:")
books.printSchema()
print("\n🔍 Sample books:")
books.show(3, truncate=50)

# Normalize book column names
books = (
    books.withColumnRenamed("Title", "book_title")
    .withColumnRenamed(
        "Descripe", "description"
    )  # Note: "Descripe" typo in your dataset
    .withColumnRenamed("Description", "description")
)

# Clean book titles for matching
books = books.withColumn(
    "book_title_clean", lower(trim(regexp_replace(col("book_title"), r"[^\w\s]", "")))
)
from pyspark.sql.functions import when


def clean_nulls(df, columns):
    for c in columns:
        df = df.withColumn(
            c,
            when(
                (col(c) == "") | (col(c) == "N/A") | (col(c) == "null"), None
            ).otherwise(col(c)),
        )
    return df


reviews = clean_nulls(reviews, ["review_text", "book_title", "review_summary"])
books = clean_nulls(books, ["authors", "categories", "description"])

# Clean book titles for matching
reviews = reviews.withColumn(
    "book_title_clean", lower(trim(regexp_replace(col("book_title"), r"[^\w\s]", "")))
)
# ======================================================
# 3️⃣ MERGE DATASETS
# ======================================================
print(f"\n🔗 Merging reviews with book details...")

# Left join to keep all reviews
merged = reviews.join(
    books.select(
        "book_title_clean",
        col("book_title").alias("book_title_details"),
        "authors",
        "categories",
        "description",
        "publisher",
        "publishedDate",
        "image",
        "previewLink",
        "infoLink",
        "ratingsCount",
    ),
    on="book_title_clean",
    how="left",
)

# Use original title from reviews
merged = merged.drop("book_title_clean", "book_title_details")

print(f"✅ Merged dataset: {merged.count()} rows")

# ======================================================
# 4️⃣ VALIDATE DATA QUALITY
# ======================================================
print("\n🔍 Data Quality Check:")

# Check how many reviews matched with book details
with_authors = merged.filter(col("authors").isNotNull()).count()
print(
    f"   Reviews with author info: {with_authors:,} ({with_authors/merged.count()*100:.1f}%)"
)

with_categories = merged.filter(col("categories").isNotNull()).count()
print(
    f"   Reviews with categories: {with_categories:,} ({with_categories/merged.count()*100:.1f}%)"
)

# Check author field quality
print("\n📊 Sample authors:")
merged.select("authors").filter(col("authors").isNotNull()).distinct().show(
    10, truncate=50
)

# Check categories quality
print("\n📊 Sample categories:")
merged.select("categories").filter(col("categories").isNotNull()).distinct().show(
    10, truncate=50
)

# ======================================================
# 5️⃣ CLEAN AND VALIDATE FIELDS
# ======================================================
print("\n🧹 Cleaning merged data...")

# Clean rating field - ensure it's a valid number between 1-5
merged = merged.withColumn("review_score", col("review_score").cast("float"))

# Remove rows with invalid ratings
before = merged.count()
merged = merged.filter(
    (col("review_score").isNotNull())
    & (col("review_score") >= 1)
    & (col("review_score") <= 5)
)
merged.cache()
after = merged.count()
print(f"   Dropped {before - after:,} rows with invalid ratings")

# Remove rows missing essential fields
merged = merged.filter(
    (col("user_id").isNotNull())
    & (col("book_title").isNotNull())
    & (col("review_score").isNotNull())
)

print(f"   Final row count: {merged.count():,}")

# ======================================================
# 6️⃣ SHOW SAMPLE OF MERGED DATA
# ======================================================
print("\n📋 Sample of merged data:")
merged.select("user_id", "book_title", "review_score", "authors", "categories").show(
    10, truncate=50
)
merged = merged.limit(1000000)
# ======================================================
# 7️⃣ SAVE MERGED DATA
# ======================================================
output_path = "hdfs:///data/amazon_book_reviews/amazon_books_merged.csv"
print(f"\n💾 Saving merged data to: {output_path}")

merged.write.mode("overwrite").csv(output_path, header=True)
print("✅ Merge complete!")

# ======================================================
# 8️⃣ STATISTICS
# ======================================================
print("\n" + "=" * 80)
print("📊 FINAL STATISTICS")
print("=" * 80)
print(f"Total reviews: {merged.count():,}")
print(f"Unique users: {merged.select('user_id').distinct().count():,}")
print(f"Unique books: {merged.select('book_title').distinct().count():,}")
print(f"Reviews with author: {merged.filter(col('authors').isNotNull()).count():,}")
print(
    f"Reviews with category: {merged.filter(col('categories').isNotNull()).count():,}"
)

print("\n⭐ Rating Distribution:")
merged.groupBy("review_score").count().orderBy("review_score").show()

print("\n" + "=" * 80)

spark.stop()
