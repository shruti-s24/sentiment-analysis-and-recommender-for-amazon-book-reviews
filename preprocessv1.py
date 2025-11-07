from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when, length, trim, regexp_replace
from pyspark.sql import Window
from pyspark.sql.functions import row_number, desc

# Initialize Spark session
spark = (
    SparkSession.builder.appName("AmazonBookDataPreprocessor")
    .config("spark.sql.ansi.enabled", "false")
    .getOrCreate()
)

print("=" * 80)
print("🧹 PREPROCESSING FOR SENTIMENT ANALYSIS")
print("=" * 80)

# ------------------------------------------------------
# 1️⃣ Read merged dataset
# ------------------------------------------------------
input_path = "hdfs:///data/amazon_book_reviews/amazon_books_merged.csv"
print(f"\n📥 Loading merged dataset from: {input_path}")

df = spark.read.csv(input_path, header=True, inferSchema=True)
print(f"✅ Loaded {df.count():,} rows")

print("\n📋 Schema:")
df.printSchema()

print("\n🔍 Sample (first 3 rows):")
df.show(3, truncate=60)

# ------------------------------------------------------
# 2️⃣ Normalize column names
# ------------------------------------------------------
print("\n🔤 Normalizing column names...")

column_mapping = {
    "User_id": "user_id",
    "user": "user_id",
    "Title": "book_title",
    "title": "book_title",
    "review/score": "rating",
    "review_score": "rating",
    "review/text": "review_text",
    "review/summary": "review_summary",
    "profileName": "profile_name",
    "ProfileName": "profile_name",
}

for old, new in column_mapping.items():
    if old in df.columns:
        df = df.withColumnRenamed(old, new)
        print(f"   ✓ {old} → {new}")

# ------------------------------------------------------
# 3️⃣ Validate essential columns exist
# ------------------------------------------------------
print("\n✅ Checking for essential columns...")

essential_for_sentiment = ["user_id", "book_title", "review_text", "rating"]
missing = [col for col in essential_for_sentiment if col not in df.columns]

if missing:
    print(f"❌ ERROR: Missing essential columns: {missing}")
    print(f"   Available columns: {df.columns}")
    spark.stop()
    exit(1)

print("✓ All essential columns present")

# ------------------------------------------------------
# 4️⃣ Clean rating values
# ------------------------------------------------------
print("\n⭐ Cleaning rating values...")

# Convert to float
df = df.withColumn("rating", col("rating").cast("float"))

# Show distribution before cleaning
print("\n📊 Rating distribution BEFORE cleaning:")
df.groupBy("rating").count().orderBy("rating").show(20)

# Keep only valid ratings (1.0 to 5.0)
before = df.count()
df = df.filter(
    (col("rating").isNotNull()) & (col("rating") >= 1.0) & (col("rating") <= 5.0)
)
after = df.count()

print(f"✓ Removed {before - after:,} rows with invalid ratings")
print(f"✓ Remaining: {after:,} rows")

print("\n📊 Rating distribution AFTER cleaning:")
df.groupBy("rating").count().orderBy("rating").show()

# ------------------------------------------------------
# 5️⃣ Drop rows with missing essential data
# ------------------------------------------------------
print("\n🔍 Removing rows with missing essential fields...")

before = df.count()
df = df.na.drop(subset=essential_for_sentiment)
after = df.count()

print(f"✓ Removed {before - after:,} rows with missing data")
print(f"✓ Remaining: {after:,} rows")

# ------------------------------------------------------
# 6️⃣ Clean review text
# ------------------------------------------------------
print("\n📝 Cleaning review text...")

# Remove rows with very short reviews (less than 10 characters)
before = df.count()
df = df.filter(length(col("review_text")) >= 10)
after = df.count()

print(f"✓ Removed {before - after:,} rows with too-short reviews")

# Show review length distribution
print("\n📊 Review length statistics:")
df.select("review_text").selectExpr("length(review_text) as len").summary().show()

# ------------------------------------------------------
# 7️⃣ Clean authors and categories (keep brackets for now)
# ------------------------------------------------------
print("\n🧹 Cleaning metadata fields...")

if "authors" in df.columns:
    before_null = df.filter(col("authors").isNull()).count()

    # Remove clearly corrupted authors (too long or containing review-like text)
    df = df.withColumn(
        "authors",
        when(
            (col("authors").isNotNull())
            & (length(col("authors")) < 150)  # Not too long
            & (
                ~col("authors").rlike(
                    "(?i)(review|amazon|page|chapter|purchase|bought|read this)"
                )
            ),
            trim(col("authors")),
        ).otherwise(None),
    )

    after_null = df.filter(col("authors").isNull()).count()
    print(f"✓ Authors: Cleaned {after_null - before_null:,} corrupted entries")

if "categories" in df.columns:
    before_null = df.filter(col("categories").isNull()).count()

    df = df.withColumn(
        "categories",
        when(
            (col("categories").isNotNull())
            & (length(col("categories")) < 100)
            & (~col("categories").rlike("(?i)(review|amazon|page|chapter|purchase)")),
            trim(col("categories")),
        ).otherwise(None),
    )

    after_null = df.filter(col("categories").isNull()).count()
    print(f"✓ Categories: Cleaned {after_null - before_null:,} corrupted entries")

# ------------------------------------------------------
# 8️⃣ Remove duplicate reviews
# ------------------------------------------------------
print("\n🗑️  Removing duplicate reviews (same user + same book)...")

before = df.count()

# For duplicates, keep the most recent or highest-rated
order_cols = [desc("rating")]
if "review_time" in df.columns:
    order_cols.insert(0, desc("review_time"))

window = Window.partitionBy("user_id", "book_title").orderBy(*order_cols)
df = df.withColumn("row_num", row_number().over(window))
df = df.filter(col("row_num") == 1).drop("row_num")

after = df.count()
print(f"✓ Removed {before - after:,} duplicate reviews")
print(f"✓ Unique user-book pairs: {after:,}")

# ------------------------------------------------------
# 9️⃣ Filter sparse users and books
# ------------------------------------------------------
print("\n📊 Filtering sparse users and books...")

# Users with at least 3 reviews
user_counts = df.groupBy("user_id").count()
print(f"   User review distribution:")
user_counts.select("count").summary("min", "25%", "50%", "75%", "max").show()

before = df.count()
qualified_users = user_counts.filter(col("count") >= 3).select("user_id")
df = df.join(qualified_users, on="user_id", how="inner")
after = df.count()
print(f"✓ Kept users with 3+ reviews: {before - after:,} rows removed")

# Books with at least 3 reviews
book_counts = df.groupBy("book_title").count()
print(f"\n   Book review distribution:")
book_counts.select("count").summary("min", "25%", "50%", "75%", "max").show()

before = df.count()
qualified_books = book_counts.filter(col("count") >= 3).select("book_title")
df = df.join(qualified_books, on="book_title", how="inner")
after = df.count()
print(f"✓ Kept books with 3+ reviews: {before - after:,} rows removed")

# ------------------------------------------------------
# 🔟 Select columns for sentiment analysis
# ------------------------------------------------------
print("\n📝 Selecting columns for sentiment analysis...")

# Essential columns for sentiment + recommender
output_columns = [
    "user_id",
    "book_title",
    "review_text",  # CRITICAL for sentiment
    "rating",
    "profile_name",
    "authors",
    "categories",
    "description",
    "publisher",
    "publishedDate",
]

# Only keep columns that exist
output_columns = [c for c in output_columns if c in df.columns]
df = df.select(*output_columns)

print(f"✓ Selected {len(output_columns)} columns:")
print(f"   {', '.join(output_columns)}")

# ------------------------------------------------------
# 1️⃣1️⃣ Final statistics
# ------------------------------------------------------
print("\n" + "=" * 80)
print("📊 FINAL DATASET STATISTICS")
print("=" * 80)

final_count = df.count()
unique_users = df.select("user_id").distinct().count()
unique_books = df.select("book_title").distinct().count()

print(f"\nTotal reviews: {final_count:,}")
print(f"Unique users: {unique_users:,}")
print(f"Unique books: {unique_books:,}")
print(f"Sparsity: {(1 - final_count / (unique_users * unique_books)) * 100:.2f}%")

if "authors" in df.columns:
    with_authors = df.filter(col("authors").isNotNull()).count()
    print(
        f"\nReviews with author info: {with_authors:,} ({with_authors/final_count*100:.1f}%)"
    )

if "categories" in df.columns:
    with_categories = df.filter(col("categories").isNotNull()).count()
    print(
        f"Reviews with categories: {with_categories:,} ({with_categories/final_count*100:.1f}%)"
    )

print("\n📋 Final sample:")
df.show(5, truncate=50)

# ------------------------------------------------------
# 1️⃣2️⃣ Save for sentiment analysis
# ------------------------------------------------------
output_path = "hdfs:///data/amazon_book_reviews/preprocessed_for_sentiment"
print(f"\n💾 Saving to: {output_path}")

df.write.mode("overwrite").csv(output_path, header=True)

print("\n✅ PREPROCESSING COMPLETE")
print("=" * 80)
print("\n📌 NEXT STEP: Run sentiment analysis")
print("   spark-submit amazon_books_sentiment.py")
print("=" * 80)

spark.stop()
