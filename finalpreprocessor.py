from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when, regexp_replace, lower, length, trim
from pyspark.sql.types import FloatType

# ------------------------------------------------------
# Spark Session (optimized for WSL / low memory)
# ------------------------------------------------------
spark = (
    SparkSession.builder.appName("FinalPreprocessForRecommender")
    .config("spark.sql.ansi.enabled", "false")
    .config("spark.driver.memory", "2g")
    .config("spark.executor.memory", "2g")
    .config("spark.sql.shuffle.partitions", "50")
    .config("spark.default.parallelism", "50")
    .getOrCreate()
)

print("=" * 80)
print("🎯 FINAL PREPROCESSING FOR RECOMMENDER (Post-Sentiment)")
print("=" * 80)

# ------------------------------------------------------
# Load sentiment-enhanced dataset
# ------------------------------------------------------
input_path = "hdfs:///data/amazon_book_reviews/output_sentiment"
print(f"\n📥 Loading: {input_path}")

df = spark.read.csv(input_path, header=True, inferSchema=True)

total_rows = df.count()
print(f"✅ Loaded {total_rows:,} rows")

# ------------------------------------------------------
# Ensure numeric columns
# ------------------------------------------------------
df = df.withColumn("rating", col("rating").cast(FloatType()))
df = df.withColumn("sentiment_score", col("sentiment_score").cast(FloatType()))

# ------------------------------------------------------
# Blend rating + sentiment → final_rating
# ------------------------------------------------------
print("\n🧮 Creating final blended rating (70% rating + 30% sentiment)...")

df = df.withColumn(
    "sentiment_scaled",
    when(col("sentiment_score").isNotNull(), (col("sentiment_score") + 1.0) * 2.5),
)

df = df.withColumn(
    "final_rating",
    when(
        col("sentiment_scaled").isNotNull(),
        col("rating") * 0.7 + col("sentiment_scaled") * 0.3,
    ).otherwise(col("rating")),
)

df = df.withColumn(
    "final_rating",
    when(col("final_rating") < 1.0, 1.0)
    .when(col("final_rating") > 5.0, 5.0)
    .otherwise(col("final_rating")),
)

df = df.withColumn("final_rating", col("final_rating").cast(FloatType()))

print("✅ final_rating generated successfully")

# ------------------------------------------------------
# Clean authors & categories
# ------------------------------------------------------
print("\n🧹 Cleaning author & category metadata...")

if "authors" in df.columns:
    df = df.withColumn("authors", regexp_replace(col("authors"), r'[\[\]\'"]', ""))
    df = df.withColumn(
        "authors",
        when(
            (length(trim(col("authors"))) == 0) | (length(col("authors")) > 150), None
        ).otherwise(trim(col("authors"))),
    )
    df = df.withColumnRenamed("authors", "author")

if "categories" in df.columns:
    df = df.withColumn(
        "categories", regexp_replace(col("categories"), r'[\[\]\'"]', "")
    )
    df = df.withColumn(
        "categories",
        when(
            (length(trim(col("categories"))) == 0) | (length(col("categories")) > 100),
            None,
        ).otherwise(trim(col("categories"))),
    )

# ------------------------------------------------------
# Select final output columns for recommender
# ------------------------------------------------------
print("\n📝 Selecting relevant columns...")

final_cols = [
    "user_id",
    "book_title",
    "rating",
    "final_rating",
    "author",
    "categories",
    "description",
    "publisher",
    "publishedDate",
    "sentiment_score",
]

final_cols = [c for c in final_cols if c in df.columns]
df = df.select(*final_cols)

print(f"✅ Final column set: {final_cols}")

# ------------------------------------------------------
# Save Dataset (CSV + Parquet - Parquet used for ALS)
# ------------------------------------------------------
csv_out = "hdfs:///data/amazon_book_reviews/cleaned_data"
parquet_out = "hdfs:///data/amazon_book_reviews/cleaned_data_parquet"

print(f"\n💾 Saving CSV → {csv_out}")
df.coalesce(20).write.mode("overwrite").csv(csv_out, header=True)

print(f"💾 Saving Parquet → {parquet_out}")
df.repartition(100).write.mode("overwrite").parquet(parquet_out)

print("\n✅ FINAL PREPROCESSING COMPLETE!")
print("=" * 80)
print("\n📌 NEXT STEP: Recommender")
print("   spark-submit recommenderv1.py")
print("=" * 80)

spark.stop()
