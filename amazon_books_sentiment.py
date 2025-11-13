from pyspark.sql import SparkSession
from pyspark.sql.functions import col
from pyspark.sql.types import FloatType
from nltk.sentiment import SentimentIntensityAnalyzer
import nltk

# Download VADER
nltk.download('vader_lexicon', quiet=True)

print("=" * 80)
print("💭 SAFE SENTIMENT ANALYSIS (LOW MEMORY)")
print("=" * 80)

# ------------------------------------------------------
# Configure Spark to fit WSL memory
# ------------------------------------------------------
spark = (
    SparkSession.builder
    .appName("AmazonBookSentimentSafe")
    .config("spark.executor.memory", "1g")
    .config("spark.driver.memory", "1g")
    .config("spark.sql.shuffle.partitions", "50")
    .config("spark.default.parallelism", "50")
    .getOrCreate()
)

# ------------------------------------------------------
# Load preprocessed dataset
# ------------------------------------------------------
input_path = "hdfs:///data/amazon_book_reviews/preprocessed_for_sentiment"
print(f"\n📥 Loading dataset: {input_path}")

df = spark.read.csv(input_path, header=True, inferSchema=True)

if "review_text" not in df.columns:
    raise ValueError("❌ review_text missing – preprocessing must include it.")

print(f"✅ Loaded {df.count():,} rows")

# ------------------------------------------------------
# Define mapPartitions sentiment function
# ------------------------------------------------------
def sentiment_partition(iterator):
    sia = SentimentIntensityAnalyzer()  # Loaded **once per partition**
    for row in iterator:
        text = row.review_text if row.review_text else ""
        score = sia.polarity_scores(text)["compound"]
        yield (*row, float(score))

# Apply transformation (RDD → DataFrame, safe & streaming)
rdd = df.rdd.mapPartitions(sentiment_partition)

# Preserve schema + add new column
cols = df.columns + ["sentiment_score"]
df_sent = rdd.toDF(cols)

# ------------------------------------------------------
# Save Output (coalesce so WSL doesn’t write 400+ files)
# ------------------------------------------------------
output_path = "hdfs:///data/amazon_book_reviews/output_sentiment"
print(f"\n💾 Saving sentiment-enhanced dataset → {output_path}")

df_sent.coalesce(20).write.mode("overwrite").csv(output_path, header=True)

print("\n✅ SAFE SENTIMENT ANALYSIS COMPLETE")
print("=" * 80)

spark.stop()
