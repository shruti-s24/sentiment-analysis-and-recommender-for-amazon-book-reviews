import os
import sys
import datetime
from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    col,
    rand,
    explode,
    when,
    avg,
    count,
    regexp_replace,
    lower,
    length,
    trim,
    lit,
)
from pyspark.ml.recommendation import ALS
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.feature import StringIndexer


# ----------------- helpers -----------------
def safe_count(df, hint=""):
    try:
        return df.count()
    except Exception as e:
        print(f"⚠️  count() failed {hint}: {e}")
        # Fallback: collect 1 row to see if empty
        try:
            one = df.limit(1).collect()
            return (
                0 if not one else 1
            )  # we don't know full size; return nonzero sentinel
        except Exception as e2:
            print(f"⚠️  fallback collect() also failed {hint}: {e2}")
            return 0


def write_log(path, text):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


# =============== BOOT BANNER (to stdout) ===============
print("=" * 100)
print("🤖 COLLABORATIVE FILTERING BOOK RECOMMENDER (WSL-Optimized)")
print("   ALS + (optional) content boosting + sentiment-aware ratings")
print("=" * 100)

log_lines = []  # we’ll buffer log lines and write at the end to logs/user_<id>.txt


def log(s=""):
    print(s)
    log_lines.append(s)


# =============== SPARK ===============
spark = (
    SparkSession.builder.appName("CollaborativeFilteringRecommender")
    .config("spark.executor.memory", "2g")
    .config("spark.driver.memory", "2g")
    .config("spark.sql.shuffle.partitions", "50")
    .config("spark.default.parallelism", "50")
    .getOrCreate()
)

# =============== INPUTS ===============
parquet_input = "hdfs:///data/amazon_book_reviews/cleaned_data_parquet"
csv_input = "hdfs:///data/amazon_book_reviews/cleaned_data"

log(f"\n📥 Trying Parquet: {parquet_input}")
fs = spark._jvm.org.apache.hadoop.fs.FileSystem.get(spark._jsc.hadoopConfiguration())
if fs.exists(spark._jvm.org.apache.hadoop.fs.Path(parquet_input)):
    df = spark.read.parquet(parquet_input)
    log("✅ Loaded Parquet")
else:
    log(f"⚠️  Parquet not found, falling back to CSV: {csv_input}")
    df = spark.read.csv(csv_input, header=True, inferSchema=True)
    log("✅ Loaded CSV")

log(f"   Columns: {df.columns}")

# Optional quick cap for dev (set to 0 to disable)
ROW_CAP = int(os.environ.get("ROW_CAP", "0"))
if ROW_CAP > 0:
    df = df.limit(ROW_CAP)
    log(f"   Using row cap: {ROW_CAP}")

total_interactions = safe_count(df, hint="for total interactions")
log(f"✅ Total interactions loaded: {total_interactions:,}")

# =============== CLEAN & PREP ===============
log("\n🔍 Validating essentials & choosing rating column...")
has_rating = "rating" in df.columns
has_final = "final_rating" in df.columns

if has_final:
    log("   Using 'final_rating' (sentiment-blended) as label")
    df = df.withColumnRenamed("final_rating", "label_rating")
elif has_rating:
    log("   Using 'rating' as label")
    df = df.withColumnRenamed("rating", "label_rating")
else:
    log("❌ ERROR: Neither 'final_rating' nor 'rating' present.")
    write_log("logs/user_UNKNOWN.txt", "\n".join(log_lines))
    spark.stop()
    sys.exit(1)

# Ensure author field exists and cleaned (light)
if "authors" in df.columns and "author" not in df.columns:
    df = df.withColumnRenamed("authors", "author")

if "author" in df.columns:
    df = df.withColumn("author", regexp_replace(col("author"), r'[\[\]\'"]', ""))
    df = df.withColumn(
        "author",
        when(
            (length(trim(col("author"))) == 0) | (length(col("author")) > 150), None
        ).otherwise(trim(col("author"))),
    )

# Clean categories lightly
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

# Keep essentials
needed = ["user_id", "book_title", "label_rating"]
for c in needed:
    if c not in df.columns:
        log(f"❌ Missing required column: {c}")
        write_log("logs/user_UNKNOWN.txt", "\n".join(log_lines))
        spark.stop()
        sys.exit(1)

df = df.na.drop(subset=needed).filter(col("label_rating").isNotNull()).cache()
post_drop = safe_count(df, hint="post-drop")
log(f"✅ Post-drop interactions: {post_drop:,}")

# =============== BASIC STATS ===============
log("\n📊 Dataset Stats:")
u = df.select("user_id").distinct().count()
b = df.select("book_title").distinct().count()
n = safe_count(df, hint="dataset size (stats)")
sparsity = (1 - (n / max(1, (u * b)))) * 100.0
log(f"   Users: {u:,} | Books: {b:,} | Interactions: {n:,} | Sparsity: {sparsity:.2f}%")

log("\n   Rating Summary:")
df.select("label_rating").summary("min", "25%", "50%", "75%", "max").show()

# =============== ENCODE KEYS ===============
log("\n🔢 Encoding user/book IDs...")
user_indexer = StringIndexer(
    inputCol="user_id", outputCol="user_index", handleInvalid="skip"
)
book_indexer = StringIndexer(
    inputCol="book_title", outputCol="book_index", handleInvalid="skip"
)
df = user_indexer.fit(df).transform(df)
df = book_indexer.fit(df).transform(df)
log("✅ Encoded user_index & book_index")

# =============== TRAIN/TEST SPLIT ===============
log("\n✂️  Train/Test Split (80/20)...")
train, test = df.randomSplit([0.8, 0.2], seed=42)
train_cnt = safe_count(train, hint="train")
test_cnt = safe_count(test, hint="test")
log(f"   Train: {train_cnt:,} | Test: {test_cnt:,}")

# =============== ALS TRAINING ===============
log("\n🏋️  Training ALS...")
als = ALS(
    rank=20,
    maxIter=12,
    regParam=0.12,
    userCol="user_index",
    itemCol="book_index",
    ratingCol="label_rating",
    coldStartStrategy="drop",
    nonnegative=True,
)
model = als.fit(train)
log("✅ Model trained")

# =============== EVALUATION ===============
log("\n🧪 Evaluating...")
pred = model.transform(test)
rmse_eval = RegressionEvaluator(
    metricName="rmse", labelCol="label_rating", predictionCol="prediction"
)
mae_eval = RegressionEvaluator(
    metricName="mae", labelCol="label_rating", predictionCol="prediction"
)
rmse = rmse_eval.evaluate(pred)
mae = mae_eval.evaluate(pred)
log(f"   RMSE: {rmse:.3f} | MAE: {mae:.3f}")

# =============== PICK A GOOD USER ===============
log("\n🎯 Selecting a target user (≥5 reviews, avg ≥ 2.5)...")
candidates = (
    df.groupBy("user_index", "user_id")
    .agg(count("*").alias("cnt"), avg("label_rating").alias("avg_r"))
    .filter((col("cnt") >= 5) & (col("avg_r") >= 2.5))
    .orderBy(rand())
    .limit(1)
    .collect()
)

if not candidates:
    # fallback: most active user
    log("⚠️  No user met (≥5 reviews & avg ≥ 2.5). Falling back to most active user.")
    fallback = (
        df.groupBy("user_index", "user_id")
        .agg(count("*").alias("cnt"), avg("label_rating").alias("avg_r"))
        .orderBy(col("cnt").desc())
        .limit(1)
        .collect()
    )
    if not fallback:
        log("❌ No users available in dataset.")
        write_log("logs/user_UNKNOWN.txt", "\n".join(log_lines))
        spark.stop()
        sys.exit(1)
    target = fallback[0]
else:
    target = candidates[0]

target_uidx = target["user_index"]
target_uid = target["user_id"]
log(
    f"   → Chosen user_id: {target_uid} (internal idx {target_uidx}), cnt={target['cnt']}, avg={target['avg_r']:.2f}"
)

# =============== USER HISTORY (brief) ===============
log("\n📚 User's recent high ratings:")
hist_cols = ["book_title", "label_rating"]
if "author" in df.columns:
    hist_cols.append("author")
if "categories" in df.columns:
    hist_cols.append("categories")

user_hist_df = (
    df.filter(col("user_index") == target_uidx)
    .select(*hist_cols)
    .orderBy(col("label_rating").desc())
    .limit(15)
)

for i, r in enumerate(user_hist_df.collect(), 1):
    r = r.asDict()
    title = str(r["book_title"])[:60]
    author = str(r.get("author", "") or "")[:30]
    log(f"   {i:2}. {r['label_rating']:.1f} ★ | {title} | {author}")

# Build simple preference signals for boosting
top_authors, top_genres = [], []
if "author" in df.columns:
    arows = (
        df.filter(
            (col("user_index") == target_uidx)
            & (col("label_rating") >= 4.0)
            & col("author").isNotNull()
        )
        .groupBy("author")
        .agg(count("*").alias("cnt"), avg("label_rating").alias("avg"))
        .orderBy(col("cnt").desc(), col("avg").desc())
        .limit(5)
        .collect()
    )
    top_authors = [r["author"] for r in arows]
if "categories" in df.columns:
    grows = (
        df.filter(
            (col("user_index") == target_uidx)
            & (col("label_rating") >= 4.0)
            & col("categories").isNotNull()
        )
        .groupBy("categories")
        .agg(count("*").alias("cnt"), avg("label_rating").alias("avg"))
        .orderBy(col("cnt").desc(), col("avg").desc())
        .limit(5)
        .collect()
    )
    top_genres = [r["categories"] for r in grows]

log("\n🧠 Preference signals:")
log(f"   Top authors: {top_authors if top_authors else '(none)'}")
log(f"   Top genres:  {top_genres if top_genres else '(none)'}")

# =============== CF RECOMMENDATIONS ===============
log("\n🔮 Generating CF recommendations...")
subset = df.filter(col("user_index") == target_uidx).select("user_index").distinct()
user_recs = model.recommendForUserSubset(subset, 100)

# safer emptiness check
has_any_rec = False
try:
    has_any_rec = user_recs.rdd.isEmpty() is False
except Exception:
    # Fallback: try to take(1)
    try:
        has_any_rec = len(user_recs.take(1)) > 0
    except Exception:
        has_any_rec = False

if not has_any_rec:
    log("❌ No CF recommendations.")
    log_path = f"logs/user_{target_uid}.txt"
    write_log(log_path, "\n".join(log_lines))
    print(f"LOG_FILE={log_path}")
    spark.stop()
    sys.exit(1)

rec_for_user = user_recs.select(explode("recommendations").alias("rec")).selectExpr(
    "rec.book_index as book_index", "rec.rating as cf_score"
)

# Books already seen by the user
seen = set(
    [
        r[0]
        for r in df.filter(col("user_index") == target_uidx)
        .select("book_index")
        .distinct()
        .collect()
    ]
)

# Metadata lookup (dedup)
meta = df.select("book_index", "book_title", "author", "categories").dropDuplicates(
    ["book_index"]
)

recs = rec_for_user.join(meta, "book_index", "left").filter(
    ~col("book_index").isin(list(seen))
)

# =============== CONTENT BOOST (optional) ===============
log("\n🚀 Applying small content boosts (author + genre)...")
author_boost = 1.0
genre_boost = 0.5

if top_authors:
    recs = recs.withColumn(
        "author_hit",
        when(col("author").isin(top_authors), author_boost).otherwise(lit(0.0)),
    )
else:
    recs = recs.withColumn("author_hit", lit(0.0))

if top_genres:
    recs = recs.withColumn(
        "genre_hit",
        when(col("categories").isin(top_genres), genre_boost).otherwise(lit(0.0)),
    )
else:
    recs = recs.withColumn("genre_hit", lit(0.0))

recs = recs.withColumn(
    "final_score", col("cf_score") + col("author_hit") + col("genre_hit")
)

# boosted items (safe)
boosted = safe_count(
    recs.filter((col("author_hit") > 0) | (col("genre_hit") > 0)), hint="boosted items"
)
log(f"   Boosted items: {boosted}")

# =============== TOP-N OUTPUT ===============
TOP_N = int(os.environ.get("TOP_N", "20"))
top_df = (
    recs.orderBy(col("final_score").desc())
    .select(
        "final_score",
        "cf_score",
        "author_hit",
        "genre_hit",
        "book_title",
        "author",
        "categories",
    )
    .limit(TOP_N)
)

top = top_df.collect()

log("\n" + "=" * 100)
log(f"🏆 TOP {len(top)} RECOMMENDATIONS")
log("=" * 100)
log(f"{'#':<3} {'Score':<7} {'CF':<6} {'Boost':<7} {'Title':<54} {'Author':<28}")
log("-" * 110)
for i, r in enumerate(top, 1):
    d = r.asDict()
    title = (d.get("book_title") or "Unknown")[:52]
    author = (d.get("author") or "Unknown")[:26]
    boost = float(d.get("author_hit", 0.0) + d.get("genre_hit", 0.0))
    log(
        f"{i:<3} {float(d['final_score']):<7.2f} {float(d['cf_score']):<6.2f} {boost:<7.2f} {title:<54} {author:<28}"
    )

# =============== SAVE RESULTS ===============
# (Optional) Save a parquet if you want, but not needed for UI.
# out_path = "hdfs:///data/amazon_book_reviews/recommendations_user_{}".format(target_uid)
# log(f"\n💾 Saving recommendations to: {out_path}")
# top_df.write.mode("overwrite").parquet(out_path)

# =============== SAVE LOG (deterministic path) ===============
log_path = f"logs/user_{target_uid}.txt"
write_log(log_path, "\n".join(log_lines))
print(f"LOG_FILE={log_path}")  # <-- UI will parse this exact path

log("\n✅ RECOMMENDATION GENERATION COMPLETE!")
spark.stop()
