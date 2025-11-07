from flask import Flask, render_template, request, jsonify
import subprocess
import json
import os
import secrets
import re

app = Flask(__name__)
app.secret_key = secrets.token_hex(16)

RESULTS_DIR = "results"
LOGS_DIR = "logs"
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(LOGS_DIR, exist_ok=True)


# ======================================================
# RUN SPARK UTILITY
# ======================================================
def run_spark(script, args=None):
    if args is None:
        args = []

    try:
        cmd = ["spark-submit", script] + args
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=400)

        # Output could contain file paths like:
        # LOG_FILE=logs/user_abc123.txt
        # RESULT_FILE=results/book_harry_potter.json

        stdout = result.stdout

        log_match = re.search(r"LOG_FILE=([^\s]+)", stdout)
        result_match = re.search(r"RESULT_FILE=([^\s]+)", stdout)

        return {
            "success": result.returncode == 0,
            "log_file": log_match.group(1) if log_match else None,
            "result_file": result_match.group(1) if result_match else None,
            "stderr": result.stderr,
        }

    except subprocess.TimeoutExpired:
        return {"success": False, "error": "Spark job timed out."}
    except Exception as e:
        return {"success": False, "error": str(e)}


# ======================================================
# ROUTES
# ======================================================


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/collaborative")
def collaborative():
    return render_template("collaborative.html")


@app.route("/content-based")
def content_based():
    return render_template("content_based.html")


# ======================================================
# RUN COLLABORATIVE
# ======================================================
@app.route("/api/run-collaborative", methods=["POST"])
def run_collaborative():
    result = run_spark("colab-filter.py")

    if not result["success"]:
        return jsonify({"success": False, "error": result.get("stderr", "Failed")}), 500

    if not result["log_file"]:
        return (
            jsonify({"success": False, "error": "No log file returned by script"}),
            500,
        )

    # Read the log
    try:
        with open(result["log_file"], "r", encoding="utf-8") as f:
            log = f.read()
    except:
        return jsonify({"success": False, "error": "Failed to read log file"}), 500

    # Extract metadata
    rmse = re.search(r"RMSE[:\s=]+(\d+\.\d+)", log)
    user = re.search(r"Target User[:\s]+(.+?)(?:\(|$)", log)
    user_id = re.search(r"\(([A-Za-z0-9_-]+)\)", log)

    # Extract recommendations from new consistent table format
    recs = []
    rec_pattern = r"(\d+)\s+([\d\.]+)\s+([\d\.]+)\s+([\d\.]+)\s+(.+?)\s{2,}(.+)$"
    for line in log.split("\n"):
        m = re.match(rec_pattern, line.strip())
        if m:
            rank, final_score, cf_score, boost, title, author = m.groups()
            recs.append(
                {
                    "rank": int(rank),
                    "title": title.strip(),
                    "author": author.strip(),
                    "score": float(final_score),
                    "boost": float(boost),
                }
            )

    return jsonify(
        {
            "success": True,
            "user": {
                "name": user.group(1).strip() if user else "Unknown",
                "id": user_id.group(1) if user_id else None,
            },
            "model": {"rmse": float(rmse.group(1)) if rmse else None},
            "recommendations": recs,
        }
    )


# ======================================================
# RUN CONTENT BASED
# ======================================================
@app.route("/api/run-content-based", methods=["POST"])
def run_content_based():
    data = request.get_json()
    title = data.get("book_title", "").strip()

    if not title:
        return jsonify({"success": False, "error": "Please enter a book title"}), 400

    result = run_spark("content-filter.py", [title])

    if not result["success"]:
        return jsonify({"success": False, "error": result.get("stderr", "Failed")}), 500

    if not result["result_file"]:
        return jsonify({"success": False, "error": "No result file returned"}), 500

    try:
        with open(result["result_file"], "r", encoding="utf-8") as f:
            data = json.load(f)
    except:
        return jsonify({"success": False, "error": "Failed to read result JSON"}), 500

    return jsonify(
        {
            "success": True,
            "matched_title": data.get("matched_title"),
            "recommendations": data.get("recommendations", []),
        }
    )


# ======================================================
if __name__ == "__main__":
    print("🚀 UI Running at http://localhost:5000")
    app.run(debug=True, host="0.0.0.0", port=5000)
