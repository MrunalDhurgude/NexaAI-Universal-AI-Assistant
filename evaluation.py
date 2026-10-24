"""
evaluation.py - Evaluation, Metrics, Comparison, and Visualization Module.

Implements:
1. Human Evaluation Rubric (Relevance, Correctness, Clarity, 1-5 scale).
2. Automated Classification Metrics (Accuracy, Extraction, Confusion counts).
3. Text Summarization Metrics (Word counts, Compression ratio).
4. Latency Analysis & Benchmark Visualizations using Matplotlib.
5. Query Consistency Evaluator.
6. CSV Export Functionality.
"""

import re
import json
import io
from typing import Dict, Any, List, Optional, Tuple
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib


# Configure matplotlib for headless / Streamlit embedding
matplotlib.use("Agg")
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")

# -----------------------------------------------------------------------------
# 1. RUBRIC DEFINITIONS
# -----------------------------------------------------------------------------

EVALUATION_RUBRIC = {
    "relevance": {
        "title": "Relevance (1-5)",
        "description": "How directly and completely does the response address the input prompt without digression?",
        "levels": {
            1: "Irrelevant / ignores prompt",
            2: "Tangentially related with severe gaps",
            3: "Adequately addresses primary topic",
            4: "Highly relevant with minor omissions",
            5: "Completely aligned with input query"
        }
    },
    "correctness": {
        "title": "Correctness (1-5)",
        "description": "Is the information factually accurate, logically sound, and free from hallucinations?",
        "levels": {
            1: "Factually wrong / severe hallucinations",
            2: "Substantial errors or misleading claims",
            3: "Mostly accurate with minor imprecision",
            4: "High factual precision and sound reasoning",
            5: "Flawlessly accurate and verifiable"
        }
    },
    "clarity": {
        "title": "Clarity & Readability (1-5)",
        "description": "Is the explanation well-formatted, coherent, and appropriately articulated for students?",
        "levels": {
            1: "Incoherent or unreadable",
            2: "Confusing structure and poor grammar",
            3: "Understandable with basic formatting",
            4: "Well-structured, concise, and clean",
            5: "Exceptional pedagogy, formatting, and flow"
        }
    }
}

# -----------------------------------------------------------------------------
# 2. AUTOMATED METRICS & LABEL PARSING
# -----------------------------------------------------------------------------

def extract_sentiment_label(raw_text: str) -> str:
    """
    Extracts the primary sentiment label (Positive, Negative, or Neutral)
    from unstructured LLM responses or JSON outputs.
    """
    if not raw_text:
        return "Unknown"

    # Attempt JSON parsing first
    try:
        # Check if text is enclosed in markdown code fences
        cleaned = raw_text.strip()
        if cleaned.startswith("```json"):
            cleaned = cleaned[7:]
        if cleaned.startswith("```"):
            cleaned = cleaned[3:]
        if cleaned.endswith("```"):
            cleaned = cleaned[:-3]

        data = json.loads(cleaned.strip())
        if isinstance(data, dict) and "sentiment" in data:
            val = str(data["sentiment"]).strip().capitalize()
            if val in ["Positive", "Negative", "Neutral"]:
                return val
    except Exception:
        pass

    # Regex search for sentiment labels
    lower = raw_text.lower()

    # Look for "sentiment: positive", "sentiment: negative", etc.
    label_match = re.search(r"(?:sentiment|classification)\s*[:=\-]?\s*\**\s*(positive|negative|neutral)", lower)
    if label_match:
        return label_match.group(1).capitalize()

    # Fallback keyword count
    pos_count = len(re.findall(r"\bpositive\b", lower))
    neg_count = len(re.findall(r"\bnegative\b", lower))
    neu_count = len(re.findall(r"\bneutral\b", lower))

    if pos_count > neg_count and pos_count > neu_count:
        return "Positive"
    elif neg_count > pos_count and neg_count > neu_count:
        return "Negative"
    elif neu_count > 0:
        return "Neutral"

    return "Unknown"


def compute_compression_ratio(original_text: str, summary_text: str) -> float:
    """Computes compression percentage: (summary_words / original_words) * 100."""
    orig_words = max(1, len(original_text.split()))
    summ_words = len(summary_text.split())
    return round((summ_words / orig_words) * 100, 2)


def compute_accuracy(y_true: List[str], y_pred: List[str]) -> Tuple[float, int, int]:
    """
    Computes classification accuracy given expected labels and predicted labels.
    Returns: (accuracy_pct, correct_count, total_count)
    """
    if not y_true or not y_pred:
        return 0.0, 0, 0

    correct = 0
    total = len(y_true)
    for true_val, pred_val in zip(y_true, y_pred):
        if str(true_val).strip().lower() == str(pred_val).strip().lower():
            correct += 1

    accuracy = round((correct / total) * 100, 2) if total > 0 else 0.0
    return accuracy, correct, total

# -----------------------------------------------------------------------------
# 3. CONSISTENCY TESTING
# -----------------------------------------------------------------------------

def evaluate_consistency(
    query_variants: List[str],
    outputs: List[str],
    task_type: str
) -> Dict[str, Any]:
    """
    Measures output consistency across semantically equivalent prompt variations.
    For sentiment: checks whether the predicted class remains identical across variants.
    For QA/summarization: checks length variance and vocabulary overlap (Jaccard similarity).
    """
    if not outputs or len(outputs) < 2:
        return {
            "consistency_score": 100.0,
            "status": "Insufficient queries for consistency variance",
            "details": []
        }

    if task_type == "sentiment":
        labels = [extract_sentiment_label(out) for out in outputs]
        # Most frequent label count
        counts = pd.Series(labels).value_counts()
        majority_pct = round((counts.iloc[0] / len(labels)) * 100, 2)
        return {
            "consistency_score": majority_pct,
            "status": "Consistent" if majority_pct == 100 else "Partially Inconsistent",
            "majority_label": counts.index[0],
            "labels": labels,
            "query_count": len(query_variants)
        }
    else:
        # Lexical Jaccard similarity across pairwise outputs
        token_sets = [set(re.findall(r"\w+", out.lower())) for out in outputs]
        similarities = []
        for i in range(len(token_sets)):
            for j in range(i + 1, len(token_sets)):
                intersection = len(token_sets[i].intersection(token_sets[j]))
                union = len(token_sets[i].union(token_sets[j]))
                if union > 0:
                    similarities.append(intersection / union)

        avg_jaccard = round((sum(similarities) / len(similarities)) * 100, 2) if similarities else 0.0
        return {
            "consistency_score": avg_jaccard,
            "status": "High Semantic Alignment" if avg_jaccard > 40 else "Moderate Variance",
            "pairwise_similarity_pct": avg_jaccard,
            "query_count": len(query_variants)
        }

# -----------------------------------------------------------------------------
# 4. VISUALIZATION FUNCTIONS (Matplotlib)
# -----------------------------------------------------------------------------

def plot_latency_comparison(records_df: pd.DataFrame) -> plt.Figure:
    """Generates a bar plot comparing average response latency by prompting technique."""
    fig, ax = plt.subplots(figsize=(7, 4), dpi=100)

    if records_df.empty or "technique" not in records_df.columns or "latency_sec" not in records_df.columns:
        ax.text(0.5, 0.5, "No Latency Data Available", ha="center", va="center", color="#888")
        ax.axis("off")
        return fig

    grouped = records_df.groupby("technique")["latency_sec"].mean().reset_index()
    colors = ["#4361EE", "#3A0CA3", "#7209B7", "#4CC9F0"][:len(grouped)]

    bars = ax.bar(grouped["technique"], grouped["latency_sec"], color=colors, width=0.55, edgecolor="#2b2d42")
    ax.set_ylabel("Average Latency (Seconds)", fontsize=11, fontweight="bold")
    ax.set_title("Response Latency Across Prompt Techniques", fontsize=12, fontweight="bold", pad=12)
    ax.tick_params(axis="x", rotation=15)
    ax.grid(axis="y", linestyle="--", alpha=0.6)

    # Annotate bar values
    for bar in bars:
        h = bar.get_height()
        ax.annotate(f"{h:.2f}s", xy=(bar.get_x() + bar.get_width() / 2, h),
                    xytext=(0, 4), textcoords="offset points", ha="center", va="bottom",
                    fontweight="bold", fontsize=9)

    plt.tight_layout()
    return fig


def plot_rubric_scores(records_df: pd.DataFrame) -> plt.Figure:
    """Plots a grouped bar chart comparing Relevance, Correctness, and Clarity across techniques."""
    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=100)

    score_cols = ["relevance", "correctness", "clarity"]
    available_cols = [c for c in score_cols if c in records_df.columns]

    if records_df.empty or "technique" not in records_df.columns or not available_cols:
        ax.text(0.5, 0.5, "No Human Evaluation Ratings Recorded Yet", ha="center", va="center", color="#888")
        ax.axis("off")
        return fig

    avg_scores = records_df.groupby("technique")[available_cols].mean()

    avg_scores.plot(
        kind="bar",
        ax=ax,
        color=["#4361EE", "#10B981", "#F59E0B"][:len(available_cols)],
        width=0.7,
        edgecolor="#1e293b"
    )

    ax.set_ylabel("Average Score (Scale 1–5)", fontsize=11, fontweight="bold")
    ax.set_ylim(0, 5.5)
    ax.set_title("Human Evaluation Scores by Prompt Technique", fontsize=12, fontweight="bold", pad=12)
    ax.tick_params(axis="x", rotation=15)
    ax.legend(["Relevance", "Correctness", "Clarity"][:len(available_cols)], loc="upper left", frameon=True)
    ax.grid(axis="y", linestyle="--", alpha=0.6)

    plt.tight_layout()
    return fig


def plot_classification_distribution(records_df: pd.DataFrame) -> plt.Figure:
    """Plots classification prediction distribution."""
    fig, ax = plt.subplots(figsize=(6, 3.8), dpi=100)

    if records_df.empty or "predicted_label" not in records_df.columns:
        ax.text(0.5, 0.5, "No Sentiment Records Evaluated", ha="center", va="center", color="#888")
        ax.axis("off")
        return fig

    counts = records_df["predicted_label"].value_counts()
    palette = {"Positive": "#10B981", "Negative": "#EF4444", "Neutral": "#6B7280", "Unknown": "#F59E0B"}
    colors = [palette.get(k, "#4361EE") for k in counts.index]

    bars = ax.bar(counts.index, counts.values, color=colors, width=0.5, edgecolor="#1e293b")
    ax.set_ylabel("Frequency", fontsize=11, fontweight="bold")
    ax.set_title("Predicted Sentiment Distribution", fontsize=12, fontweight="bold", pad=10)
    ax.grid(axis="y", linestyle="--", alpha=0.5)

    for bar in bars:
        h = bar.get_height()
        ax.annotate(f"{int(h)}", xy=(bar.get_x() + bar.get_width() / 2, h),
                    xytext=(0, 3), textcoords="offset points", ha="center", va="bottom",
                    fontweight="bold", fontsize=10)

    plt.tight_layout()
    return fig

# -----------------------------------------------------------------------------
# 5. DATA EXPORT HELPER
# -----------------------------------------------------------------------------

def export_records_to_csv(records: List[Dict[str, Any]]) -> str:
    """Converts recorded test evaluations into a downloadable CSV string."""
    if not records:
        return ""
    df = pd.DataFrame(records)
    return df.to_csv(index=False)
