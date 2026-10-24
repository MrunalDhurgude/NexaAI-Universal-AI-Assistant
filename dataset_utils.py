"""
dataset_utils.py - Dataset Ingestion, Validation, Profiling, and Batch Utilities for NexaAI.

Academic Context: AIDS Engineering 5th-Semester NLP Micro Project.
Features:
- Robust CSV parsing for custom uploads and local benchmarks.
- Dataset profiling (row counts, column counts, missing value summaries, dtypes).
- Automated candidate text column detection for LLM evaluation.
- Batch generation and format validation.
- Safe CSV export utilities for empirical evaluation tables.
"""

import io
from typing import Dict, Any, List, Optional, Tuple
import pandas as pd


def load_dataset(file_or_path: Any) -> pd.DataFrame:
    """
    Safely loads a dataset from a file path, StringIO, or Streamlit UploadedFile.
    Handles encoding detection and strip whitespace.
    """
    try:
        if isinstance(file_or_path, pd.DataFrame):
            return file_or_path.copy()

        df = pd.read_csv(file_or_path, encoding="utf-8")
        # Strip string columns whitespace
        for col in df.select_dtypes(include=["object"]).columns:
            df[col] = df[col].astype(str).str.strip()
        return df
    except UnicodeDecodeError:
        try:
            return pd.read_csv(file_or_path, encoding="latin1")
        except Exception as exc:
            raise ValueError(f"Unable to parse dataset with UTF-8 or Latin1 encoding: {exc}")
    except Exception as exc:
        raise ValueError(f"Failed to load dataset: {exc}")


def get_dataset_summary(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Computes standard statistical profiling metrics for a dataset:
    - Row and column counts
    - Missing value summary per column
    - Data types
    - Estimated memory usage
    """
    if df is None or df.empty:
        return {
            "num_rows": 0,
            "num_columns": 0,
            "columns": [],
            "missing_values": {},
            "dtypes": {},
            "memory_kb": 0.0,
            "is_empty": True
        }

    missing_dict = df.isnull().sum().to_dict()
    dtypes_dict = {col: str(dtype) for col, dtype in df.dtypes.items()}
    memory_kb = round(df.memory_usage(deep=True).sum() / 1024, 2)

    return {
        "num_rows": len(df),
        "num_columns": len(df.columns),
        "columns": list(df.columns),
        "missing_values": missing_dict,
        "dtypes": dtypes_dict,
        "memory_kb": memory_kb,
        "is_empty": False
    }


def detect_text_columns(df: pd.DataFrame) -> List[str]:
    """
    Detects potential text columns in the dataset suitable for NLP evaluation.
    Heuristics:
    1. Object or string dtype.
    2. Average string length > 10 characters or named input_text/text/prompt/question.
    """
    if df is None or df.empty:
        return []

    candidate_cols = []
    # Prioritize obvious NLP names
    priority_names = {"input_text", "text", "query", "prompt", "question", "sentence", "content", "review"}
    for col in df.columns:
        if col.lower() in priority_names:
            candidate_cols.append(col)

    # Next check string columns with decent length
    for col in df.select_dtypes(include=["object", "string"]).columns:
        if col not in candidate_cols:
            sample_non_null = df[col].dropna()
            if not sample_non_null.empty:
                avg_len = sample_non_null.astype(str).str.len().mean()
                if avg_len >= 12:
                    candidate_cols.append(col)

    # Fallback to all string columns
    if not candidate_cols:
        candidate_cols = list(df.select_dtypes(include=["object"]).columns)

    return candidate_cols


def validate_evaluation_dataset(df: pd.DataFrame) -> Tuple[bool, str, List[str]]:
    """
    Checks if a dataset has standard benchmark evaluation columns:
    Standard benchmark: ['task_type', 'input_text', 'expected_output']
    Returns: (is_standard_benchmark, message, available_columns)
    """
    if df is None or df.empty:
        return False, "Dataset is empty.", []

    cols = [c.lower() for c in df.columns]
    has_input = any(c in cols for c in ["input_text", "text", "query", "prompt"])
    has_expected = any(c in cols for c in ["expected_output", "expected", "label", "target"])
    has_task = "task_type" in cols or "task" in cols

    if has_input and has_expected and has_task:
        return True, "Standard benchmark dataset format detected (with ground-truth labels).", list(df.columns)
    elif has_input:
        return True, "Custom dataset detected. Text queries can be extracted for evaluation.", list(df.columns)
    else:
        return False, "Dataset missing clear text column for NLP processing.", list(df.columns)


def export_dataframe_to_csv(df: pd.DataFrame) -> str:
    """Exports DataFrame to UTF-8 CSV string for Streamlit download buttons."""
    if df is None or df.empty:
        return ""
    return df.to_csv(index=False)
