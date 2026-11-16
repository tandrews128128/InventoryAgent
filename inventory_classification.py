"""Multicriteria Inventory Classification (ABCD Analysis) Module.

Provides deterministic supply chain inventory classification using weighted multicriteria
scoring across Sales ($) and Hits (transaction frequency / pick velocity).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


@dataclass
class ClassificationThresholds:
    """Cumulative contribution percentage cutoffs for A, B, C, and D classes."""

    threshold_a: float = 80.0
    threshold_b: float = 90.0
    threshold_c: float = 95.0
    threshold_d: float = 100.0

    def validate(self) -> None:
        """Validate that thresholds are in strict ascending order and within (0, 100]."""
        if not (0.0 < self.threshold_a < self.threshold_b < self.threshold_c <= self.threshold_d <= 100.0):
            raise ValueError(
                f"Thresholds must be in strict ascending order (0 < A < B < C <= D <= 100). "
                f"Got: A={self.threshold_a}, B={self.threshold_b}, C={self.threshold_c}, D={self.threshold_d}"
            )


@dataclass
class ClassificationWeights:
    """Weights for multicriteria scoring model (must sum to 100% or 1.0)."""

    sales_weight: float = 0.60
    hits_weight: float = 0.40

    def validate(self, tol: float = 1e-4) -> None:
        """Validate that weights sum to 1.0 (or 100.0)."""
        total = self.sales_weight + self.hits_weight
        if abs(total - 1.0) > tol and abs(total - 100.0) > tol:
            raise ValueError(
                f"Weights must sum to 100% (or 1.0). "
                f"Got Sales={self.sales_weight}, Hits={self.hits_weight} (sum={total})"
            )


def assign_classes_by_straddling_threshold(
    cumulative_percentages: pd.Series,
    thresholds: ClassificationThresholds,
) -> List[str]:
    """Assign ABCD class based on cumulative contribution thresholds using the straddling rule.

    Rule: Include the item that crosses each cumulative threshold in the higher-priority class,
    then begin the next class with the following item.
    """
    classes: List[str] = []
    current_class = "A"

    for cum_val in cumulative_percentages:
        assigned = current_class
        classes.append(assigned)

        # After assigning current item to higher-priority class, advance class if boundary crossed
        if current_class == "A" and cum_val >= thresholds.threshold_a:
            current_class = "B"
        elif current_class == "B" and cum_val >= thresholds.threshold_b:
            current_class = "C"
        elif current_class == "C" and cum_val >= thresholds.threshold_c:
            current_class = "D"

    return classes


def normalize_min_max(series: pd.Series) -> pd.Series:
    """Normalize a series to [0, 100] using Min-Max scaling.

    Formula: (x - min) / (max - min) * 100.
    Ensures that larger values produce strictly higher scores.
    If max == min, returns 100.0 for all items.
    """
    clean_series = pd.to_numeric(series, errors="coerce").fillna(0.0)
    s_min = clean_series.min()
    s_max = clean_series.max()

    if s_max == s_min:
        return pd.Series(100.0, index=series.index, dtype=float)

    return ((clean_series - s_min) / (s_max - s_min)) * 100.0


def identify_classification_columns(df: pd.DataFrame) -> Dict[str, str]:
    """Detect presence and mapping of inventory classification columns.

    Required logical columns:
    - item_number: 'Item Number' (or synonyms)
    - item_name: 'Item Name' (or synonyms)
    - uom: 'UoM' or 'UOM'
    - sales: 'Sales ($)' or 'Sales'
    - hits: 'Hits'
    """
    cols = list(df.columns)
    mapping: Dict[str, str] = {}

    # Find Item Number
    for c in cols:
        if c.strip().lower() in ["item number", "item_number", "item no", "item #", "sku"]:
            mapping["item_number"] = c
            break

    # Find Item Name
    for c in cols:
        if c.strip().lower() in ["item name", "item_name", "description", "product name", "product"]:
            mapping["item_name"] = c
            break

    # Find UoM
    for c in cols:
        if c.strip().lower() in ["uom", "u_o_m", "unit of measure", "unit"]:
            mapping["uom"] = c
            break

    # Find Sales ($)
    for c in cols:
        if "sales" in c.strip().lower():
            mapping["sales"] = c
            break

    # Find Hits
    for c in cols:
        if "hits" in c.strip().lower() or "picks" in c.strip().lower() or "frequency" in c.strip().lower():
            mapping["hits"] = c
            break

    return mapping


def is_classification_dataframe(df: pd.DataFrame) -> bool:
    """Check if dataframe contains the columns required for multicriteria classification."""
    mapping = identify_classification_columns(df)
    return all(k in mapping for k in ["item_number", "item_name", "uom", "sales", "hits"])


def classify_inventory_multicriteria(
    df: pd.DataFrame,
    weights: Optional[ClassificationWeights] = None,
    thresholds: Optional[ClassificationThresholds] = None,
) -> pd.DataFrame:
    """Execute complete multicriteria inventory classification across Sales and Hits.

    Returns a new DataFrame preserving original columns and appending 13 calculated columns:
    1. Sales %
    2. Hits %
    3. Sales Rank
    4. Hits Rank
    5. Cumulative Sales %
    6. Cumulative Hits %
    7. Sales Class
    8. Hits Class
    9. Normalized Sales Score
    10. Normalized Hits Score
    11. Combined Score
    12. Combined Score %
    13. Cumulative Combined Score %
    14. Final ABCD Class
    """
    if df.empty:
        raise ValueError("Cannot classify an empty inventory DataFrame.")

    col_map = identify_classification_columns(df)
    missing = [req for req in ["item_number", "item_name", "uom", "sales", "hits"] if req not in col_map]
    if missing:
        raise ValueError(f"Missing required classification columns: {missing}. Available: {list(df.columns)}")

    if weights is None:
        weights = ClassificationWeights()
    weights.validate()

    # Normalize weights so they sum to 1.0 regardless of whether passed as 60/40 or 0.6/0.4
    total_w = weights.sales_weight + weights.hits_weight
    w_sales = weights.sales_weight / total_w
    w_hits = weights.hits_weight / total_w

    if thresholds is None:
        thresholds = ClassificationThresholds()
    thresholds.validate()

    sales_col = col_map["sales"]
    hits_col = col_map["hits"]

    # Work on a fresh copy
    result_df = df.copy()

    # Numeric conversion
    numeric_sales = pd.to_numeric(result_df[sales_col].astype(str).str.replace("$", "", regex=False).str.replace(",", "", regex=False), errors="coerce").fillna(0.0)
    numeric_hits = pd.to_numeric(result_df[hits_col].astype(str).str.replace(",", "", regex=False), errors="coerce").fillna(0.0)

    # 1. Total Sales and Total Hits
    total_sales = float(numeric_sales.sum())
    total_hits = float(numeric_hits.sum())

    # 2. Percentage Contributions
    result_df["Sales %"] = (numeric_sales / total_sales * 100.0) if total_sales > 0 else 0.0
    result_df["Hits %"] = (numeric_hits / total_hits * 100.0) if total_hits > 0 else 0.0

    # 3. Descending Ranks (Rank 1 = Highest)
    result_df["Sales Rank"] = numeric_sales.rank(ascending=False, method="min").astype(int)
    result_df["Hits Rank"] = numeric_hits.rank(ascending=False, method="min").astype(int)

    # 4. Sales-Only Cumulative % & Classification
    # Sort temporary view by sales descending
    sales_sorted = result_df.sort_values(by=[sales_col, col_map["item_number"]], ascending=[False, True]).copy()
    sales_sorted["Cumulative Sales %"] = sales_sorted["Sales %"].cumsum()
    # Cap cumulative at 100.0
    sales_sorted["Cumulative Sales %"] = sales_sorted["Cumulative Sales %"].clip(upper=100.0)
    sales_sorted["Sales Class"] = assign_classes_by_straddling_threshold(sales_sorted["Cumulative Sales %"], thresholds)

    # Re-merge sales metrics back to original indexing
    result_df["Cumulative Sales %"] = sales_sorted.loc[result_df.index, "Cumulative Sales %"]
    result_df["Sales Class"] = sales_sorted.loc[result_df.index, "Sales Class"]

    # 5. Hits-Only Cumulative % & Classification
    hits_sorted = result_df.sort_values(by=[hits_col, col_map["item_number"]], ascending=[False, True]).copy()
    hits_sorted["Cumulative Hits %"] = hits_sorted["Hits %"].cumsum()
    hits_sorted["Cumulative Hits %"] = hits_sorted["Cumulative Hits %"].clip(upper=100.0)
    hits_sorted["Hits Class"] = assign_classes_by_straddling_threshold(hits_sorted["Cumulative Hits %"], thresholds)

    # Re-merge hits metrics back to original indexing
    result_df["Cumulative Hits %"] = hits_sorted.loc[result_df.index, "Cumulative Hits %"]
    result_df["Hits Class"] = hits_sorted.loc[result_df.index, "Hits Class"]

    # 6. Min-Max Normalization (0 - 100 scale, larger values = higher scores)
    result_df["Normalized Sales Score"] = normalize_min_max(numeric_sales)
    result_df["Normalized Hits Score"] = normalize_min_max(numeric_hits)

    # 7. Combined Weighted Score
    # Combined Score = (Sales Weight * Normalized Sales Score) + (Hits Weight * Normalized Hits Score)
    result_df["Combined Score"] = (w_sales * result_df["Normalized Sales Score"]) + (w_hits * result_df["Normalized Hits Score"])

    # 8. Combined Score % and Cumulative Combined Score %
    total_combined = float(result_df["Combined Score"].sum())
    if total_combined > 0:
        result_df["Combined Score %"] = (result_df["Combined Score"] / total_combined) * 100.0
    else:
        result_df["Combined Score %"] = 100.0 / len(result_df)

    # Sort items by Combined Score descending to compute cumulative percentage and final class
    combined_sorted = result_df.sort_values(by=["Combined Score", sales_col, hits_col], ascending=[False, False, False]).copy()
    combined_sorted["Cumulative Combined Score %"] = combined_sorted["Combined Score %"].cumsum()
    combined_sorted["Cumulative Combined Score %"] = combined_sorted["Cumulative Combined Score %"].clip(upper=100.0)
    combined_sorted["Final ABCD Class"] = assign_classes_by_straddling_threshold(combined_sorted["Cumulative Combined Score %"], thresholds)

    # Map back to result dataframe
    result_df["Cumulative Combined Score %"] = combined_sorted.loc[result_df.index, "Cumulative Combined Score %"]
    result_df["Final ABCD Class"] = combined_sorted.loc[result_df.index, "Final ABCD Class"]

    return result_df


def generate_class_summary_metrics(
    classified_df: pd.DataFrame,
    class_column: str = "Final ABCD Class",
) -> pd.DataFrame:
    """Generate summary aggregation table across A, B, C, D classes.

    Metrics returned for each class:
    - Number of Items
    - % of Total Items
    - Total Sales ($)
    - % of Total Sales
    - Total Hits
    - % of Total Hits
    """
    col_map = identify_classification_columns(classified_df)
    sales_col = col_map.get("sales", "Sales ($)")
    hits_col = col_map.get("hits", "Hits")

    # Numeric clean
    sales_num = pd.to_numeric(classified_df[sales_col].astype(str).str.replace("$", "", regex=False).str.replace(",", "", regex=False), errors="coerce").fillna(0.0)
    hits_num = pd.to_numeric(classified_df[hits_col].astype(str).str.replace(",", "", regex=False), errors="coerce").fillna(0.0)

    work_df = pd.DataFrame({
        "Class": classified_df[class_column],
        "Sales": sales_num,
        "Hits": hits_num,
    })

    total_items = len(work_df)
    total_sales = float(work_df["Sales"].sum())
    total_hits = float(work_df["Hits"].sum())

    classes = ["A", "B", "C", "D"]
    rows: List[Dict[str, Any]] = []

    for cls in classes:
        sub = work_df[work_df["Class"] == cls]
        item_count = len(sub)
        item_pct = (item_count / total_items * 100.0) if total_items > 0 else 0.0
        cls_sales = float(sub["Sales"].sum())
        sales_pct = (cls_sales / total_sales * 100.0) if total_sales > 0 else 0.0
        cls_hits = float(sub["Hits"].sum())
        hits_pct = (cls_hits / total_hits * 100.0) if total_hits > 0 else 0.0

        rows.append({
            "Class": cls,
            "Item Count": item_count,
            "% of Total Items": round(item_pct, 2),
            "Total Sales ($)": round(cls_sales, 2),
            "% of Total Sales": round(sales_pct, 2),
            "Total Hits": round(cls_hits, 2),
            "% of Total Hits": round(hits_pct, 2),
        })

    summary_df = pd.DataFrame(rows)
    return summary_df
