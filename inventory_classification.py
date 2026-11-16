"""Multicriteria Inventory Classification & Stratification Module.

Implements 3-step Pareto inventory stratification using:
1. Sales and POS Hits separate Pareto classification (A: 0-60%, B: 61-80%, C: 81-90%, D: 91-100%)
   with conversion to numerical scores (A=4, B=3, C=2, D=1).
2. Combined Weighted Score: 0.65 * Sales Score + 0.35 * Hits Score.
3. Final Inventory Class assignment based on combined score:
   - A: 4.00 - 3.25
   - B: 3.24 - 2.50
   - C: 2.49 - 1.75
   - D: < 1.75
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple
import pandas as pd


# Numerical score mapping for classes
CLASS_SCORE_MAP: Dict[str, int] = {
    "A": 4,
    "B": 3,
    "C": 2,
    "D": 1,
}

# Score to Final Inventory Class mapping brackets
FINAL_CLASS_BRACKETS: List[Tuple[float, float, str]] = [
    (3.25, 4.00, "A"),
    (2.50, 3.2499, "B"),
    (1.75, 2.4999, "C"),
    (0.00, 1.7499, "D"),
]


def score_from_letter_class(letter_class: str) -> int:
    """Convert letter class (A, B, C, D) to numerical score (A=4, B=3, C=2, D=1)."""
    return CLASS_SCORE_MAP.get(str(letter_class).strip().upper(), 1)


def assign_pareto_letter_class(
    cumulative_pct: float,
    threshold_a: float = 60.0,
    threshold_b: float = 80.0,
    threshold_c: float = 90.0,
) -> str:
    """Assign letter class based on cumulative contribution percentage.

    Thresholds:
    A: 0 - 60%
    B: 61 - 80% (above 60% to 80%)
    C: 81 - 90% (above 80% to 90%)
    D: 91 - 100% (above 90% to 100%)
    """
    if cumulative_pct <= threshold_a:
        return "A"
    elif cumulative_pct <= threshold_b:
        return "B"
    elif cumulative_pct <= threshold_c:
        return "C"
    else:
        return "D"


def assign_classes_with_straddling(
    cumulative_percentages: pd.Series,
    threshold_a: float = 60.0,
    threshold_b: float = 80.0,
    threshold_c: float = 90.0,
) -> List[str]:
    """Assign ABCD class along sorted series.

    Boundary items crossing a threshold are included in the higher-priority class,
    and subsequent items start the next class.
    """
    classes: List[str] = []
    current_class = "A"

    for cum_val in cumulative_percentages:
        classes.append(current_class)
        if current_class == "A" and cum_val >= threshold_a:
            current_class = "B"
        elif current_class == "B" and cum_val >= threshold_b:
            current_class = "C"
        elif current_class == "C" and cum_val >= threshold_c:
            current_class = "D"

    return classes


def calculate_combined_weighted_score(
    sales_score: int | float,
    hits_score: int | float,
    sales_weight: float = 0.65,
    hits_weight: float = 0.35,
) -> float:
    """Calculate Combined Weighted Score = sales_weight * Sales Score + hits_weight * Hits Score.

    Example: Sales Class A (4) & Hits Class B (3) -> 0.65 * 4 + 0.35 * 3 = 3.65.
    """
    return round((sales_weight * float(sales_score)) + (hits_weight * float(hits_score)), 4)


def assign_final_inventory_class(combined_score: float) -> str:
    """Assign final inventory class from the Combined Weighted Score.

    Final Inventory Class Brackets:
    A: 4.00 - 3.25
    B: 3.24 - 2.50
    C: 2.49 - 1.75
    D: < 1.75
    """
    score = round(float(combined_score), 4)
    if score >= 3.245:
        return "A"
    elif score >= 2.495:
        return "B"
    elif score >= 1.745:
        return "C"
    else:
        return "D"


def identify_classification_columns(df: pd.DataFrame) -> Dict[str, str]:
    """Identify column names in the uploaded DataFrame corresponding to required inventory fields."""
    cols = list(df.columns)
    mapping: Dict[str, str] = {}

    # Item Number
    for c in cols:
        if c.strip().lower() in ["item number", "item_number", "item no", "item #", "sku", "item"]:
            mapping["item_number"] = c
            break

    # Item Name
    for c in cols:
        if c.strip().lower() in ["item name", "item_name", "description", "product name", "product"]:
            mapping["item_name"] = c
            break

    # UoM
    for c in cols:
        if c.strip().lower() in ["uom", "u_o_m", "unit of measure", "unit"]:
            mapping["uom"] = c
            break

    # Sales ($)
    for c in cols:
        if "sales" in c.strip().lower():
            mapping["sales"] = c
            break

    # Hits / POS Hits
    for c in cols:
        c_low = c.strip().lower()
        if "hit" in c_low or "pos" in c_low or "pick" in c_low:
            mapping["hits"] = c
            break

    return mapping


def is_classification_dataframe(df: pd.DataFrame) -> bool:
    """Check if the DataFrame contains columns required for Sales and Hits stratification."""
    mapping = identify_classification_columns(df)
    return all(k in mapping for k in ["item_number", "item_name", "uom", "sales", "hits"])


def validate_classification_schema(df: pd.DataFrame) -> Tuple[bool, List[str]]:
    """Validate that the DataFrame contains required ABCD classification columns.

    Returns (is_valid, missing_columns_list).
    Required canonical columns: Item Number, Item Name, UoM, Sales ($), Hits.
    """
    mapping = identify_classification_columns(df)
    missing: List[str] = []
    canonical_labels = {
        "item_number": "Item Number",
        "item_name": "Item Name",
        "uom": "UoM",
        "sales": "Sales ($)",
        "hits": "Hits",
    }
    for key, label in canonical_labels.items():
        if key not in mapping:
            missing.append(label)

    return (len(missing) == 0, missing)


def stratify_inventory_multicriteria(
    df: pd.DataFrame,
    sales_weight: float = 0.65,
    hits_weight: float = 0.35,
    threshold_a: float = 60.0,
    threshold_b: float = 80.0,
    threshold_c: float = 90.0,
) -> pd.DataFrame:
    """Execute the 3-step multicriteria inventory stratification on the uploaded DataFrame.

    Step 1:
      - Calculate Sales % and POS Hits % contribution.
      - Calculate cumulative percentages and classify into A (0-60%), B (61-80%), C (81-90%), D (91-100%).
      - Convert letter classes to numerical scores (A=4, B=3, C=2, D=1).
    Step 2:
      - Combined Weighted Score = 0.65(Sales Score) + 0.35(Hits Score).
    Step 3:
      - Assign Final Inventory Class:
        A: 4.00 - 3.25
        B: 3.24 - 2.50
        C: 2.49 - 1.75
        D: < 1.75
    """
    if df.empty:
        raise ValueError("Cannot classify an empty inventory DataFrame.")

    col_map = identify_classification_columns(df)
    missing = [req for req in ["item_number", "item_name", "uom", "sales", "hits"] if req not in col_map]
    if missing:
        raise ValueError(f"Missing required inventory columns: {missing}. Found: {list(df.columns)}")

    sales_col = col_map["sales"]
    hits_col = col_map["hits"]
    item_id_col = col_map["item_number"]

    result_df = df.copy()

    # Clean numeric representations
    numeric_sales = pd.to_numeric(
        result_df[sales_col].astype(str).str.replace("$", "", regex=False).str.replace(",", "", regex=False),
        errors="coerce",
    ).fillna(0.0)

    numeric_hits = pd.to_numeric(
        result_df[hits_col].astype(str).str.replace(",", "", regex=False),
        errors="coerce",
    ).fillna(0.0)

    total_sales = float(numeric_sales.sum())
    total_hits = float(numeric_hits.sum())

    # Step 1: Classify Sales and POS Hits Separately
    # Sales contribution & Cumulative %
    result_df["Sales %"] = (numeric_sales / total_sales * 100.0) if total_sales > 0 else 0.0
    sales_sorted = result_df.sort_values(by=[sales_col, item_id_col], ascending=[False, True]).copy()
    sales_sorted["Cumulative Sales %"] = sales_sorted["Sales %"].cumsum().clip(upper=100.0)
    sales_sorted["Sales Class"] = assign_classes_with_straddling(
        sales_sorted["Cumulative Sales %"], threshold_a, threshold_b, threshold_c
    )
    sales_sorted["Sales Score"] = sales_sorted["Sales Class"].map(score_from_letter_class)

    # POS Hits contribution & Cumulative %
    result_df["Hits %"] = (numeric_hits / total_hits * 100.0) if total_hits > 0 else 0.0
    hits_sorted = result_df.sort_values(by=[hits_col, item_id_col], ascending=[False, True]).copy()
    hits_sorted["Cumulative Hits %"] = hits_sorted["Hits %"].cumsum().clip(upper=100.0)
    hits_sorted["Hits Class"] = assign_classes_with_straddling(
        hits_sorted["Cumulative Hits %"], threshold_a, threshold_b, threshold_c
    )
    hits_sorted["Hits Score"] = hits_sorted["Hits Class"].map(score_from_letter_class)

    # Merge metrics back onto original index
    result_df["Cumulative Sales %"] = sales_sorted.loc[result_df.index, "Cumulative Sales %"]
    result_df["Sales Class"] = sales_sorted.loc[result_df.index, "Sales Class"]
    result_df["Sales Score"] = sales_sorted.loc[result_df.index, "Sales Score"]

    result_df["Cumulative Hits %"] = hits_sorted.loc[result_df.index, "Cumulative Hits %"]
    result_df["Hits Class"] = hits_sorted.loc[result_df.index, "Hits Class"]
    result_df["Hits Score"] = hits_sorted.loc[result_df.index, "Hits Score"]

    # Step 2: Combined Weighted Score
    # Combined Weighted Score = 0.65(Sales Score) + 0.35(Hits Score)
    result_df["Combined Weighted Score"] = (sales_weight * result_df["Sales Score"]) + (hits_weight * result_df["Hits Score"])
    result_df["Combined Weighted Score"] = result_df["Combined Weighted Score"].round(4)

    # Step 3: Assign Final Inventory Class
    result_df["Final Inventory Class"] = result_df["Combined Weighted Score"].apply(assign_final_inventory_class)

    return result_df


def generate_classification_summary(
    classified_df: pd.DataFrame,
    class_column: str = "Final Inventory Class",
) -> pd.DataFrame:
    """Generate the Required Classification Summary.

    Reports:
    - Number of items in each final category (A, B, C, D)
    - % of total items
    - Total Sales for all items in each category ($)
    - % of total Sales
    - Total Hits in each category
    - % of total Hits
    """
    col_map = identify_classification_columns(classified_df)
    sales_col = col_map.get("sales", "Sales ($)")
    hits_col = col_map.get("hits", "Hits")

    sales_num = pd.to_numeric(
        classified_df[sales_col].astype(str).str.replace("$", "", regex=False).str.replace(",", "", regex=False),
        errors="coerce",
    ).fillna(0.0)

    hits_num = pd.to_numeric(
        classified_df[hits_col].astype(str).str.replace(",", "", regex=False),
        errors="coerce",
    ).fillna(0.0)

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
            "Final Inventory Class": cls,
            "Number of Items": item_count,
            "% of Items": round(item_pct, 2),
            "Total Sales ($)": round(cls_sales, 2),
            "% of Total Sales": round(sales_pct, 2),
            "Total Hits": round(cls_hits, 2),
            "% of Total Hits": round(hits_pct, 2),
        })

    summary_df = pd.DataFrame(rows)
    return summary_df
