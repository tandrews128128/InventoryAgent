"""Supply Chain Analytics Inventory Tools.

Deterministic, mathematically rigorous numerical calculation tools for the five
fundamental inventory management decision models:
1. calculate_eoq()
2. calculate_quantity_discount()
3. calculate_rop() (with alias calculate_safety_stock_and_rop)
4. calculate_periodic_review()
5. calculate_newsvendor()

Each tool:
- strictly validates all inputs (raises ValueError with descriptive guidance)
- never invents missing numerical inputs
- returns structured results with typed dataclasses
- returns explicit intermediate calculations
- returns the final answer
- identifies core model assumptions
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

# Attempt to import scipy for statistical distributions, with pure-Python fallback
try:
    from scipy import stats as sp_stats  # type: ignore

    _HAS_SCIPY = True
except ImportError:  # pragma: no cover
    _HAS_SCIPY = False


def norm_cdf(x: float) -> float:
    """Standard normal cumulative distribution function Phi(x)."""
    if _HAS_SCIPY:
        return float(sp_stats.norm.cdf(x))
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def norm_pdf(x: float) -> float:
    """Standard normal probability density function phi(x)."""
    if _HAS_SCIPY:
        return float(sp_stats.norm.pdf(x))
    return (1.0 / math.sqrt(2.0 * math.pi)) * math.exp(-0.5 * x * x)


def norm_ppf(p: float) -> float:
    """Percent point function (inverse CDF) for the standard normal distribution.
    Uses SciPy when available; otherwise uses Peter J. Acklam's high-precision rational approximation.
    """
    if not (0.0 < p < 1.0):
        raise ValueError(f"Probability p must be strictly in (0, 1), got {p}")

    if _HAS_SCIPY:
        return float(sp_stats.norm.ppf(p))

    # Acklam's approximation (accurate to 1.15e-9)
    a = [
        -3.969683028665376e01,
        2.209460984245205e02,
        -2.759285104469687e02,
        1.383577518672690e02,
        -3.066479806614716e01,
        2.506628277459239e00,
    ]
    b = [
        -5.447609879822406e01,
        1.615858368580409e02,
        -1.556989798598866e02,
        6.680131188771972e01,
        -1.328068155288572e01,
    ]
    c = [
        -7.784894002430293e-03,
        -3.223964580411365e-01,
        -2.400758277161838e00,
        -2.549732539343734e00,
        4.374664141464968e00,
        2.938163982698783e00,
    ]
    d = [
        7.784695709041462e-03,
        3.224671290700398e-01,
        2.445134137142996e00,
        3.754408661907416e00,
    ]

    p_low = 0.02425
    p_high = 1.0 - p_low

    if p < p_low:
        q = math.sqrt(-2.0 * math.log(p))
        return (
            ((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]
        ) / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0)
    elif p <= p_high:
        q = p - 0.5
        r = q * q
        return (
            (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q
        ) / (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1.0)
    else:
        q = math.sqrt(-2.0 * math.log(1.0 - p))
        return -(
            ((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]
        ) / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0)


# =====================================================================
# Student Verification Mixin
# =====================================================================

class StudentVerificationMixin:
    """Provides structured 8-step verification breakdown for student learning and auditing."""

    def get_student_verification(self) -> Dict[str, Any]:
        """Return the structured 8-section breakdown required for student verification."""
        return {
            "1. Inputs Extracted": getattr(self, "inputs_extracted", {}),
            "2. Variable Definitions": getattr(self, "variable_definitions", {}),
            "3. Formula": getattr(self, "formula", ""),
            "4. Formula with Substituted Values": getattr(self, "formula_with_substituted_values", ""),
            "5. Intermediate Calculations": getattr(self, "intermediate_calculations", {}),
            "6. Final Answer": getattr(self, "final_answer", {}),
            "7. Verification Checks": getattr(self, "verification_checks", []),
            "8. Assumptions": getattr(self, "assumptions", []),
            "inputs_extracted": getattr(self, "inputs_extracted", {}),
            "variable_definitions": getattr(self, "variable_definitions", {}),
            "formula": getattr(self, "formula", ""),
            "formula_with_substituted_values": getattr(self, "formula_with_substituted_values", ""),
            "intermediate_calculations": getattr(self, "intermediate_calculations", {}),
            "final_answer": getattr(self, "final_answer", {}),
            "verification_checks": getattr(self, "verification_checks", []),
            "assumptions": getattr(self, "assumptions", []),
        }

    def format_student_verification_markdown(self) -> str:
        """Format the 8-step verification breakdown into clean, professional Markdown."""
        data = self.get_student_verification()
        lines: List[str] = [
            "## 🎓 Step-by-Step Student Verification Breakdown\n",
            "### 1. Inputs Extracted",
        ]
        inputs_dict = data["inputs_extracted"]
        if inputs_dict:
            for k, v in inputs_dict.items():
                if k == "discount_tiers" and isinstance(v, list):
                    lines.append(f"- **{k}:** {len(v)} price break tiers:")
                    for t in v:
                        lines.append(f"  - Min Qty: `{t.get('min_qty')}`, Max Qty: `{t.get('max_qty')}`, Unit Price: `${t.get('price')}`")
                else:
                    lines.append(f"- **{k}:** `{v}`")
        else:
            lines.append("- *(No inputs specified)*")
        lines.append("")

        lines.append("### 2. Variable Definitions")
        for sym, desc in data["variable_definitions"].items():
            lines.append(f"- **{sym}:** {desc}")
        lines.append("")

        lines.append("### 3. Formula")
        lines.append("```text")
        lines.append(str(data["formula"]).strip())
        lines.append("```\n")

        lines.append("### 4. Formula with Substituted Values")
        lines.append("```text")
        lines.append(str(data["formula_with_substituted_values"]).strip())
        lines.append("```\n")

        lines.append("### 5. Intermediate Calculations")
        for k, v in data["intermediate_calculations"].items():
            if isinstance(v, list):
                lines.append(f"- **{k}:**")
                for item in v:
                    lines.append(f"  - `{item}`")
            elif isinstance(v, dict):
                lines.append(f"- **{k}:**")
                for sub_k, sub_v in v.items():
                    lines.append(f"  - {sub_k}: `{sub_v}`")
            else:
                lines.append(f"- **{k}:** `{v}`")
        lines.append("")

        lines.append("### 6. Final Answer")
        for k, v in data["final_answer"].items():
            lines.append(f"- **{k}:** `{v}`")
        lines.append("")

        lines.append("### 7. Verification Checks")
        for check in data["verification_checks"]:
            status_icon = "PASSED ✅" if check.get("passed") else "FAILED ❌"
            lines.append(f"- **{check.get('check')}:** {status_icon} — *{check.get('details')}*")
        lines.append("")

        lines.append("### 8. Assumptions")
        for idx, asm in enumerate(data["assumptions"], 1):
            lines.append(f"{idx}. {asm}")
        lines.append("")

        return "\n".join(lines)


# =====================================================================
# 1. Economic Order Quantity (EOQ) Model
# =====================================================================

@dataclass
class EOQResult(StudentVerificationMixin):
    """Structured output metrics for the Economic Order Quantity model."""
    optimal_order_quantity: float
    annual_demand: float
    order_cost: float
    holding_cost_per_unit: float
    unit_cost: Optional[float]
    annual_ordering_cost: float
    annual_holding_cost: float
    total_inventory_cost: float
    annual_purchasing_cost: Optional[float]
    total_annual_cost: float
    orders_per_year: float
    cycle_time_years: float
    cycle_time_days: Optional[float]
    average_cycle_inventory: float
    reorder_point_deterministic: Optional[float]
    lead_time_days: Optional[float]
    intermediate_calculations: Dict[str, Any]
    final_answer: Dict[str, Any]
    assumptions: List[str]
    inputs_extracted: Dict[str, Any] = field(default_factory=dict)
    variable_definitions: Dict[str, str] = field(default_factory=dict)
    formula: str = ""
    formula_with_substituted_values: str = ""
    verification_checks: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def calculate_eoq(
    demand: Optional[float] = None,
    order_cost: Optional[float] = None,
    holding_cost: Optional[float] = None,
    unit_cost: Optional[float] = None,
    lead_time_days: Optional[float] = None,
    work_days_per_year: float = 250.0,
    **kwargs,
) -> EOQResult:
    """Calculate the Economic Order Quantity Q = sqrt(2DS/H) and cost breakdown.

    Strictly validates inputs and never invents missing numerical values.

    Args:
        demand: Annual demand (D, units/year). Must be strictly positive.
        order_cost: Setup or ordering cost per order (S, $). Must be strictly positive.
        holding_cost: Annual holding cost per unit (H, $/unit/year). Must be strictly positive.
        unit_cost: Unit purchase cost (C, $/unit, optional).
        lead_time_days: Supplier lead time in days (optional).
        work_days_per_year: Number of operating workdays per year (default 250).

    Returns:
        EOQResult with inputs, intermediate calculations, final answer, and assumptions.
    """
    # Strict input validation - never invent missing numerical inputs
    if demand is None:
        raise ValueError("Missing required input 'demand'. Numerical annual demand must be provided.")
    if not isinstance(demand, (int, float)) or math.isnan(demand) or demand <= 0:
        raise ValueError(f"Demand must be a positive number (> 0), got {demand}.")

    if order_cost is None:
        raise ValueError("Missing required input 'order_cost'. Setup/ordering cost per order must be provided.")
    if not isinstance(order_cost, (int, float)) or math.isnan(order_cost) or order_cost <= 0:
        raise ValueError(f"Order cost must be a positive number (> 0), got {order_cost}.")

    if holding_cost is None:
        raise ValueError("Missing required input 'holding_cost'. Unit holding cost per year must be provided.")
    if not isinstance(holding_cost, (int, float)) or math.isnan(holding_cost) or holding_cost <= 0:
        raise ValueError(f"Holding cost must be a positive number (> 0), got {holding_cost}.")

    if unit_cost is not None:
        if not isinstance(unit_cost, (int, float)) or math.isnan(unit_cost) or unit_cost < 0:
            raise ValueError(f"Unit cost cannot be negative, got {unit_cost}.")

    if lead_time_days is not None:
        if not isinstance(lead_time_days, (int, float)) or math.isnan(lead_time_days) or lead_time_days < 0:
            raise ValueError(f"Lead time days cannot be negative, got {lead_time_days}.")

    if work_days_per_year <= 0:
        raise ValueError(f"Workdays per year must be positive, got {work_days_per_year}.")

    # Deterministic Formula: Q = sqrt(2DS / H)
    numerator = 2.0 * float(demand) * float(order_cost)
    q_star = math.sqrt(numerator / float(holding_cost))

    # Intermediate Calculations
    orders_per_year = demand / q_star
    annual_ordering_cost = orders_per_year * order_cost
    average_cycle_inventory = q_star / 2.0
    annual_holding_cost = average_cycle_inventory * holding_cost
    total_inventory_cost = annual_ordering_cost + annual_holding_cost

    annual_purchasing_cost = (demand * unit_cost) if unit_cost is not None else None
    total_annual_cost = total_inventory_cost + (annual_purchasing_cost or 0.0)

    cycle_time_years = q_star / demand
    cycle_time_days = cycle_time_years * work_days_per_year

    reorder_point = None
    if lead_time_days is not None:
        daily_demand = demand / work_days_per_year
        reorder_point = daily_demand * lead_time_days

    intermediate = {
        "numerator_2DS": round(numerator, 4),
        "unrounded_Q": q_star,
        "orders_per_year": round(orders_per_year, 4),
        "annual_ordering_cost": round(annual_ordering_cost, 2),
        "average_cycle_inventory": round(average_cycle_inventory, 2),
        "annual_holding_cost": round(annual_holding_cost, 2),
        "total_inventory_cost": round(total_inventory_cost, 2),
        "annual_purchasing_cost": round(annual_purchasing_cost, 2) if annual_purchasing_cost is not None else None,
        "cycle_time_days": round(cycle_time_days, 2),
        "reorder_point_deterministic": round(reorder_point, 2) if reorder_point is not None else None,
    }

    final_answer = {
        "optimal_order_quantity": round(q_star, 2),
        "total_inventory_cost": round(total_inventory_cost, 2),
        "orders_per_year": round(orders_per_year, 2),
        "total_annual_cost": round(total_annual_cost, 2),
    }

    assumptions = [
        "Annual demand (D) is deterministic, constant, and uniformly distributed over time.",
        "Ordering/setup cost (S) is fixed and invariant to order quantity.",
        "Unit inventory holding cost (H) is linear and constant per unit per year.",
        "Replenishment is instantaneous: entire order quantity arrives in a single batch.",
        "No shortages or stockouts are permitted; all demand must be met from stock.",
        "Purchasing price is constant with no volume price breaks.",
    ]

    # 1. Inputs Extracted
    inputs_extracted = {
        "annual_demand_D": float(demand),
        "ordering_cost_S": float(order_cost),
        "holding_cost_H": float(holding_cost),
    }
    if unit_cost is not None:
        inputs_extracted["unit_purchase_cost_C"] = float(unit_cost)
    if lead_time_days is not None:
        inputs_extracted["lead_time_days_L"] = float(lead_time_days)
    if work_days_per_year != 250.0:
        inputs_extracted["work_days_per_year"] = float(work_days_per_year)

    # 2. Variable Definitions
    variable_definitions = {
        "D": "Annual demand rate (units/year)",
        "S": "Ordering or setup cost incurred per order ($/order)",
        "H": "Holding or carrying cost per unit per year ($/unit/year)",
        "Q": "Order quantity or batch size (units)",
        "Q*": "Economic Order Quantity minimizing total inventory holding and ordering costs (units)",
        "N": "Order frequency (orders/year): N = D / Q*",
        "T": "Cycle time between orders (years): T = Q* / D",
        "AOC": "Annual Ordering Cost ($/year): AOC = (D / Q*) * S",
        "AHC": "Annual Holding Cost ($/year): AHC = (Q* / 2) * H",
        "TIC": "Total Annual Inventory Cost ($/year): TIC = AOC + AHC",
    }
    if unit_cost is not None:
        variable_definitions["C"] = "Unit purchase price ($/unit)"
        variable_definitions["APC"] = "Annual Purchasing Cost ($/year): APC = D * C"
        variable_definitions["TAC"] = "Total Annual Cost ($/year): TAC = TIC + APC"

    # 3. Formula
    formula = (
        "Optimal Order Quantity: Q* = sqrt((2 * D * S) / H)\n"
        "Orders per Year:        N = D / Q*\n"
        "Annual Ordering Cost:   AOC = (D / Q*) * S\n"
        "Annual Holding Cost:    AHC = (Q* / 2) * H\n"
        "Total Inventory Cost:   TIC = AOC + AHC = (D / Q*) * S + (Q* / 2) * H"
    )

    # 4. Formula with Substituted Values
    formula_with_substituted_values = (
        f"Q* = sqrt((2 * {demand:g} * {order_cost:g}) / {holding_cost:g})\n"
        f"   = sqrt({numerator:g} / {holding_cost:g})\n"
        f"   = sqrt({(numerator / holding_cost):g})\n"
        f"   = {q_star:.2f} units\n"
        f"Orders per Year (N)  = {demand:g} / {q_star:.2f} = {orders_per_year:.2f} orders/year\n"
        f"Annual Ordering Cost = ({demand:g} / {q_star:.2f}) * ${order_cost:g} = {orders_per_year:.2f} * ${order_cost:g} = ${annual_ordering_cost:.2f}\n"
        f"Annual Holding Cost  = ({q_star:.2f} / 2) * ${holding_cost:g} = {average_cycle_inventory:.2f} * ${holding_cost:g} = ${annual_holding_cost:.2f}\n"
        f"Total Inventory Cost = ${annual_ordering_cost:.2f} + ${annual_holding_cost:.2f} = ${total_inventory_cost:.2f}"
    )

    # 7. Verification Checks
    cost_diff = abs(annual_ordering_cost - annual_holding_cost)
    is_balanced = cost_diff <= 1.0 or (cost_diff / max(1.0, annual_holding_cost)) < 0.01
    second_deriv = (2.0 * demand * order_cost) / (q_star ** 3)

    verification_checks = [
        {
            "check": "Positive Order Quantity",
            "passed": q_star > 0,
            "details": f"Q* = {round(q_star, 2)} > 0 units",
        },
        {
            "check": "First-Order Optimality (AOC == AHC at EOQ)",
            "passed": is_balanced,
            "details": f"Ordering Cost (${round(annual_ordering_cost, 2)}) vs Holding Cost (${round(annual_holding_cost, 2)}), delta = ${round(cost_diff, 2)}",
        },
        {
            "check": "Second-Order Convexity (d^2TC/dQ^2 > 0)",
            "passed": second_deriv > 0,
            "details": f"d^2TC/dQ^2 = 2DS/Q^3 = {round(second_deriv, 6)} > 0 (strictly convex global cost minimum)",
        },
    ]

    return EOQResult(
        optimal_order_quantity=round(q_star, 2),
        annual_demand=round(demand, 2),
        order_cost=round(order_cost, 2),
        holding_cost_per_unit=round(holding_cost, 2),
        unit_cost=round(unit_cost, 2) if unit_cost is not None else None,
        annual_ordering_cost=round(annual_ordering_cost, 2),
        annual_holding_cost=round(annual_holding_cost, 2),
        total_inventory_cost=round(total_inventory_cost, 2),
        annual_purchasing_cost=round(annual_purchasing_cost, 2) if annual_purchasing_cost is not None else None,
        total_annual_cost=round(total_annual_cost, 2),
        orders_per_year=round(orders_per_year, 2),
        cycle_time_years=round(cycle_time_years, 4),
        cycle_time_days=round(cycle_time_days, 2),
        average_cycle_inventory=round(average_cycle_inventory, 2),
        reorder_point_deterministic=round(reorder_point, 2) if reorder_point is not None else None,
        lead_time_days=lead_time_days,
        intermediate_calculations=intermediate,
        final_answer=final_answer,
        assumptions=assumptions,
        inputs_extracted=inputs_extracted,
        variable_definitions=variable_definitions,
        formula=formula,
        formula_with_substituted_values=formula_with_substituted_values,
        verification_checks=verification_checks,
    )


# =====================================================================
# 2. Quantity Discount Model
# =====================================================================

@dataclass
class DiscountTierEvaluation:
    tier_index: int
    min_quantity: float
    max_quantity: Optional[float]
    unit_price: float
    holding_cost_per_unit: float
    raw_eoq: float
    realized_order_quantity: float
    is_raw_eoq_feasible: bool
    annual_purchase_cost: float
    annual_ordering_cost: float
    annual_holding_cost: float
    total_annual_cost: float

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class QuantityDiscountResult(StudentVerificationMixin):
    """Structured output metrics for the Quantity Discount Model."""
    optimal_order_quantity: float
    optimal_tier_index: int
    optimal_unit_price: float
    minimum_total_annual_cost: float
    optimal_annual_purchase_cost: float
    optimal_annual_ordering_cost: float
    optimal_annual_holding_cost: float
    annual_demand: float
    order_cost: float
    holding_cost_rate: Optional[float]
    tier_evaluations: List[DiscountTierEvaluation]
    base_case_cost: Optional[float]
    annual_cost_savings: Optional[float]
    intermediate_calculations: Dict[str, Any]
    final_answer: Dict[str, Any]
    assumptions: List[str]
    inputs_extracted: Dict[str, Any] = field(default_factory=dict)
    variable_definitions: Dict[str, str] = field(default_factory=dict)
    formula: str = ""
    formula_with_substituted_values: str = ""
    verification_checks: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["tier_evaluations"] = [t.to_dict() for t in self.tier_evaluations]
        return data


def calculate_quantity_discount(
    demand: Optional[float] = None,
    order_cost: Optional[float] = None,
    discount_tiers: Optional[List[Dict[str, Any]]] = None,
    holding_cost_rate: Optional[float] = None,
    fixed_holding_cost: Optional[float] = None,
    **kwargs,
) -> QuantityDiscountResult:
    """Calculate the optimal order quantity under an All-Units Quantity Discount schedule.

    Evaluates feasible quantities and compares total annual costs across all price tiers.
    Supports fixed holding cost ($) and percentage holding cost (% of unit price).
    Strictly validates inputs and never invents missing numerical values.

    Args:
        demand: Annual demand (units/year). Must be strictly positive.
        order_cost: Setup/ordering cost per order ($). Must be strictly positive.
        discount_tiers: List of tiers with keys 'min_qty', 'max_qty' (optional or None), and 'price'.
        holding_cost_rate: Annual carrying fraction (e.g., 0.20 for 20% of unit price).
        fixed_holding_cost: Fixed annual carrying cost per unit ($/unit/year).

    Returns:
        QuantityDiscountResult comparing all tiers and recommending the global minimum cost order quantity.
    """
    # Parameter normalization for flexible holding cost inputs
    h_rate = holding_cost_rate if holding_cost_rate is not None else kwargs.get("holding_rate")
    h_fixed = fixed_holding_cost if fixed_holding_cost is not None else kwargs.get("holding_cost")

    if demand is None:
        raise ValueError("Missing required input 'demand'. Annual demand must be provided.")
    if not isinstance(demand, (int, float)) or math.isnan(demand) or demand <= 0:
        raise ValueError(f"Demand must be a positive number (> 0), got {demand}.")

    if order_cost is None:
        raise ValueError("Missing required input 'order_cost'. Order cost must be provided.")
    if not isinstance(order_cost, (int, float)) or math.isnan(order_cost) or order_cost <= 0:
        raise ValueError(f"Order cost must be a positive number (> 0), got {order_cost}.")

    if h_rate is None and h_fixed is None:
        raise ValueError(
            "Missing holding cost specification. Either 'holding_cost_rate' (e.g. 0.20) "
            "or 'fixed_holding_cost' ($/unit/year) must be provided. Missing holding costs cannot be invented."
        )

    if h_rate is not None:
        if not isinstance(h_rate, (int, float)) or math.isnan(h_rate) or h_rate <= 0:
            raise ValueError(f"Holding cost rate must be positive, got {h_rate}.")

    if h_fixed is not None:
        if not isinstance(h_fixed, (int, float)) or math.isnan(h_fixed) or h_fixed <= 0:
            raise ValueError(f"Fixed holding cost must be positive, got {h_fixed}.")

    if not discount_tiers or not isinstance(discount_tiers, list):
        raise ValueError("Missing required discount tiers. 'discount_tiers' must be a non-empty list of price tiers.")

    # Sort tiers by min_qty ascending
    sorted_tiers = sorted(discount_tiers, key=lambda t: t.get("min_qty", 0))
    evaluations: List[DiscountTierEvaluation] = []

    for idx, tier in enumerate(sorted_tiers):
        if "price" not in tier or tier["price"] is None:
            raise ValueError(f"Tier {idx + 1} is missing required 'price'.")
        price = float(tier["price"])
        if price <= 0:
            raise ValueError(f"Price for tier {idx + 1} must be positive, got {price}.")

        min_q = float(tier.get("min_qty", 0))
        max_q = float(tier["max_qty"]) if tier.get("max_qty") is not None else None

        if min_q < 0:
            raise ValueError(f"Tier {idx + 1} min_qty cannot be negative, got {min_q}.")
        if max_q is not None and max_q < min_q:
            raise ValueError(f"Tier {idx + 1} max_qty ({max_q}) cannot be less than min_qty ({min_q}).")

        # Determine holding cost for this price tier
        h_tier = (holding_cost_rate * price) if holding_cost_rate is not None else float(fixed_holding_cost)  # type: ignore

        # Unconstrained EOQ for this price tier
        raw_eoq = math.sqrt((2.0 * demand * order_cost) / h_tier)

        # Check feasibility within the bracket [min_q, max_q]
        is_feasible = (raw_eoq >= min_q) and (max_q is None or raw_eoq <= max_q)

        if is_feasible:
            candidate_q = raw_eoq
        elif raw_eoq < min_q:
            candidate_q = min_q
        else:
            candidate_q = raw_eoq

        pur_cost = demand * price
        ord_cost = (demand / candidate_q) * order_cost
        hld_cost = (candidate_q / 2.0) * h_tier
        tot_cost = pur_cost + ord_cost + hld_cost

        evaluations.append(
            DiscountTierEvaluation(
                tier_index=idx + 1,
                min_quantity=min_q,
                max_quantity=max_q,
                unit_price=round(price, 4),
                holding_cost_per_unit=round(h_tier, 4),
                raw_eoq=round(raw_eoq, 2),
                realized_order_quantity=round(candidate_q, 2),
                is_raw_eoq_feasible=is_feasible,
                annual_purchase_cost=round(pur_cost, 2),
                annual_ordering_cost=round(ord_cost, 2),
                annual_holding_cost=round(hld_cost, 2),
                total_annual_cost=round(tot_cost, 2),
            )
        )

    best_eval = min(evaluations, key=lambda e: e.total_annual_cost)
    base_case_cost = evaluations[0].total_annual_cost
    savings = base_case_cost - best_eval.total_annual_cost

    intermediate = {
        "holding_cost_mode": "percentage_rate" if holding_cost_rate is not None else "fixed_per_unit",
        "evaluated_tiers": [t.to_dict() for t in evaluations],
        "base_case_cost": round(base_case_cost, 2),
        "annual_cost_savings": round(savings, 2),
    }

    final_answer = {
        "optimal_order_quantity": best_eval.realized_order_quantity,
        "optimal_tier_index": best_eval.tier_index,
        "optimal_unit_price": best_eval.unit_price,
        "minimum_total_annual_cost": best_eval.total_annual_cost,
        "annual_cost_savings": round(savings, 2),
    }

    assumptions = [
        "All-Units discount applies: price discount applies to every unit in the order once bracket minimum is met.",
        "Annual demand is deterministic and constant over the planning horizon.",
        "Working capital and storage capacity are sufficient to accommodate larger batch sizes.",
        "Holding cost is either constant per unit or proportional to the discounted purchase price.",
    ]

    # 1. Inputs Extracted
    inputs_extracted = {
        "annual_demand_D": float(demand),
        "ordering_cost_S": float(order_cost),
        "discount_tiers": discount_tiers,
    }
    if h_rate is not None:
        inputs_extracted["holding_cost_rate_I"] = float(h_rate)
    if h_fixed is not None:
        inputs_extracted["fixed_holding_cost_H"] = float(h_fixed)

    # 2. Variable Definitions
    variable_definitions = {
        "D": "Annual demand (units/year)",
        "S": "Ordering / Setup cost per order ($/order)",
        "P_i": "Unit purchase price for tier i ($/unit)",
        "H_i": "Carrying cost per unit per year for tier i ($/unit/year): H_i = I * P_i (or fixed H)",
        "Raw EOQ_i": "Unconstrained economic order quantity for tier i: sqrt((2 * D * S) / H_i)",
        "Candidate Q_i": "Feasible lot size for tier i: Raw EOQ if within [min_qty, max_qty], else min_qty",
        "Purchase Cost": "Annual procurement expense: D * P_i ($/year)",
        "Ordering Cost": "Annual order setup expense: (D / Candidate Q_i) * S ($/year)",
        "Holding Cost": "Annual inventory carrying expense: (Candidate Q_i / 2) * H_i ($/year)",
        "Total Annual Cost (TAC)": "Purchase Cost + Ordering Cost + Holding Cost ($/year)",
    }

    # 3. Formula
    formula = (
        "Tier Holding Cost:   H_i = (I * P_i) if percentage rate else fixed_H\n"
        "Unconstrained EOQ:   Raw EOQ_i = sqrt((2 * D * S) / H_i)\n"
        "Feasible Lot Size:   Candidate Q_i = max(Raw EOQ_i, min_qty_i) [if Raw EOQ_i <= max_qty_i]\n"
        "Annual Purchase Cost: APC_i = D * P_i\n"
        "Annual Ordering Cost: AOC_i = (D / Q_i) * S\n"
        "Annual Holding Cost:  AHC_i = (Q_i / 2) * H_i\n"
        "Total Annual Cost:    TAC_i = APC_i + AOC_i + AHC_i\n"
        "Optimal Decision:    Select tier i* with minimum TAC_i"
    )

    # 4. Formula with Substituted Values (Winning Tier & Comparisons)
    best_h_sub = f"({h_rate:g} * ${best_eval.unit_price:g} = ${best_eval.holding_cost_per_unit:.2f})" if h_rate is not None else f"${best_eval.holding_cost_per_unit:.2f}"
    formula_with_substituted_values = (
        f"Optimal Tier: Tier {best_eval.tier_index} (Price = ${best_eval.unit_price:g}/unit, Bracket = [{best_eval.min_quantity:g}, {best_eval.max_quantity or 'inf'}])\n"
        f"Holding Cost H = {best_h_sub}\n"
        f"Raw EOQ = sqrt((2 * {demand:g} * {order_cost:g}) / {best_eval.holding_cost_per_unit:g}) = {best_eval.raw_eoq:.2f} units\n"
        f"Candidate Lot Size Q = {best_eval.realized_order_quantity:.2f} units (Feasibility adjustment = {'Raw EOQ inside bracket' if best_eval.is_raw_eoq_feasible else 'Adjusted up to bracket minimum'})\n"
        f"Total Annual Cost breakdown:\n"
        f"  Purchase Cost: {demand:g} * ${best_eval.unit_price:g} = ${best_eval.annual_purchase_cost:.2f}\n"
        f"  Ordering Cost: ({demand:g} / {best_eval.realized_order_quantity:.2f}) * ${order_cost:g} = ${best_eval.annual_ordering_cost:.2f}\n"
        f"  Holding Cost:  ({best_eval.realized_order_quantity:.2f} / 2) * ${best_eval.holding_cost_per_unit:g} = ${best_eval.annual_holding_cost:.2f}\n"
        f"  Total Cost:    ${best_eval.annual_purchase_cost:.2f} + ${best_eval.annual_ordering_cost:.2f} + ${best_eval.annual_holding_cost:.2f} = ${best_eval.total_annual_cost:.2f}\n"
        f"Net Annual Savings vs Base Case: ${base_case_cost:.2f} - ${best_eval.total_annual_cost:.2f} = ${savings:.2f}"
    )

    # 7. Verification Checks
    is_min_compliant = best_eval.realized_order_quantity >= best_eval.min_quantity
    is_max_compliant = best_eval.max_quantity is None or best_eval.realized_order_quantity <= best_eval.max_quantity
    is_global_min = all(best_eval.total_annual_cost <= e.total_annual_cost for e in evaluations)

    verification_checks = [
        {
            "check": "Tier Minimum Quantity Compliance",
            "passed": is_min_compliant,
            "details": f"Candidate Q ({best_eval.realized_order_quantity}) >= Tier Min ({best_eval.min_quantity})",
        },
        {
            "check": "Tier Maximum Quantity Compliance",
            "passed": is_max_compliant,
            "details": f"Candidate Q ({best_eval.realized_order_quantity}) <= Tier Max ({best_eval.max_quantity if best_eval.max_quantity is not None else 'infinity'})",
        },
        {
            "check": "Global Total Cost Minimization",
            "passed": is_global_min,
            "details": f"Selected Tier {best_eval.tier_index} Total Cost (${best_eval.total_annual_cost:.2f}) is lowest among all {len(evaluations)} price tiers",
        },
        {
            "check": "Cost Savings Consistency",
            "passed": math.isclose(savings, base_case_cost - best_eval.total_annual_cost, abs_tol=0.01),
            "details": f"Savings = ${savings:.2f} (Base Tier 1 ${base_case_cost:.2f} - Tier {best_eval.tier_index} ${best_eval.total_annual_cost:.2f})",
        },
    ]

    return QuantityDiscountResult(
        optimal_order_quantity=best_eval.realized_order_quantity,
        optimal_tier_index=best_eval.tier_index,
        optimal_unit_price=best_eval.unit_price,
        minimum_total_annual_cost=best_eval.total_annual_cost,
        optimal_annual_purchase_cost=best_eval.annual_purchase_cost,
        optimal_annual_ordering_cost=best_eval.annual_ordering_cost,
        optimal_annual_holding_cost=best_eval.annual_holding_cost,
        annual_demand=round(demand, 2),
        order_cost=round(order_cost, 2),
        holding_cost_rate=holding_cost_rate,
        tier_evaluations=evaluations,
        base_case_cost=round(base_case_cost, 2),
        annual_cost_savings=round(savings, 2),
        intermediate_calculations=intermediate,
        final_answer=final_answer,
        assumptions=assumptions,
        inputs_extracted=inputs_extracted,
        variable_definitions=variable_definitions,
        formula=formula,
        formula_with_substituted_values=formula_with_substituted_values,
        verification_checks=verification_checks,
    )


# =====================================================================
# 3. Safety Stock and Reorder Point (ROP) Model
# =====================================================================

@dataclass
class ROPResult(StudentVerificationMixin):
    """Structured output metrics for Safety Stock and Reorder Point."""
    reorder_point: float
    safety_stock: float
    expected_demand_during_lead_time: float
    z_score: Optional[float]
    target_service_level: Optional[float]
    combined_lead_time_std_dev: float
    lead_time: float
    average_demand_per_period: float
    std_demand_per_period: float
    std_lead_time: float
    variability_scenario: str
    annual_safety_stock_holding_cost: Optional[float]
    expected_units_short_per_cycle: Optional[float]
    implied_fill_rate: Optional[float]
    intermediate_calculations: Dict[str, Any]
    final_answer: Dict[str, Any]
    assumptions: List[str]
    inputs_extracted: Dict[str, Any] = field(default_factory=dict)
    variable_definitions: Dict[str, str] = field(default_factory=dict)
    formula: str = ""
    formula_with_substituted_values: str = ""
    verification_checks: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# Alias for backward compatibility
SafetyStockROPResult = ROPResult


def calculate_rop(
    avg_demand: Optional[float] = None,
    lead_time: Optional[float] = None,
    service_level: Optional[float] = None,
    std_demand: Optional[float] = None,
    std_lead_time: Optional[float] = None,
    unit_holding_cost: Optional[float] = None,
    order_quantity: Optional[float] = None,
    *,
    avg_demand_per_period: Optional[float] = None,
    lead_time_periods: Optional[float] = None,
    std_demand_per_period: Optional[float] = None,
    std_lead_time_periods: Optional[float] = None,
    **kwargs,
) -> ROPResult:
    """Calculate Safety Stock and Reorder Point under continuous review (s, Q).

    Supports all 4 variability scenarios:
    1. Constant demand and constant lead time:
       sigma_LT = 0, SS = 0, ROP = d * L
    2. Variable demand and constant lead time:
       sigma_LT = sigma_d * sqrt(L), SS = z * sigma_LT, ROP = d * L + SS
    3. Constant demand and variable lead time:
       sigma_LT = d * sigma_L, SS = z * sigma_LT, ROP = d * L + SS
    4. Variable demand and variable lead time:
       sigma_LT = sqrt(L * sigma_d^2 + d^2 * sigma_L^2)
       SS = z * sigma_LT
       ROP = d * L + SS

    Strictly validates inputs and never invents missing numerical values.

    Args:
        avg_demand: Average demand per period (d).
        lead_time: Replenishment lead time in the same periods (L).
        service_level: Desired cycle service level (CSL) in (0, 1). Required if stochastic.
        std_demand: Standard deviation of demand per period (sigma_d). Default 0.
        std_lead_time: Standard deviation of lead time (sigma_L). Default 0.
        unit_holding_cost: Unit carrying cost per period/year (optional).
        order_quantity: Replenishment lot size (Q, optional for fill rate).

    Returns:
        ROPResult with ROP, SS, intermediate calculations, and assumptions.
    """
    # Parameter normalization for flexible naming
    d = avg_demand if avg_demand is not None else avg_demand_per_period
    l_time = lead_time if lead_time is not None else lead_time_periods
    s_d = std_demand if std_demand is not None else (std_demand_per_period if std_demand_per_period is not None else 0.0)
    s_l = std_lead_time if std_lead_time is not None else (std_lead_time_periods if std_lead_time_periods is not None else 0.0)

    # Input validations - never invent missing numerical inputs
    if d is None:
        raise ValueError("Missing required input 'avg_demand'. Average demand must be provided.")
    if not isinstance(d, (int, float)) or math.isnan(d) or d < 0:
        raise ValueError(f"Average demand must be non-negative (>= 0), got {d}.")

    if l_time is None:
        raise ValueError("Missing required input 'lead_time'. Replenishment lead time must be provided.")
    if not isinstance(l_time, (int, float)) or math.isnan(l_time) or l_time < 0:
        raise ValueError(f"Lead time must be non-negative (>= 0), got {l_time}.")

    if not isinstance(s_d, (int, float)) or math.isnan(s_d) or s_d < 0:
        raise ValueError(f"Demand standard deviation cannot be negative, got {s_d}.")

    if not isinstance(s_l, (int, float)) or math.isnan(s_l) or s_l < 0:
        raise ValueError(f"Lead time standard deviation cannot be negative, got {s_l}.")

    # Identify the 4 scenarios
    has_demand_var = s_d > 0
    has_lt_var = s_l > 0

    ddlt = float(d) * float(l_time)

    # If any variability exists, service level is required - cannot be invented!
    if has_demand_var or has_lt_var:
        if service_level is None:
            raise ValueError(
                "Missing required input 'service_level'. A cycle service level in (0, 1) "
                "(e.g. 0.95 for 95%) is required when demand or lead time is stochastic."
            )
        if not isinstance(service_level, (int, float)) or math.isnan(service_level) or not (0.0 < service_level < 1.0):
            raise ValueError(f"Service level must be strictly between 0 and 1 exclusive (e.g. 0.95), got {service_level}.")
        z = norm_ppf(service_level)
    else:
        z = 0.0

    # Compute sigma_LT according to the exact scenario
    if has_demand_var and has_lt_var:
        scenario = "variable_demand_variable_lead_time"
        # sigma_LT = sqrt(L * sigma_d^2 + d^2 * sigma_L^2)
        var_lt = (l_time * (s_d ** 2)) + ((d ** 2) * (s_l ** 2))
        sigma_lt = math.sqrt(var_lt)
    elif has_demand_var and not has_lt_var:
        scenario = "variable_demand_constant_lead_time"
        # sigma_LT = sigma_d * sqrt(L)
        sigma_lt = s_d * math.sqrt(l_time)
    elif not has_demand_var and has_lt_var:
        scenario = "constant_demand_variable_lead_time"
        # sigma_LT = d * sigma_L
        sigma_lt = d * s_l
    else:
        scenario = "constant_demand_constant_lead_time"
        sigma_lt = 0.0

    safety_stock = z * sigma_lt if sigma_lt > 0 else 0.0
    rop = ddlt + safety_stock

    # Optional shortage and fill rate metrics
    esc = None
    fill_rate = None
    if sigma_lt > 0 and z > 0:
        loss_fn_g = norm_pdf(z) - z * (1.0 - norm_cdf(z))
        esc = sigma_lt * loss_fn_g
        if order_quantity and order_quantity > 0:
            fill_rate = max(0.0, min(1.0, 1.0 - (esc / order_quantity)))

    ss_holding_cost = (safety_stock * unit_holding_cost) if unit_holding_cost is not None else None

    intermediate = {
        "expected_demand_during_lead_time_DDLT": round(ddlt, 4),
        "lead_time_std_dev_sigma_LT": round(sigma_lt, 4),
        "z_score": round(z, 4) if (has_demand_var or has_lt_var) else None,
        "safety_stock_SS": round(safety_stock, 4),
        "variability_scenario": scenario,
        "expected_shortage_per_cycle": round(esc, 4) if esc is not None else None,
        "implied_fill_rate": round(fill_rate, 4) if fill_rate is not None else None,
    }

    final_answer = {
        "reorder_point": round(rop, 2),
        "safety_stock": round(safety_stock, 2),
        "variability_scenario": scenario,
    }

    assumptions = [
        "Continuous review (s, Q) policy: inventory level is tracked in real-time.",
        "When stochastic, demand per period follows a stationary, independent normal distribution.",
        "Supplier replenishment lead time is independent of customer demand.",
        "Unmet customer demand during stockout periods is completely backordered.",
        "Service level is specified as Cycle Service Level (CSL, Type 1 service level).",
    ]

    # 1. Inputs Extracted
    inputs_extracted = {
        "average_demand_per_period_d": float(d),
        "lead_time_periods_L": float(l_time),
    }
    if service_level is not None:
        inputs_extracted["cycle_service_level_CSL"] = float(service_level)
    if s_d > 0:
        inputs_extracted["std_dev_demand_sigma_d"] = float(s_d)
    if s_l > 0:
        inputs_extracted["std_dev_lead_time_sigma_L"] = float(s_l)
    if unit_holding_cost is not None:
        inputs_extracted["unit_holding_cost"] = float(unit_holding_cost)
    if order_quantity is not None:
        inputs_extracted["order_quantity_Q"] = float(order_quantity)

    # 2. Variable Definitions
    variable_definitions = {
        "d": "Average customer demand rate per period (units/period)",
        "L": "Average supplier replenishment lead time (periods)",
        "sigma_d": "Standard deviation of demand per period (units/period)",
        "sigma_L": "Standard deviation of supplier lead time (periods)",
        "sigma_LT": "Combined standard deviation of demand during lead time (units)",
        "CSL": "Cycle Service Level / non-stockout probability during lead time",
        "z": "Standard normal safety factor corresponding to CSL: z = Phi^-1(CSL)",
        "DDLT": "Expected Demand During Lead Time: DDLT = d * L (units)",
        "SS": "Safety Stock buffer to absorb variability: SS = z * sigma_LT (units)",
        "ROP": "Reorder Point threshold triggering replenishment: ROP = DDLT + SS (units)",
    }

    # 3. Formula
    if not has_demand_var and not has_lt_var:
        formula = (
            "Scenario 1: Constant Demand & Constant Lead Time\n"
            "Demand During Lead Time: DDLT = d * L\n"
            "Lead Time Std Dev:       sigma_LT = 0\n"
            "Safety Stock:             SS = 0\n"
            "Reorder Point:           ROP = DDLT = d * L"
        )
        formula_sub = (
            f"DDLT = {d:g} * {l_time:g} = {ddlt:.2f} units\n"
            f"sigma_LT = 0.0 units (deterministic demand and lead time)\n"
            f"SS = 0.0 units (no safety stock needed when variability is zero)\n"
            f"ROP = {ddlt:.2f} + 0.0 = {rop:.2f} units"
        )
    elif has_demand_var and not has_lt_var:
        formula = (
            "Scenario 2: Variable Demand & Constant Lead Time\n"
            "Demand During Lead Time: DDLT = d * L\n"
            "Lead Time Std Dev:       sigma_LT = sigma_d * sqrt(L)\n"
            "Safety Factor:           z = Phi^-1(CSL)\n"
            "Safety Stock:             SS = z * sigma_LT = z * sigma_d * sqrt(L)\n"
            "Reorder Point:           ROP = DDLT + SS = (d * L) + (z * sigma_d * sqrt(L))"
        )
        formula_sub = (
            f"DDLT = {d:g} * {l_time:g} = {ddlt:.2f} units\n"
            f"sigma_LT = {s_d:g} * sqrt({l_time:g}) = {s_d:g} * {math.sqrt(l_time):.4f} = {sigma_lt:.2f} units\n"
            f"Safety Factor z = Phi^-1({service_level:.4f}) = {z:.4f}\n"
            f"SS = {z:.4f} * {sigma_lt:.2f} = {safety_stock:.2f} units\n"
            f"ROP = {ddlt:.2f} + {safety_stock:.2f} = {rop:.2f} units"
        )
    elif not has_demand_var and has_lt_var:
        formula = (
            "Scenario 3: Constant Demand & Variable Lead Time\n"
            "Demand During Lead Time: DDLT = d * L\n"
            "Lead Time Std Dev:       sigma_LT = d * sigma_L\n"
            "Safety Factor:           z = Phi^-1(CSL)\n"
            "Safety Stock:             SS = z * sigma_LT = z * d * sigma_L\n"
            "Reorder Point:           ROP = DDLT + SS = (d * L) + (z * d * sigma_L)"
        )
        formula_sub = (
            f"DDLT = {d:g} * {l_time:g} = {ddlt:.2f} units\n"
            f"sigma_LT = {d:g} * {s_l:g} = {sigma_lt:.2f} units\n"
            f"Safety Factor z = Phi^-1({service_level:.4f}) = {z:.4f}\n"
            f"SS = {z:.4f} * {sigma_lt:.2f} = {safety_stock:.2f} units\n"
            f"ROP = {ddlt:.2f} + {safety_stock:.2f} = {rop:.2f} units"
        )
    else:
        formula = (
            "Scenario 4: Variable Demand & Variable Lead Time\n"
            "Demand During Lead Time: DDLT = d * L\n"
            "Lead Time Std Dev:       sigma_LT = sqrt(L * sigma_d^2 + d^2 * sigma_L^2)\n"
            "Safety Factor:           z = Phi^-1(CSL)\n"
            "Safety Stock:             SS = z * sigma_LT = z * sqrt(L * sigma_d^2 + d^2 * sigma_L^2)\n"
            "Reorder Point:           ROP = DDLT + SS = (d * L) + SS"
        )
        term1 = l_time * (s_d ** 2)
        term2 = (d ** 2) * (s_l ** 2)
        formula_sub = (
            f"DDLT = {d:g} * {l_time:g} = {ddlt:.2f} units\n"
            f"sigma_LT = sqrt({l_time:g} * ({s_d:g})^2 + ({d:g})^2 * ({s_l:g})^2)\n"
            f"         = sqrt({term1:.2f} + {term2:.2f})\n"
            f"         = sqrt({(term1 + term2):.2f}) = {sigma_lt:.2f} units\n"
            f"Safety Factor z = Phi^-1({service_level:.4f}) = {z:.4f}\n"
            f"SS = {z:.4f} * {sigma_lt:.2f} = {safety_stock:.2f} units\n"
            f"ROP = {ddlt:.2f} + {safety_stock:.2f} = {rop:.2f} units"
        )

    # 4. Formula with Substituted Values
    formula_with_substituted_values = formula_sub

    # 7. Verification Checks
    verification_checks = [
        {
            "check": "Reorder Point Exceeds Expected Lead Time Demand",
            "passed": rop >= ddlt,
            "details": f"ROP ({round(rop, 2)}) >= DDLT ({round(ddlt, 2)})",
        },
        {
            "check": "Non-Negative Safety Stock Buffer",
            "passed": safety_stock >= 0,
            "details": f"SS = {round(safety_stock, 2)} >= 0 units",
        },
        {
            "check": "Additive Exactness (ROP == DDLT + SS)",
            "passed": math.isclose(rop, ddlt + safety_stock, abs_tol=1e-4),
            "details": f"{round(rop, 2)} == {round(ddlt, 2)} + {round(safety_stock, 2)}",
        },
    ]
    if service_level is not None:
        verification_checks.append({
            "check": "Service Level Quantile Alignment",
            "passed": True if z is None else (z >= 0 if service_level >= 0.5 else z < 0),
            "details": f"Target CSL = {round(service_level*100, 2)}% maps to standard normal z = {round(z, 4) if z is not None else 0.0}",
        })

    return ROPResult(
        reorder_point=round(rop, 2),
        safety_stock=round(safety_stock, 2),
        expected_demand_during_lead_time=round(ddlt, 2),
        z_score=round(z, 4) if (has_demand_var or has_lt_var) else None,
        target_service_level=round(service_level, 4) if service_level is not None else None,
        combined_lead_time_std_dev=round(sigma_lt, 2),
        lead_time=round(l_time, 2),
        average_demand_per_period=round(d, 2),
        std_demand_per_period=round(s_d, 2),
        std_lead_time=round(s_l, 2),
        variability_scenario=scenario,
        annual_safety_stock_holding_cost=round(ss_holding_cost, 2) if ss_holding_cost is not None else None,
        expected_units_short_per_cycle=round(esc, 3) if esc is not None else None,
        implied_fill_rate=round(fill_rate, 4) if fill_rate is not None else None,
        intermediate_calculations=intermediate,
        final_answer=final_answer,
        assumptions=assumptions,
        inputs_extracted=inputs_extracted,
        variable_definitions=variable_definitions,
        formula=formula,
        formula_with_substituted_values=formula_with_substituted_values,
        verification_checks=verification_checks,
    )


# Backward-compatible alias
calculate_safety_stock_and_rop = calculate_rop


# =====================================================================
# 4. Fixed-Period Review Model (Order-Up-To Level)
# =====================================================================

@dataclass
class PeriodicReviewResult(StudentVerificationMixin):
    """Structured output metrics for the Periodic Review (T, S) Model."""
    target_inventory_level_S: float
    recommended_order_quantity_Q: float
    safety_stock: float
    review_period_T: float
    lead_time_L: float
    protection_interval_T_plus_L: float
    expected_demand_during_protection_interval: float
    std_dev_protection_interval: float
    z_score: float
    service_level: float
    current_inventory_position: float
    current_on_hand: float
    current_on_order: float
    current_backorders: float
    average_cycle_stock: float
    expected_average_inventory: float
    intermediate_calculations: Dict[str, Any]
    final_answer: Dict[str, Any]
    assumptions: List[str]
    inputs_extracted: Dict[str, Any] = field(default_factory=dict)
    variable_definitions: Dict[str, str] = field(default_factory=dict)
    formula: str = ""
    formula_with_substituted_values: str = ""
    verification_checks: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def calculate_periodic_review(
    avg_demand: Optional[float] = None,
    review_period: Optional[float] = None,
    lead_time: Optional[float] = None,
    service_level: Optional[float] = None,
    std_demand: Optional[float] = None,
    on_hand: float = 0.0,
    on_order: float = 0.0,
    backorders: float = 0.0,
    *,
    avg_demand_per_period: Optional[float] = None,
    review_period_T: Optional[float] = None,
    lead_time_L: Optional[float] = None,
    std_demand_per_period: Optional[float] = None,
    **kwargs,
) -> PeriodicReviewResult:
    """Calculate target inventory level (S) and replenishment order quantity (Q)
    under a periodic review (T, S) inventory control system.

    Formulas:
    - Protection period = review period + lead time (T + L)
    - Target inventory = demand during protection period + safety stock
    - Order quantity = max(0, target inventory - inventory position)

    Strictly validates inputs and never invents missing numerical values.

    Args:
        avg_demand: Average demand per period (d).
        review_period: Time between reviews (T). Must be > 0.
        lead_time: Supplier delivery lead time (L). Must be >= 0.
        service_level: Target cycle service level in (0, 1).
        std_demand: Standard deviation of demand per period (sigma_d). Must be >= 0.
        on_hand: Physically available inventory.
        on_order: Inventory on order or in transit.
        backorders: Unfulfilled customer backorders.

    Returns:
        PeriodicReviewResult with target inventory, order quantity, intermediate calculations, and assumptions.
    """
    # Parameter normalization
    d = avg_demand if avg_demand is not None else avg_demand_per_period
    t_period = review_period if review_period is not None else review_period_T
    l_time = lead_time if lead_time is not None else lead_time_L
    s_d = std_demand if std_demand is not None else std_demand_per_period

    # Input validations - never invent missing numerical inputs
    if d is None:
        raise ValueError("Missing required input 'avg_demand'. Average demand must be provided.")
    if not isinstance(d, (int, float)) or math.isnan(d) or d < 0:
        raise ValueError(f"Average demand must be non-negative (>= 0), got {d}.")

    if t_period is None:
        raise ValueError("Missing required input 'review_period'. Review period (T) must be provided.")
    if not isinstance(t_period, (int, float)) or math.isnan(t_period) or t_period <= 0:
        raise ValueError(f"Review period (T) must be strictly positive (> 0), got {t_period}.")

    if l_time is None:
        raise ValueError("Missing required input 'lead_time'. Supplier lead time (L) must be provided.")
    if not isinstance(l_time, (int, float)) or math.isnan(l_time) or l_time < 0:
        raise ValueError(f"Lead time (L) cannot be negative, got {l_time}.")

    if s_d is None:
        raise ValueError("Missing required input 'std_demand'. Standard deviation of demand must be provided.")
    if not isinstance(s_d, (int, float)) or math.isnan(s_d) or s_d < 0:
        raise ValueError(f"Standard deviation of demand must be non-negative, got {s_d}.")

    if service_level is None:
        raise ValueError("Missing required input 'service_level'. Service level in (0, 1) must be provided.")
    if not isinstance(service_level, (int, float)) or math.isnan(service_level) or not (0.0 < service_level < 1.0):
        raise ValueError(f"Service level must be strictly between 0 and 1 exclusive (e.g. 0.95), got {service_level}.")

    if on_hand < 0:
        raise ValueError(f"On-hand inventory cannot be negative, got {on_hand}.")
    if on_order < 0:
        raise ValueError(f"On-order inventory cannot be negative, got {on_order}.")
    if backorders < 0:
        raise ValueError(f"Backorders cannot be negative, got {backorders}.")

    # Protection Period = review period + lead time
    protection_period = float(t_period) + float(l_time)

    # Demand during protection period = d * (T + L)
    demand_prot = float(d) * protection_period

    # Std dev during protection period = sigma_d * sqrt(T + L)
    sigma_prot = float(s_d) * math.sqrt(protection_period)

    # Safety Stock = z * sigma_prot
    z = norm_ppf(service_level)
    safety_stock = z * sigma_prot

    # Target Inventory = demand during protection period + safety stock
    target_inventory = demand_prot + safety_stock

    # Inventory Position = on_hand + on_order - backorders
    inventory_position = float(on_hand) + float(on_order) - float(backorders)

    # Order Quantity = max(0, target inventory - inventory position)
    order_quantity = max(0.0, target_inventory - inventory_position)

    avg_cycle_stock = (float(d) * float(t_period)) / 2.0
    expected_avg_inv = safety_stock + avg_cycle_stock

    intermediate = {
        "protection_period_T_plus_L": round(protection_period, 4),
        "demand_during_protection_period": round(demand_prot, 4),
        "protection_period_std_dev": round(sigma_prot, 4),
        "z_score": round(z, 4),
        "safety_stock": round(safety_stock, 4),
        "target_inventory_level_S": round(target_inventory, 4),
        "inventory_position_IP": round(inventory_position, 4),
        "unconstrained_order_gap": round(target_inventory - inventory_position, 4),
    }

    final_answer = {
        "target_inventory": round(target_inventory, 2),
        "order_quantity": round(order_quantity, 2),
        "safety_stock": round(safety_stock, 2),
        "inventory_position": round(inventory_position, 2),
    }

    assumptions = [
        "Inventory is reviewed at fixed periodic intervals of duration T.",
        "Orders arrive after a deterministic lead time L periods.",
        "System is vulnerable to stockouts over the full protection interval of (T + L) periods.",
        "Periodic demand is independent and identically distributed normal random variable.",
        "Unmet customer demand is completely backordered until replenishment arrives.",
    ]

    # 1. Inputs Extracted
    inputs_extracted = {
        "avg_demand_per_period_d": float(d),
        "review_period_T": float(t_period),
        "lead_time_L": float(l_time),
        "cycle_service_level_CSL": float(service_level),
        "std_dev_demand_per_period_sigma_d": float(s_d),
        "current_on_hand": float(on_hand),
        "current_on_order": float(on_order),
        "current_backorders": float(backorders),
    }

    # 2. Variable Definitions
    variable_definitions = {
        "T": "Review period / time interval between order opportunities (periods)",
        "L": "Supplier replenishment delivery lead time (periods)",
        "T + L": "Protection interval / risk vulnerability window (periods)",
        "d": "Average demand rate per replenishment period (units/period)",
        "sigma_d": "Standard deviation of periodic customer demand (units/period)",
        "sigma_T_plus_L": "Standard deviation of demand over protection interval: sigma_d * sqrt(T + L)",
        "CSL": "Target cycle service level / non-stockout probability over (T + L)",
        "z": "Safety factor for CSL: z = Phi^-1(CSL)",
        "SS": "Safety stock buffer: SS = z * sigma_d * sqrt(T + L)",
        "D_prot": "Expected demand during protection interval: D_prot = d * (T + L)",
        "S": "Target inventory level / Order-Up-To ceiling: S = D_prot + SS",
        "IP": "Current inventory position: IP = On_Hand + On_Order - Backorders",
        "Q": "Recommended replenishment order quantity to place now: Q = max(0, S - IP)",
    }

    # 3. Formula
    formula = (
        "Protection Interval:          Protection Interval = T + L\n"
        "Expected Demand in Protection: D_prot = d * (T + L)\n"
        "Std Dev in Protection:        sigma_prot = sigma_d * sqrt(T + L)\n"
        "Safety Factor:                z = Phi^-1(CSL)\n"
        "Safety Stock:                 SS = z * sigma_prot = z * sigma_d * sqrt(T + L)\n"
        "Target Order-Up-To Level:     S = D_prot + SS = (d * (T + L)) + (z * sigma_d * sqrt(T + L))\n"
        "Current Inventory Position:   IP = On_Hand + On_Order - Backorders\n"
        "Order Quantity:               Q = max(0, S - IP)"
    )

    # 4. Formula with Substituted Values
    sqrt_prot = math.sqrt(protection_period)
    formula_with_substituted_values = (
        f"Protection Interval = {t_period:g} + {l_time:g} = {protection_period:.1f} periods\n"
        f"Expected Demand in Protection D_prot = {d:g} * ({t_period:g} + {l_time:g}) = {d:g} * {protection_period:.1f} = {demand_prot:.2f} units\n"
        f"Std Dev in Protection sigma_prot = {s_d:g} * sqrt({protection_period:.1f}) = {s_d:g} * {sqrt_prot:.4f} = {sigma_prot:.2f} units\n"
        f"Safety Factor z = Phi^-1({service_level:.4f}) = {z:.4f}\n"
        f"Safety Stock SS = {z:.4f} * {sigma_prot:.2f} = {safety_stock:.2f} units\n"
        f"Target Order-Up-To Level (S) = {demand_prot:.2f} + {safety_stock:.2f} = {target_inventory:.2f} units\n"
        f"Current Inventory Position (IP) = {on_hand:g} (on-hand) + {on_order:g} (on-order) - {backorders:g} (backorders) = {inventory_position:.2f} units\n"
        f"Recommended Replenishment (Q) = max(0, {target_inventory:.2f} - {inventory_position:.2f}) = {order_quantity:.2f} units"
    )

    # 7. Verification Checks
    verification_checks = [
        {
            "check": "Base Stock Conservation Check (S == D_prot + SS)",
            "passed": math.isclose(target_inventory, demand_prot + safety_stock, abs_tol=1e-3),
            "details": f"S ({round(target_inventory, 2)}) == Expected Demand ({round(demand_prot, 2)}) + Safety Stock ({round(safety_stock, 2)})",
        },
        {
            "check": "Inventory Position Identity",
            "passed": math.isclose(inventory_position, on_hand + on_order - backorders, abs_tol=1e-4),
            "details": f"IP ({round(inventory_position, 2)}) == On-Hand ({on_hand}) + On-Order ({on_order}) - Backorders ({backorders})",
        },
        {
            "check": "Non-Negative Order Quantity",
            "passed": order_quantity >= 0,
            "details": f"Order quantity Q = {round(order_quantity, 2)} units >= 0",
        },
        {
            "check": "Surplus Ordering Prevention Rule",
            "passed": (order_quantity == 0.0) if (inventory_position >= target_inventory) else (order_quantity > 0.0),
            "details": f"IP ({round(inventory_position, 2)}) vs S ({round(target_inventory, 2)}) -> Q = {round(order_quantity, 2)} units",
        },
    ]

    return PeriodicReviewResult(
        target_inventory_level_S=round(target_inventory, 2),
        recommended_order_quantity_Q=round(order_quantity, 2),
        safety_stock=round(safety_stock, 2),
        review_period_T=round(t_period, 2),
        lead_time_L=round(l_time, 2),
        protection_interval_T_plus_L=round(protection_period, 2),
        expected_demand_during_protection_interval=round(demand_prot, 2),
        std_dev_protection_interval=round(sigma_prot, 2),
        z_score=round(z, 4),
        service_level=round(service_level, 4),
        current_inventory_position=round(inventory_position, 2),
        current_on_hand=round(on_hand, 2),
        current_on_order=round(on_order, 2),
        current_backorders=round(backorders, 2),
        average_cycle_stock=round(avg_cycle_stock, 2),
        expected_average_inventory=round(expected_avg_inv, 2),
        intermediate_calculations=intermediate,
        final_answer=final_answer,
        assumptions=assumptions,
        inputs_extracted=inputs_extracted,
        variable_definitions=variable_definitions,
        formula=formula,
        formula_with_substituted_values=formula_with_substituted_values,
        verification_checks=verification_checks,
    )


# =====================================================================
# 5. Newsvendor Model (Single-Period Stochastic Inventory)
# =====================================================================

@dataclass
class NewsvendorResult(StudentVerificationMixin):
    """Structured output metrics for the Single-Period Newsvendor model."""
    optimal_order_quantity: float
    critical_ratio: float
    cost_of_underage_Cu: float
    cost_of_overage_Co: float
    selling_price: float
    purchase_cost: float
    salvage_value: float
    shortage_penalty: float
    distribution_type: str
    mean_demand: Optional[float]
    std_demand: Optional[float]
    z_score: Optional[float]
    expected_sales: float
    expected_leftover_units: float
    expected_lost_sales: float
    expected_profit: float
    expected_profit_perfect_information: Optional[float]
    expected_value_of_perfect_information: Optional[float]
    intermediate_calculations: Dict[str, Any]
    final_answer: Dict[str, Any]
    assumptions: List[str]
    inputs_extracted: Dict[str, Any] = field(default_factory=dict)
    variable_definitions: Dict[str, str] = field(default_factory=dict)
    formula: str = ""
    formula_with_substituted_values: str = ""
    verification_checks: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def calculate_newsvendor(
    selling_price: Optional[float] = None,
    purchase_cost: Optional[float] = None,
    salvage_value: float = 0.0,
    shortage_penalty: float = 0.0,
    mean_demand: Optional[float] = None,
    std_demand: Optional[float] = None,
    discrete_demand: Optional[List[float]] = None,
    discrete_probs: Optional[List[float]] = None,
    *,
    underage_cost: Optional[float] = None,
    overage_cost: Optional[float] = None,
    cost_of_underage: Optional[float] = None,
    cost_of_overage: Optional[float] = None,
    shortage_cost: Optional[float] = None,
    excess_cost: Optional[float] = None,
    **kwargs,
) -> NewsvendorResult:
    """Calculate underage cost, overage cost, critical ratio, and recommended order quantity
    for the single-period Newsvendor problem.

    Economic Formulations:
    - Underage Cost Cu = selling_price - purchase_cost + shortage_penalty (or directly given)
    - Overage Cost Co = purchase_cost - salvage_value (or directly given)
    - Critical Ratio CR = Cu / (Cu + Co)
    - Recommended Order Quantity Q* satisfies P(Demand <= Q*) = CR

    Strictly validates inputs and never invents missing numerical values.
    """
    # Direct Cu / Co parameter binding
    direct_cu = underage_cost if underage_cost is not None else (cost_of_underage if cost_of_underage is not None else shortage_cost)
    direct_co = overage_cost if overage_cost is not None else (cost_of_overage if cost_of_overage is not None else excess_cost)

    if direct_cu is not None and direct_co is not None:
        if direct_cu <= 0:
            raise ValueError(f"Underage cost must be strictly positive, got {direct_cu}.")
        if direct_co <= 0:
            raise ValueError(f"Overage cost must be strictly positive, got {direct_co}.")
        cu = float(direct_cu)
        co = float(direct_co)
        p_eff = selling_price if selling_price is not None else (cu + co)
        c_eff = purchase_cost if purchase_cost is not None else co
        v_eff = salvage_value if salvage_value is not None else 0.0
        g_eff = shortage_penalty if shortage_penalty is not None else 0.0
    else:
        # Input validation for unit economics - never invent missing numerical inputs
        if selling_price is None:
            raise ValueError("Missing required input 'selling_price' (or direct 'underage_cost'). Selling/retail price must be provided.")
        if not isinstance(selling_price, (int, float)) or math.isnan(selling_price) or selling_price <= 0:
            raise ValueError(f"Selling price must be a positive number (> 0), got {selling_price}.")

        if purchase_cost is None:
            raise ValueError("Missing required input 'purchase_cost' (or direct 'overage_cost'). Unit purchase cost must be provided.")
        if not isinstance(purchase_cost, (int, float)) or math.isnan(purchase_cost) or purchase_cost <= 0:
            raise ValueError(f"Purchase cost must be a positive number (> 0), got {purchase_cost}.")

        if salvage_value is None:
            salvage_value = 0.0
        if not isinstance(salvage_value, (int, float)) or math.isnan(salvage_value) or salvage_value < 0:
            raise ValueError(f"Salvage value cannot be negative, got {salvage_value}.")

        if shortage_penalty is None:
            shortage_penalty = 0.0
        if not isinstance(shortage_penalty, (int, float)) or math.isnan(shortage_penalty) or shortage_penalty < 0:
            raise ValueError(f"Shortage penalty cannot be negative, got {shortage_penalty}.")

        if selling_price <= purchase_cost:
            raise ValueError(f"Selling price (${selling_price}) must be strictly greater than purchase cost (${purchase_cost}).")
        if salvage_value > purchase_cost:
            raise ValueError(f"Salvage value (${salvage_value}) cannot exceed purchase cost (${purchase_cost}).")

        # Underage Cost Cu = Selling Price - Purchase Cost + Shortage Penalty
        cu = float(selling_price) - float(purchase_cost) + float(shortage_penalty)
        # Overage Cost Co = Purchase Cost - Salvage Value
        co = float(purchase_cost) - float(salvage_value)
        p_eff = float(selling_price)
        c_eff = float(purchase_cost)
        v_eff = float(salvage_value)
        g_eff = float(shortage_penalty)

    if (cu + co) <= 0:
        raise ValueError("Sum of underage cost and overage cost must be strictly positive.")

    # Critical Ratio CR = Cu / (Cu + Co)
    critical_ratio = cu / (cu + co)

    # Discrete Distribution Scenario
    if discrete_demand is not None and discrete_probs is not None:
        if len(discrete_demand) != len(discrete_probs):
            raise ValueError("Discrete demand and probabilities lists must have identical lengths.")
        if not math.isclose(sum(discrete_probs), 1.0, rel_tol=1e-3, abs_tol=1e-3):
            raise ValueError(f"Discrete probabilities must sum to 1.0, got {sum(discrete_probs)}.")

        pairs = sorted(zip(discrete_demand, discrete_probs), key=lambda x: x[0])
        cum_prob = 0.0
        q_star = pairs[-1][0]
        for d_val, p_val in pairs:
            cum_prob += p_val
            if cum_prob >= critical_ratio:
                q_star = d_val
                break

        exp_sales = sum(min(q_star, d_pt) * p_pt for d_pt, p_pt in pairs)
        exp_leftovers = sum(max(0.0, q_star - d_pt) * p_pt for d_pt, p_pt in pairs)
        exp_lost = sum(max(0.0, d_pt - q_star) * p_pt for d_pt, p_pt in pairs)
        mean_d = sum(d_pt * p_pt for d_pt, p_pt in pairs)

        exp_profit = (
            (p_eff * exp_sales)
            + (v_eff * exp_leftovers)
            - (c_eff * q_star)
            - (g_eff * exp_lost)
        )
        evpi_profit = sum(((p_eff - c_eff) * d_pt) * p_pt for d_pt, p_pt in pairs)
        evpi = max(0.0, evpi_profit - exp_profit)

        intermediate = {
            "underage_cost_Cu": round(cu, 4),
            "overage_cost_Co": round(co, 4),
            "critical_ratio_CR": round(critical_ratio, 4),
            "distribution_type": "discrete",
            "discrete_pairs": pairs,
            "expected_sales": round(exp_sales, 2),
            "expected_leftovers": round(exp_leftovers, 2),
            "expected_lost_sales": round(exp_lost, 2),
            "expected_profit": round(exp_profit, 2),
            "evpi": round(evpi, 2),
        }

        final_answer = {
            "recommended_order_quantity": round(q_star, 2),
            "critical_ratio": round(critical_ratio, 4),
            "underage_cost": round(cu, 2),
            "overage_cost": round(co, 2),
            "expected_profit": round(exp_profit, 2),
        }

        assumptions = [
            "Single-period inventory problem: ordering occurs before season starts with no replenishment during season.",
            "Demand follows a known empirical discrete probability distribution.",
            "Unsold units at period end are salvaged at unit salvage value.",
            "Unfulfilled demand results in foregone profit and customer goodwill penalty.",
        ]

        # 1. Inputs Extracted (Discrete)
        inputs_extracted = {
            "selling_price_P": float(p_eff),
            "purchase_cost_C": float(c_eff),
            "salvage_value_V": float(v_eff),
            "shortage_penalty_g": float(g_eff),
            "cost_of_underage_Cu": float(cu),
            "cost_of_overage_Co": float(co),
            "distribution_type": "discrete",
            "discrete_demand_points": discrete_demand,
            "discrete_probabilities": discrete_probs,
        }

        # 2. Variable Definitions
        variable_definitions = {
            "P": "Selling / realization price per unit ($/unit)",
            "C": "Procurement / production cost per unit ($/unit)",
            "V": "Salvage / recovery value per unsold leftover unit ($/unit)",
            "g": "Shortage penalty / customer goodwill loss per unfulfilled unit ($/unit)",
            "Cu": "Cost of Underage (profit foregone per unmet unit): Cu = P - C + g",
            "Co": "Cost of Overage (net loss per unsold leftover unit): Co = C - V",
            "CR": "Critical Ratio / target service level: CR = Cu / (Cu + Co)",
            "D_i": "Discrete demand outcome points",
            "P(D = D_i)": "Probability mass of demand outcome D_i",
            "F(D_i)": "Cumulative probability distribution: P(Demand <= D_i)",
            "Q*": "Optimal order quantity: smallest D_i such that F(D_i) >= CR",
        }

        # 3. Formula
        formula = (
            "Cost of Underage:   Cu = P - C + g (or direct underage margin)\n"
            "Cost of Overage:    Co = C - V (or direct overage loss)\n"
            "Critical Ratio:     CR = Cu / (Cu + Co)\n"
            "Optimality Rule:    Find smallest Q* in discrete support where P(Demand <= Q*) >= CR"
        )

        # 4. Formula with Substituted Values
        formula_with_substituted_values = (
            f"Underage Cost Cu = ${p_eff:g} - ${c_eff:g} + ${g_eff:g} = ${cu:.2f}\n"
            f"Overage Cost  Co = ${c_eff:g} - ${v_eff:g} = ${co:.2f}\n"
            f"Critical Ratio CR = ${cu:.2f} / (${cu:.2f} + ${co:.2f}) = ${cu:.2f} / ${(cu + co):.2f} = {critical_ratio:.4f} ({critical_ratio*100:.2f}%)\n"
            f"Optimal Order Quantity Q* = {q_star:.2f} units (cumulative probability F({q_star:.2f}) = {cum_prob:.4f} >= {critical_ratio:.4f})\n"
            f"Expected Profit = ${exp_profit:.2f}"
        )

        # 7. Verification Checks
        verification_checks = [
            {
                "check": "Critical Ratio Probability Boundary Check (0 < CR < 1)",
                "passed": 0.0 < critical_ratio < 1.0,
                "details": f"CR = {round(critical_ratio, 4)} is within (0, 1)",
            },
            {
                "check": "Economic Penalty Positivity (Cu > 0 and Co > 0)",
                "passed": cu > 0 and co > 0,
                "details": f"Cu = ${round(cu, 2)} > 0 and Co = ${round(co, 2)} > 0",
            },
            {
                "check": "Discrete Cumulative Probability Threshold Compliance",
                "passed": cum_prob >= critical_ratio,
                "details": f"Cumulative probability F(Q*={q_star}) = {round(cum_prob, 4)} >= CR ({round(critical_ratio, 4)})",
            },
            {
                "check": "Profit Upper Bound via Perfect Information",
                "passed": exp_profit <= evpi_profit + 1e-4,
                "details": f"Expected Profit (${round(exp_profit, 2)}) <= Perfect Info Profit (${round(evpi_profit, 2)})",
            },
        ]

        return NewsvendorResult(
            optimal_order_quantity=round(q_star, 2),
            critical_ratio=round(critical_ratio, 4),
            cost_of_underage_Cu=round(cu, 2),
            cost_of_overage_Co=round(co, 2),
            selling_price=round(p_eff, 2),
            purchase_cost=round(c_eff, 2),
            salvage_value=round(v_eff, 2),
            shortage_penalty=round(g_eff, 2),
            distribution_type="discrete",
            mean_demand=round(mean_d, 2),
            std_demand=None,
            z_score=None,
            expected_sales=round(exp_sales, 2),
            expected_leftover_units=round(exp_leftovers, 2),
            expected_lost_sales=round(exp_lost, 2),
            expected_profit=round(exp_profit, 2),
            expected_profit_perfect_information=round(evpi_profit, 2),
            expected_value_of_perfect_information=round(evpi, 2),
            intermediate_calculations=intermediate,
            final_answer=final_answer,
            assumptions=assumptions,
            inputs_extracted=inputs_extracted,
            variable_definitions=variable_definitions,
            formula=formula,
            formula_with_substituted_values=formula_with_substituted_values,
            verification_checks=verification_checks,
        )

    # Continuous Normal Distribution Scenario
    if mean_demand is None:
        raise ValueError("Missing required input 'mean_demand'. Expected demand must be provided.")
    if not isinstance(mean_demand, (int, float)) or math.isnan(mean_demand) or mean_demand <= 0:
        raise ValueError(f"Mean demand must be positive (> 0), got {mean_demand}.")

    if std_demand is None:
        raise ValueError("Missing required input 'std_demand'. Standard deviation of demand must be provided.")
    if not isinstance(std_demand, (int, float)) or math.isnan(std_demand) or std_demand <= 0:
        raise ValueError(f"Standard deviation of demand must be positive (> 0), got {std_demand}.")

    z = norm_ppf(critical_ratio)
    q_star = mean_demand + z * std_demand

    loss_fn = norm_pdf(z) - z * (1.0 - norm_cdf(z))
    exp_lost = std_demand * loss_fn
    exp_sales = mean_demand - exp_lost
    exp_leftovers = q_star - exp_sales

    exp_profit = (
        (p_eff * exp_sales)
        + (v_eff * exp_leftovers)
        - (c_eff * q_star)
        - (g_eff * exp_lost)
    )

    evpi_profit = (p_eff - c_eff) * mean_demand
    evpi = max(0.0, evpi_profit - exp_profit)

    intermediate = {
        "underage_cost_Cu": round(cu, 4),
        "overage_cost_Co": round(co, 4),
        "critical_ratio_CR": round(critical_ratio, 4),
        "normal_z_score": round(z, 4),
        "expected_lost_sales": round(exp_lost, 2),
        "expected_sales": round(exp_sales, 2),
        "expected_leftovers": round(exp_leftovers, 2),
        "expected_profit": round(exp_profit, 2),
        "evpi": round(evpi, 2),
    }

    final_answer = {
        "recommended_order_quantity": round(q_star, 2),
        "critical_ratio": round(critical_ratio, 4),
        "underage_cost": round(cu, 2),
        "overage_cost": round(co, 2),
        "expected_profit": round(exp_profit, 2),
    }

    assumptions = [
        "Single-period inventory problem: ordering occurs before season starts with no replenishment during season.",
        "Demand follows a stationary, continuous normal distribution N(mean, std^2).",
        "Unsold units at period end are salvaged at unit salvage value.",
        "Unfulfilled demand results in foregone profit and customer goodwill penalty.",
        "Stocking quantity Q* optimizes the marginal trade-off: P(Demand <= Q*) = Cu / (Cu + Co).",
    ]

    # 1. Inputs Extracted (Continuous Normal)
    inputs_extracted = {
        "selling_price_P": float(p_eff),
        "purchase_cost_C": float(c_eff),
        "salvage_value_V": float(v_eff),
        "shortage_penalty_g": float(g_eff),
        "cost_of_underage_Cu": float(cu),
        "cost_of_overage_Co": float(co),
        "distribution_type": "normal",
        "mean_demand_mu": float(mean_demand),
        "std_demand_sigma": float(std_demand),
    }

    # 2. Variable Definitions
    variable_definitions = {
        "P": "Selling price / realized retail revenue per unit ($/unit)",
        "C": "Procurement / wholesale purchase cost per unit ($/unit)",
        "V": "Salvage / clearance liquidation value per unsold unit ($/unit)",
        "g": "Shortage penalty / customer goodwill loss per unfulfilled unit ($/unit)",
        "Cu": "Cost of Underage (profit margin foregone per stockout unit): Cu = P - C + g",
        "Co": "Cost of Overage (net financial loss per unsold leftover unit): Co = C - V",
        "CR": "Critical Ratio / optimal non-stockout service level probability: CR = Cu / (Cu + Co)",
        "mu": "Mean expected demand over single selling period (units)",
        "sigma": "Standard deviation of periodic customer demand (units)",
        "z*": "Standard normal critical fractile: z* = Phi^-1(CR)",
        "Q*": "Optimal order quantity: Q* = mu + z* * sigma",
    }

    # 3. Formula
    formula = (
        "Cost of Underage:   Cu = P - C + g (or direct underage penalty)\n"
        "Cost of Overage:    Co = C - V (or direct overage penalty)\n"
        "Critical Ratio:     CR = Cu / (Cu + Co)\n"
        "Critical Fractile:  z* = Phi^-1(CR)\n"
        "Optimal Order Q:    Q* = mu + z* * sigma\n"
        "Expected Profit:    E[Profit] = (P * E[Sales]) + (V * E[Leftover]) - (C * Q*) - (g * E[LostSales])"
    )

    # 4. Formula with Substituted Values
    safety_buffer = z * std_demand
    formula_with_substituted_values = (
        f"Cost of Underage Cu = ${p_eff:g} - ${c_eff:g} + ${g_eff:g} = ${cu:.2f}\n"
        f"Cost of Overage  Co = ${c_eff:g} - ${v_eff:g} = ${co:.2f}\n"
        f"Critical Ratio   CR = ${cu:.2f} / (${cu:.2f} + ${co:.2f}) = ${cu:.2f} / ${(cu + co):.2f} = {critical_ratio:.4f} ({critical_ratio*100:.2f}%)\n"
        f"Critical Fractile z* = Phi^-1({critical_ratio:.4f}) = {z:.4f}\n"
        f"Optimal Order Quantity Q* = {mean_demand:g} + ({z:.4f} * {std_demand:g})\n"
        f"                          = {mean_demand:g} + {safety_buffer:.2f}\n"
        f"                          = {q_star:.2f} units\n"
        f"Expected Sales = {exp_sales:.2f} units, Expected Leftover = {exp_leftovers:.2f} units, Expected Lost Sales = {exp_lost:.2f} units\n"
        f"Expected Profit = ${exp_profit:.2f}\n"
        f"Expected Value of Perfect Information (EVPI) = ${evpi:.2f}"
    )

    # 7. Verification Checks
    verification_checks = [
        {
            "check": "Critical Ratio Probability Boundary Check (0 < CR < 1)",
            "passed": 0.0 < critical_ratio < 1.0,
            "details": f"CR = {round(critical_ratio, 4)} is within (0, 1)",
        },
        {
            "check": "Economic Penalty Positivity (Cu > 0 and Co > 0)",
            "passed": cu > 0 and co > 0,
            "details": f"Cu = ${round(cu, 2)} > 0 and Co = ${round(co, 2)} > 0",
        },
        {
            "check": "Optimal Stocking Direction relative to Mean Demand",
            "passed": (q_star >= mean_demand) if (critical_ratio >= 0.5) else (q_star < mean_demand),
            "details": f"CR ({round(critical_ratio, 4)}) {' >= 0.5 -> Q* >= mu' if critical_ratio >= 0.5 else ' < 0.5 -> Q* < mu'}: Q* = {round(q_star, 2)} vs mu = {mean_demand}",
        },
        {
            "check": "Profit Upper Bound via Perfect Information",
            "passed": exp_profit <= evpi_profit + 1e-4,
            "details": f"Expected Profit (${round(exp_profit, 2)}) <= Perfect Info Profit (${round(evpi_profit, 2)})",
        },
    ]

    return NewsvendorResult(
        optimal_order_quantity=round(q_star, 2),
        critical_ratio=round(critical_ratio, 4),
        cost_of_underage_Cu=round(cu, 2),
        cost_of_overage_Co=round(co, 2),
        selling_price=round(p_eff, 2),
        purchase_cost=round(c_eff, 2),
        salvage_value=round(v_eff, 2),
        shortage_penalty=round(g_eff, 2),
        distribution_type="normal",
        mean_demand=round(mean_demand, 2),
        std_demand=round(std_demand, 2),
        z_score=round(z, 4),
        expected_sales=round(exp_sales, 2),
        expected_leftover_units=round(exp_leftovers, 2),
        expected_lost_sales=round(exp_lost, 2),
        expected_profit=round(exp_profit, 2),
        expected_profit_perfect_information=round(evpi_profit, 2),
        expected_value_of_perfect_information=round(evpi, 2),
        intermediate_calculations=intermediate,
        final_answer=final_answer,
        assumptions=assumptions,
        inputs_extracted=inputs_extracted,
        variable_definitions=variable_definitions,
        formula=formula,
        formula_with_substituted_values=formula_with_substituted_values,
        verification_checks=verification_checks,
    )
