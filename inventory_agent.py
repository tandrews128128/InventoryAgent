"""Supply Chain Analytics Inventory Decision Agent.

Implements the decision pipeline:
Customer Question -> Identify Model -> Extract Inputs -> Select Python Tool -> Calculate -> Verify -> Explain

Supported Models:
1. Economic Order Quantity (EOQ)
2. Quantity Discount Model
3. Safety Stock and Reorder Point (ROP)
4. Fixed-Period Review Model
5. Newsvendor Model
"""

from __future__ import annotations

import enum
import math
import re
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

from inventory_tools import (
    EOQResult,
    NewsvendorResult,
    PeriodicReviewResult,
    QuantityDiscountResult,
    SafetyStockROPResult,
    calculate_eoq,
    calculate_newsvendor,
    calculate_periodic_review,
    calculate_quantity_discount,
    calculate_safety_stock_and_rop,
)

try:
    from tabulate import tabulate
    _HAS_TABULATE = True
except ImportError:
    _HAS_TABULATE = False


class InventoryModelType(str, enum.Enum):
    EOQ = "Economic Order Quantity (EOQ)"
    QUANTITY_DISCOUNT = "Quantity Discount Model"
    SAFETY_STOCK_ROP = "Safety Stock & Reorder Point"
    PERIODIC_REVIEW = "Fixed-Period Review Model"
    NEWSVENDOR = "Newsvendor Model"
    UNKNOWN = "Unknown Model"


@dataclass
class VerificationCheck:
    name: str
    passed: bool
    details: str


@dataclass
class VerificationResult:
    is_valid: bool
    checks: List[VerificationCheck]
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "is_valid": self.is_valid,
            "checks": [asdict(c) for c in self.checks],
            "notes": self.notes,
        }


@dataclass
class AgentDecisionResponse:
    question: str
    model_type: InventoryModelType
    extracted_inputs: Dict[str, Any]
    selected_tool: str
    calculation_result: Any
    verification: Optional[VerificationResult] = None
    explanation: str = ""
    missing_inputs: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        calc_dict = (
            self.calculation_result.to_dict()
            if self.calculation_result is not None and hasattr(self.calculation_result, "to_dict")
            else self.calculation_result
        )
        return {
            "question": self.question,
            "model_type": self.model_type.value,
            "extracted_inputs": self.extracted_inputs,
            "missing_inputs": self.missing_inputs,
            "selected_tool": self.selected_tool,
            "calculation_result": calc_dict,
            "verification": self.verification.to_dict() if self.verification is not None else None,
            "explanation": self.explanation,
        }


def format_table(headers: List[str], rows: List[List[Any]]) -> str:
    """Format tabular data into Markdown."""
    if _HAS_TABULATE:
        return tabulate(rows, headers=headers, tablefmt="github")
    
    # Fallback Markdown table generator
    header_line = "| " + " | ".join(str(h) for h in headers) + " |"
    separator_line = "| " + " | ".join(["---"] * len(headers)) + " |"
    data_lines = [
        "| " + " | ".join(str(cell) for cell in row) + " |" for row in rows
    ]
    return "\n".join([header_line, separator_line] + data_lines)


class InventoryDecisionAgent:
    """Intelligent Supply Chain Inventory Decision Agent executing the 7-step pipeline."""

    def __init__(self):
        self.tool_mapping: Dict[InventoryModelType, Tuple[str, Callable]] = {
            InventoryModelType.EOQ: ("calculate_eoq", calculate_eoq),
            InventoryModelType.QUANTITY_DISCOUNT: ("calculate_quantity_discount", calculate_quantity_discount),
            InventoryModelType.SAFETY_STOCK_ROP: ("calculate_safety_stock_and_rop", calculate_safety_stock_and_rop),
            InventoryModelType.PERIODIC_REVIEW: ("calculate_periodic_review", calculate_periodic_review),
            InventoryModelType.NEWSVENDOR: ("calculate_newsvendor", calculate_newsvendor),
        }

    # =================================================================
    # STEP 1: Identify Model
    # =================================================================
    def identify_model(self, question: str) -> InventoryModelType:
        """Classify the customer's natural-language inventory question into one of 5 models."""
        q_lower = question.lower()

        # Score indicators for each model
        scores = {
            InventoryModelType.QUANTITY_DISCOUNT: 0,
            InventoryModelType.NEWSVENDOR: 0,
            InventoryModelType.PERIODIC_REVIEW: 0,
            InventoryModelType.SAFETY_STOCK_ROP: 0,
            InventoryModelType.EOQ: 0,
        }

        # Quantity Discount cues
        if any(w in q_lower for w in ["discount", "price break", "volume discount", "tier", "tiers", "all-units", "bracket", "price breaks"]):
            scores[InventoryModelType.QUANTITY_DISCOUNT] += 10
        if "tier" in q_lower or "@ $" in q_lower or "units @" in q_lower or "per unit for" in q_lower:
            scores[InventoryModelType.QUANTITY_DISCOUNT] += 5

        # Newsvendor cues
        if any(w in q_lower for w in ["newsvendor", "newsboy", "salvage", "spoilage", "perishable", "critical ratio", "single period", "season", "seasonal", "fashion"]):
            scores[InventoryModelType.NEWSVENDOR] += 10
        if ("salvage" in q_lower or "clearance" in q_lower) and ("retail" in q_lower or "selling price" in q_lower or "cost" in q_lower):
            scores[InventoryModelType.NEWSVENDOR] += 6

        # Fixed-Period Review cues
        if any(w in q_lower for w in ["periodic review", "fixed period", "fixed-period", "order up to", "order-up-to", "base stock", "review period", "protection interval", "reviewed every"]):
            scores[InventoryModelType.PERIODIC_REVIEW] += 10
        if "on-hand" in q_lower or "on hand" in q_lower or "on order" in q_lower or "in transit" in q_lower or "inventory position" in q_lower:
            scores[InventoryModelType.PERIODIC_REVIEW] += 4

        # Safety Stock & Reorder Point cues
        if any(w in q_lower for w in ["safety stock", "reorder point", "rop", "service level", "cycle service level", "csl", "stockout probability"]):
            scores[InventoryModelType.SAFETY_STOCK_ROP] += 8
        if "lead time" in q_lower and ("std" in q_lower or "deviation" in q_lower or "variance" in q_lower):
            scores[InventoryModelType.SAFETY_STOCK_ROP] += 4

        # EOQ cues
        if any(w in q_lower for w in ["eoq", "economic order quantity", "order quantity", "batch size", "lot size", "wilson"]):
            scores[InventoryModelType.EOQ] += 7
        if ("demand" in q_lower) and ("order cost" in q_lower or "ordering cost" in q_lower or "setup cost" in q_lower) and ("holding cost" in q_lower or "carrying cost" in q_lower):
            scores[InventoryModelType.EOQ] += 5

        # Disambiguate
        # If user mentions discount tiers, QUANTITY_DISCOUNT wins over EOQ
        if scores[InventoryModelType.QUANTITY_DISCOUNT] > 0 and scores[InventoryModelType.QUANTITY_DISCOUNT] >= scores[InventoryModelType.EOQ]:
            return InventoryModelType.QUANTITY_DISCOUNT

        # If user explicitly mentions periodic review or review period, PERIODIC_REVIEW wins over SAFETY_STOCK_ROP
        if scores[InventoryModelType.PERIODIC_REVIEW] > scores[InventoryModelType.SAFETY_STOCK_ROP] and scores[InventoryModelType.PERIODIC_REVIEW] > 4:
            return InventoryModelType.PERIODIC_REVIEW

        # Highest score
        sorted_scores = sorted(scores.items(), key=lambda item: item[1], reverse=True)
        top_model, top_score = sorted_scores[0]

        if top_score > 0:
            return top_model
        return InventoryModelType.UNKNOWN

    # =================================================================
    # STEP 2: Extract Inputs
    # =================================================================
    def extract_inputs(self, question: str, model_type: InventoryModelType) -> Dict[str, Any]:
        """Extract domain parameters from question text with advanced terminology recognition.

        Carefully examines the entire question and distinguishes between closely related
        supply chain concepts before declaring any parameter missing.
        """
        clean_text = question.replace(",", "")  # Remove comma in 10,000 -> 10000

        inputs: Dict[str, Any] = {}

        # Generic pattern matcher returning the first matched float
        def find_pattern(patterns: List[str]) -> Optional[float]:
            for pat in patterns:
                m = re.search(pat, clean_text, re.IGNORECASE)
                if m:
                    try:
                        return float(m.group(1))
                    except (ValueError, IndexError):
                        continue
            return None

        # -------------------------------------------------------------
        # 1. EOQ Extraction
        # -------------------------------------------------------------
        if model_type == InventoryModelType.EOQ:
            # Demand: annual demand, yearly demand, annual requirement, annual sales, etc.
            demand = find_pattern([
                r"(?:annual|yearly)\s*(?:demand|sales|requirement|usage|consumption)\s*(?:is|=|:|of)?\s*([0-9]+(?:\.[0-9]+)?)",
                r"demand\s*(?:is|=|:|of)?\s*([0-9]+(?:\.[0-9]+)?)\s*(?:units|items|cases|pieces)?\s*(?:per\s+year|annually|a\s+year)",
                r"([0-9]+(?:\.[0-9]+)?)\s*(?:units|items|cases|pieces)\s*(?:per\s+year|annually|a\s+year)",
                r"sells?\s*([0-9]+(?:\.[0-9]+)?)\s*(?:units|items|cases)?\s*(?:per\s+year|annually|a\s+year)",
                r"demand\s*(?:is|=|:|of)\s*([0-9]+(?:\.[0-9]+)?)",
                r"([0-9]+(?:\.[0-9]+)?)\s*(?:units|items)\s*(?:demanded|required)",
            ])

            # Ordering Cost: ordering cost, order cost, setup cost, setup expense,
            # cost per order, cost of placing an order, ordering/setup cost, order placement cost
            order_cost = find_pattern([
                r"(?:ordering\s*\/\s*setup\s*cost|ordering\s+and\s+setup\s*cost)\s*(?:is|=|:|of|\$)?\s*\$?([0-9]+(?:\.[0-9]+)?)",
                r"(?:cost\s+of\s+placing\s+an?\s+order|cost\s+per\s+order|order\s+placement\s+cost|order\s+charge)\s*(?:is|=|:|of|\$)?\s*\$?([0-9]+(?:\.[0-9]+)?)",
                r"(?:ordering\s*cost|order\s*cost|setup\s*cost|setup\s*expense|ordering\s*expense)\s*(?:is|=|:|of|\$)?\s*\$?([0-9]+(?:\.[0-9]+)?)",
                r"costs?\s*\$?([0-9]+(?:\.[0-9]+)?)\s*(?:to\s+(?:place\s+an?\s+order|order|setup)|per\s+order|\/order)",
                r"\$?([0-9]+(?:\.[0-9]+)?)\s*(?:per\s+order|\/order)",
                r"\$?([0-9]+(?:\.[0-9]+)?)\s*(?:is\s+the\s+)?(?:ordering\s*\/\s*setup\s*cost|ordering\s*cost|order\s*cost|setup\s*cost|setup\s*expense|order\s+placement\s+cost)",
            ])

            # Holding Cost: holding cost, carrying cost, holding expense, carrying expense,
            # inventory carrying cost, cost of holding, cost of carrying, cost to hold
            holding_cost = find_pattern([
                r"(?:fixed\s+)?(?:inventory\s+)?(?:holding|carrying)\s*(?:cost|expense|charge)\s*(?:is|=|:|of|\$)?\s*\$?([0-9]+(?:\.[0-9]+)?)",
                r"(?:cost\s+to\s+hold|cost\s+of\s+carrying|cost\s+of\s+holding)\s*(?:inventory\s*)?(?:is|=|:|of|\$)?\s*\$?([0-9]+(?:\.[0-9]+)?)",
                r"\$?([0-9]+(?:\.[0-9]+)?)\s*(?:per\s+unit\s*per\s*year|\/unit\/year)",
                r"\$?([0-9]+(?:\.[0-9]+)?)\s*(?:per\s+unit|\/unit)\s*(?:holding|carrying)\s*cost",
                r"\$?([0-9]+(?:\.[0-9]+)?)\s*(?:is\s+the\s+)?(?:holding|carrying)\s*(?:cost|expense)",
            ])

            # Percentage Holding Cost Rate (if holding cost given as % of unit cost)
            holding_rate = find_pattern([
                r"(?:holding|carrying)\s*(?:cost\s*)?(?:rate|percentage)\s*(?:is|=|:|of)?\s*([0-9]+(?:\.[0-9]+)?)\s*%",
                r"([0-9]+(?:\.[0-9]+)?)\s*%\s*(?:holding|carrying|inventory\s+carrying)\s*(?:cost|rate|expense)?",
                r"([0-9]+(?:\.[0-9]+)?)\s*%\s*of\s*(?:unit\s+)?(?:price|cost)",
            ])

            # Unit Purchase Price: unit purchase price, unit cost, item cost, wholesale cost
            unit_cost = find_pattern([
                r"(?:unit\s+purchase\s+price|unit\s+price|purchase\s+price|purchase\s+cost|wholesale\s+cost|wholesale\s+price|unit\s+cost|item\s+cost|material\s+cost)\s*(?:is|=|:|of|\$)?\s*\$?([0-9]+(?:\.[0-9]+)?)",
                r"bought\s*for\s*\$?([0-9]+(?:\.[0-9]+)?)",
                r"costs?\s*\$?([0-9]+(?:\.[0-9]+)?)\s*(?:per\s+unit|per\s+item|each)(?!\s*(?:per\s+year|holding|carrying))",
            ])

            # Lead Time (Days or Workdays)
            lead_time = find_pattern([
                r"(?:average\s+)?lead\s*time\s*(?:is|=|:|of)?\s*([0-9]+(?:\.[0-9]+)?)\s*(?:days?|workdays?)",
                r"([0-9]+(?:\.[0-9]+)?)\s*(?:days?|workdays?)\s*(?:average\s+)?lead\s*time",
            ])

            # If holding cost was expressed as a percentage rate and unit cost is known
            if holding_cost is None and holding_rate is not None and unit_cost is not None:
                rate_decimal = holding_rate / 100.0 if holding_rate > 1.0 else holding_rate
                holding_cost = rate_decimal * unit_cost

            if demand is not None:
                inputs["demand"] = demand
            if order_cost is not None:
                inputs["order_cost"] = order_cost
            if holding_cost is not None:
                inputs["holding_cost"] = holding_cost
            if unit_cost is not None:
                inputs["unit_cost"] = unit_cost
            if lead_time is not None:
                inputs["lead_time_days"] = lead_time

        # -------------------------------------------------------------
        # 2. Quantity Discount Extraction
        # -------------------------------------------------------------
        elif model_type == InventoryModelType.QUANTITY_DISCOUNT:
            demand = find_pattern([
                r"(?:annual|yearly)\s*(?:demand|sales|requirement|usage)\s*(?:is|=|:|of)?\s*([0-9]+(?:\.[0-9]+)?)",
                r"demand\s*(?:is|=|:|of)?\s*([0-9]+(?:\.[0-9]+)?)\s*(?:units|items|cases)?\s*(?:per\s+year|annually|a\s+year)?",
                r"([0-9]+(?:\.[0-9]+)?)\s*(?:units|items|cases)\s*(?:per\s+year|annually|a\s+year)",
                r"demand\s*(?:is|=|:|of)\s*([0-9]+(?:\.[0-9]+)?)",
                r"([0-9]+(?:\.[0-9]+)?)\s*(?:units|items)\s*(?:demanded|required)",
            ])
            order_cost = find_pattern([
                r"(?:ordering\s*\/\s*setup\s*cost|ordering\s+and\s+setup\s*cost)\s*(?:is|=|:|of|\$)?\s*\$?([0-9]+(?:\.[0-9]+)?)",
                r"(?:cost\s+of\s+placing\s+an?\s+order|cost\s+per\s+order|order\s+placement\s+cost)\s*(?:is|=|:|of|\$)?\s*\$?([0-9]+(?:\.[0-9]+)?)",
                r"(?:ordering\s*cost|order\s*cost|setup\s*cost|setup\s*expense)\s*(?:is|=|:|of|\$)?\s*\$?([0-9]+(?:\.[0-9]+)?)",
                r"costs?\s*\$?([0-9]+(?:\.[0-9]+)?)\s*(?:to\s+(?:place|order|setup)|per\s+order|\/order)",
                r"\$?([0-9]+(?:\.[0-9]+)?)\s*(?:per\s+order|\/order)",
                r"\$?([0-9]+(?:\.[0-9]+)?)\s*(?:is\s+the\s+)?(?:ordering\s*\/\s*setup\s*cost|ordering\s*cost|order\s*cost|setup\s*cost|setup\s*expense)",
            ])

            # Distinguish percentage holding cost from fixed holding cost
            holding_rate = find_pattern([
                r"(?:percentage\s+holding\s+cost|holding\s*(?:cost\s*)?rate|carrying\s*(?:cost\s*)?rate|carrying\s*rate)\s*(?:is|=|:|of)?\s*([0-9]+(?:\.[0-9]+)?)\s*%",
                r"([0-9]+(?:\.[0-9]+)?)\s*%\s*(?:holding|carrying|inventory\s+carrying)\s*(?:cost|rate|expense)?",
                r"([0-9]+(?:\.[0-9]+)?)\s*%\s*of\s*(?:unit\s+)?(?:price|cost)",
                r"holding\s*(?:cost\s*)?rate\s*(?:is|=|:|of)?\s*(0\.[0-9]+)",
            ])

            fixed_holding = find_pattern([
                r"fixed\s*(?:holding|carrying)\s*cost\s*(?:is|=|:|of|\$)?\s*\$?([0-9]+(?:\.[0-9]+)?)",
                r"(?:holding|carrying)\s*cost\s*(?:is|=|:|of|\$)?\s*\$?([0-9]+(?:\.[0-9]+)?)\s*(?:per\s+unit|\/unit)(?!\s*%)",
                r"\$?([0-9]+(?:\.[0-9]+)?)\s*(?:per\s+unit|\/unit)\s*(?:holding|carrying)\s*cost",
                r"holding\s*cost\s*(?:is|=|:|of|\$)?\s*\$?([0-9]+(?:\.[0-9]+)?)\s*(?:per\s+unit|\/unit)?(?!\s*%)",
            ])

            if demand is not None:
                inputs["demand"] = demand
            if order_cost is not None:
                inputs["order_cost"] = order_cost
            if holding_rate is not None:
                rate_val = holding_rate / 100.0 if holding_rate > 1.0 else holding_rate
                inputs["holding_cost_rate"] = rate_val
                inputs["holding_rate"] = rate_val
            elif fixed_holding is not None:
                inputs["fixed_holding_cost"] = fixed_holding
                inputs["holding_cost"] = fixed_holding

            # Extract quantity breaks & unit prices
            tiers: List[Dict[str, Any]] = []

            # Match patterns like: Tier 1: 0 to 499: $10 or 0-499 units @ $10 or between 0 and 499 units: $10
            tier_regex = re.finditer(
                r"(?:tier\s*[0-9]+[:\s]*)?(?:between\s+)?([0-9]+)\s*(?:to|-|and)\s*([0-9]+)\s*(?:units|items)?\s*(?:@|:|at|\$|for)\s*\$?([0-9]+(?:\.[0-9]+)?)",
                clean_text,
                re.IGNORECASE,
            )
            for m in tier_regex:
                tiers.append({
                    "min_qty": float(m.group(1)),
                    "max_qty": float(m.group(2)),
                    "price": float(m.group(3)),
                })

            # Match open-ended tiers like: 1000+ @ $9 or 1000 and above @ $9 or over 1000 units @ $9 or 1000 or more: $9
            open_tier_regex = re.finditer(
                r"(?:tier\s*[0-9]+[:\s]*)?(?:over|above|orders\s+of\s+at\s+least|at\s+least)?\s*([0-9]+)\s*(?:\+|and\s+(?:above|over|more)|or\s+more)\s*(?:units|items)?\s*(?:@|:|at|\$|for)\s*\$?([0-9]+(?:\.[0-9]+)?)",
                clean_text,
                re.IGNORECASE,
            )
            for m in open_tier_regex:
                tiers.append({
                    "min_qty": float(m.group(1)),
                    "max_qty": None,
                    "price": float(m.group(2)),
                })

            if tiers:
                inputs["discount_tiers"] = tiers

        # -------------------------------------------------------------
        # 3. Safety Stock & Reorder Point Extraction
        # -------------------------------------------------------------
        elif model_type == InventoryModelType.SAFETY_STOCK_ROP:
            # Distinguish:
            # - average demand
            # - demand standard deviation
            # - average lead time
            # - lead-time standard deviation
            # - service level

            # Step A: Extract Lead Time Standard Deviation FIRST (to prevent collision with lead time)
            std_lead_time = find_pattern([
                r"(?:lead[\s-]time|leadtime)\s+standard\s+deviation\s*(?:is|=|:|of)?\s*([0-9]+(?:\.[0-9]+)?)",
                r"standard\s+deviation\s+of\s+(?:the\s+)?(?:lead[\s-]time|leadtime)\s*(?:is|=|:|of)?\s*([0-9]+(?:\.[0-9]+)?)",
                r"(?:lead[\s-]time|leadtime)\s+std(?:\s*dev)?\s*(?:is|=|:|of)?\s*([0-9]+(?:\.[0-9]+)?)",
                r"std(?:\s*dev)?\s+of\s+(?:the\s+)?(?:lead[\s-]time|leadtime)\s*(?:is|=|:|of)?\s*([0-9]+(?:\.[0-9]+)?)",
                r"(?:lead[\s-]time|leadtime)\s+variability\s*(?:is|=|:|of)?\s*([0-9]+(?:\.[0-9]+)?)",
                r"sigma\s*(?:_?l|lead\s*time)\s*(?:is|=|:)?\s*([0-9]+(?:\.[0-9]+)?)",
            ])

            # Step B: Extract Average Lead Time
            lead_time = find_pattern([
                r"(?:average|mean|expected)\s+lead\s*time\s*(?:is|=|:|of)?\s*([0-9]+(?:\.[0-9]+)?)",
                r"(?:supplier|delivery)?\s*lead\s*time\s*(?:is|=|:|of)?\s*([0-9]+(?:\.[0-9]+)?)\s*(?:days?|weeks?|months?)",
                r"([0-9]+(?:\.[0-9]+)?)\s*(?:days?|weeks?|months?)\s*(?:average\s+|supplier\s+)?lead\s*time",
                r"lead\s*time\s*(?:is|=|:|of)\s*([0-9]+(?:\.[0-9]+)?)",
            ])

            # Step C: Extract Demand Standard Deviation (to prevent collision with average demand)
            std_demand = find_pattern([
                r"demand\s+standard\s+deviation\s*(?:is|=|:|of)?\s*([0-9]+(?:\.[0-9]+)?)",
                r"standard\s+deviation\s+of\s+(?:daily\s+|weekly\s+|monthly\s+)?demand\s*(?:is|=|:|of)?\s*([0-9]+(?:\.[0-9]+)?)",
                r"(?:daily|weekly|monthly)?\s*demand\s+std(?:\s*dev)?\s*(?:is|=|:|of)?\s*([0-9]+(?:\.[0-9]+)?)",
                r"std(?:\s*dev)?\s+of\s+(?:daily\s+|weekly\s+|monthly\s+)?demand\s*(?:is|=|:|of)?\s*([0-9]+(?:\.[0-9]+)?)",
                r"standard\s*deviation\s*(?:is|=|:|of)\s*([0-9]+(?:\.[0-9]+)?)",
                r"std(?:\s*dev)?\s*(?:is|=|:|of)\s*([0-9]+(?:\.[0-9]+)?)",
                r"sigma\s*(?:_?d|demand)\s*(?:is|=|:)?\s*([0-9]+(?:\.[0-9]+)?)",
            ])

            # Step D: Extract Average Demand
            avg_demand = find_pattern([
                r"(?:average|mean|expected)\s+demand\s*(?:is|=|:|of)?\s*([0-9]+(?:\.[0-9]+)?)",
                r"(?:daily|weekly|monthly)\s+demand\s*(?:is|=|:|of)?\s*([0-9]+(?:\.[0-9]+)?)",
                r"demand\s*(?:is|=|:|of)?\s*([0-9]+(?:\.[0-9]+)?)\s*(?:units|items|cases)?\s*(?:per\s+day|per\s+week|a\s+day|a\s+week)",
                r"demand\s*(?:rate\s*)?(?:is|=|:|of)\s*([0-9]+(?:\.[0-9]+)?)",
                r"([0-9]+(?:\.[0-9]+)?)\s*(?:units|items)?\s*(?:demanded\s+per\s+day|demanded\s+per\s+week)",
            ])

            # Step E: Extract Service Level (CSL)
            service_level = find_pattern([
                r"([0-9]+(?:\.[0-9]+)?)\s*%\s*(?:cycle\s+)?service\s*level",
                r"([0-9]+(?:\.[0-9]+)?)\s*%\s*(?:csl|in-?stock\s+probability)",
                r"(?:cycle\s+)?service\s*level\s*(?:is|=|:|of)?\s*([0-9]+(?:\.[0-9]+)?)\s*%",
                r"csl\s*(?:is|=|:|of)?\s*([0-9]+(?:\.[0-9]+)?)\s*%",
                r"target\s+service\s*level\s*(?:is|=|:|of)?\s*([0-9]+(?:\.[0-9]+)?)\s*%",
                r"(?:cycle\s+)?service\s*level\s*(?:is|=|:|of)?\s*(0\.[0-9]+)",
                r"csl\s*(?:is|=|:|of)?\s*(0\.[0-9]+)",
            ])

            holding_cost = find_pattern([
                r"(?:inventory\s+)?holding\s*cost\s*(?:is|=|:|of)?\s*\$?([0-9]+(?:\.[0-9]+)?)",
                r"carrying\s*cost\s*(?:is|=|:|of)?\s*\$?([0-9]+(?:\.[0-9]+)?)",
            ])

            order_qty = find_pattern([
                r"(?:order\s*quantity|lot\s*size|order\s*size)\s*(?:is|=|:|of)?\s*([0-9]+(?:\.[0-9]+)?)",
            ])

            if avg_demand is not None:
                inputs["avg_demand"] = avg_demand
                inputs["avg_demand_per_period"] = avg_demand
            if lead_time is not None:
                inputs["lead_time"] = lead_time
                inputs["lead_time_periods"] = lead_time
            if service_level is not None:
                inputs["service_level"] = service_level / 100.0 if service_level > 1.0 else service_level
            if std_demand is not None:
                inputs["std_demand"] = std_demand
                inputs["std_demand_per_period"] = std_demand
            if std_lead_time is not None:
                inputs["std_lead_time"] = std_lead_time
                inputs["std_lead_time_periods"] = std_lead_time
            if holding_cost is not None:
                inputs["unit_holding_cost"] = holding_cost
            if order_qty is not None:
                inputs["order_quantity"] = order_qty

        # -------------------------------------------------------------
        # 4. Fixed-Period Review Extraction
        # -------------------------------------------------------------
        elif model_type == InventoryModelType.PERIODIC_REVIEW:
            # Distinguish:
            # - demand
            # - review period
            # - lead time
            # - safety stock
            # - inventory position
            # - on-hand inventory
            # - on-order inventory
            # - backorders

            # Review Period (T)
            review_period = find_pattern([
                r"review\s*(?:period|interval)\s*(?:is|=|:|of)?\s*([0-9]+(?:\.[0-9]+)?)",
                r"reviewed\s*every\s*([0-9]+(?:\.[0-9]+)?)\s*(?:days?|weeks?|months?)",
                r"every\s*([0-9]+(?:\.[0-9]+)?)\s*(?:days?|weeks?|months?)",
                r"order\s*(?:cycle|interval)\s*(?:is|=|:|of)?\s*([0-9]+(?:\.[0-9]+)?)",
                r"periodic\s+review\s+(?:of|every)?\s*([0-9]+(?:\.[0-9]+)?)\s*(?:days?|weeks?|months?)",
            ])

            # Supplier Lead Time (L)
            lead_time = find_pattern([
                r"(?:supplier|delivery)?\s*lead\s*time\s*(?:is|=|:|of)?\s*([0-9]+(?:\.[0-9]+)?)\s*(?:days?|weeks?|months?)",
                r"([0-9]+(?:\.[0-9]+)?)\s*(?:days?|weeks?|months?)\s*lead\s*time",
                r"lead\s*time\s*(?:is|=|:|of)\s*([0-9]+(?:\.[0-9]+)?)",
            ])

            # Demand (d)
            avg_demand = find_pattern([
                r"(?:weekly|daily|average|mean|periodic)\s*demand\s*(?:is|=|:|of)?\s*([0-9]+(?:\.[0-9]+)?)",
                r"demand\s*(?:is|=|:|of)\s*([0-9]+(?:\.[0-9]+)?)\s*(?:units|items)?",
            ])

            # Demand Standard Deviation
            std_demand = find_pattern([
                r"demand\s+standard\s+deviation\s*(?:is|are|=|:|of)?\s*([0-9]+(?:\.[0-9]+)?)",
                r"standard\s+deviation\s+of\s+(?:daily\s+|weekly\s+)?demand\s*(?:is|are|=|:|of)?\s*([0-9]+(?:\.[0-9]+)?)",
                r"(?:with\s+)?standard\s*deviation\s*(?:is|are|=|:|of)?\s*([0-9]+(?:\.[0-9]+)?)",
                r"std(?:\s*dev)?\s*(?:of\s+demand)?\s*(?:is|are|=|:|of)?\s*([0-9]+(?:\.[0-9]+)?)",
                r"sigma\s*(?:is|are|=|:)?\s*([0-9]+(?:\.[0-9]+)?)",
            ])

            # Service Level (CSL)
            service_level = find_pattern([
                r"([0-9]+(?:\.[0-9]+)?)\s*%\s*(?:cycle\s+)?service\s*level",
                r"([0-9]+(?:\.[0-9]+)?)\s*%\s*csl",
                r"(?:cycle\s+)?service\s*level\s*(?:is|are|=|:|of)?\s*([0-9]+(?:\.[0-9]+)?)\s*%",
                r"service\s*level\s*(?:is|are|=|:|of)?\s*(0\.[0-9]+)",
            ])

            # Direct Safety Stock (if specified)
            safety_stock = find_pattern([
                r"safety\s*stock\s*(?:is|are|=|:|of)?\s*([0-9]+(?:\.[0-9]+)?)",
            ])

            # Direct Inventory Position (if specified)
            inventory_pos = find_pattern([
                r"inventory\s*position\s*(?:is|are|=|:|of)?\s*([0-9]+(?:\.[0-9]+)?)",
            ])

            # On-Hand Inventory
            on_hand = find_pattern([
                r"(?:current\s+)?on-?hand\s*(?:inventory|stock)?\s*(?:is|are|=|:|of)?\s*([0-9]+(?:\.[0-9]+)?)",
                r"([0-9]+(?:\.[0-9]+)?)\s*(?:units)?\s*on-?hand",
                r"stock\s+on\s+hand\s*(?:is|are|=|:|of)?\s*([0-9]+(?:\.[0-9]+)?)",
            ])

            # On-Order / In-Transit Inventory
            on_order = find_pattern([
                r"(?:current\s+)?on-?order\s*(?:inventory|stock)?\s*(?:is|are|=|:|of)?\s*([0-9]+(?:\.[0-9]+)?)",
                r"(?:current\s+)?in-?transit\s*(?:inventory|stock)?\s*(?:is|are|=|:|of)?\s*([0-9]+(?:\.[0-9]+)?)",
                r"([0-9]+(?:\.[0-9]+)?)\s*(?:units)?\s*(?:on-?order|in-?transit)",
            ])

            # Backorders
            backorders = find_pattern([
                r"backorders?\s*(?:is|are|=|:|of)?\s*([0-9]+(?:\.[0-9]+)?)",
                r"backlog\s*(?:is|are|=|:|of)?\s*([0-9]+(?:\.[0-9]+)?)",
                r"([0-9]+(?:\.[0-9]+)?)\s*(?:units)?\s*backordered",
            ])

            if avg_demand is not None:
                inputs["avg_demand"] = avg_demand
                inputs["avg_demand_per_period"] = avg_demand
            if review_period is not None:
                inputs["review_period"] = review_period
                inputs["review_period_T"] = review_period
            if lead_time is not None:
                inputs["lead_time"] = lead_time
                inputs["lead_time_L"] = lead_time
            if service_level is not None:
                inputs["service_level"] = service_level / 100.0 if service_level > 1.0 else service_level
            if std_demand is not None:
                inputs["std_demand"] = std_demand
                inputs["std_demand_per_period"] = std_demand
            if safety_stock is not None:
                inputs["safety_stock"] = safety_stock
            if inventory_pos is not None:
                inputs["inventory_position"] = inventory_pos
            if on_hand is not None:
                inputs["on_hand"] = on_hand
            if on_order is not None:
                inputs["on_order"] = on_order
            if backorders is not None:
                inputs["backorders"] = backorders

        # -------------------------------------------------------------
        # 5. Newsvendor Extraction
        # -------------------------------------------------------------
        elif model_type == InventoryModelType.NEWSVENDOR:
            # Distinguish:
            # - selling price
            # - purchase cost
            # - salvage value
            # - underage / shortage cost
            # - overage / excess cost
            # - demand distribution
            # - mean demand
            # - demand standard deviation

            # Underage / Shortage Cost (Cu)
            underage = find_pattern([
                r"(?:underage\s*cost|cost\s+of\s+underage|shortage\s*cost|cost\s+of\s+understocking|loss\s+per\s+unmet\s+unit)\s*(?:is|are|=|:|of|\$)?\s*\$?([0-9]+(?:\.[0-9]+)?)",
                r"\$?([0-9]+(?:\.[0-9]+)?)\s*(?:is\s+the\s+)?(?:underage\s*cost|cost\s+of\s+underage|shortage\s*cost)",
            ])

            # Overage / Excess Cost (Co)
            overage = find_pattern([
                r"(?:overage\s*cost|cost\s+of\s+overage|excess\s*cost|cost\s+of\s+overstocking|loss\s+per\s+unsold\s+unit)\s*(?:is|are|=|:|of|\$)?\s*\$?([0-9]+(?:\.[0-9]+)?)",
                r"\$?([0-9]+(?:\.[0-9]+)?)\s*(?:is\s+the\s+)?(?:overage\s*cost|cost\s+of\s+overage|excess\s*cost)",
            ])

            # Shortage / Goodwill Penalty
            penalty = find_pattern([
                r"(?:shortage|goodwill|penalty)\s*(?:penalty|loss|cost)?\s*(?:is|are|=|:|of|\$)?\s*\$?([0-9]+(?:\.[0-9]+)?)",
            ])

            # Selling / Retail Price
            price = find_pattern([
                r"(?:selling|retail|sale)\s*price\s*(?:is|are|=|:|of|\$)?\s*\$?([0-9]+(?:\.[0-9]+)?)",
                r"sells?\s*(?:.*?)\s*for\s*\$?([0-9]+(?:\.[0-9]+)?)",
                r"(?<!wholesale\s)(?<!purchase\s)price\s*(?:is|are|=|:|of|\$)?\s*\$?([0-9]+(?:\.[0-9]+)?)",
                r"\$?([0-9]+(?:\.[0-9]+)?)\s*(?:retail\s+price|selling\s+price|retail|sale\s+price)",
            ])

            # Purchase / Wholesale Cost
            cost = find_pattern([
                r"(?:purchase\s+cost|wholesale\s+cost|wholesale\s+price|procurement\s+cost|unit\s+cost|acquisition\s+cost)\s*(?:is|are|=|:|of|\$)?\s*\$?([0-9]+(?:\.[0-9]+)?)",
                r"costs?\s*\$?([0-9]+(?:\.[0-9]+)?)\s*(?:to\s+(?:buy|procure|acquire|make)|per\s+unit|each)",
                r"(?<!underage\s)(?<!overage\s)(?<!excess\s)(?<!shortage\s)(?<!holding\s)(?<!setup\s)(?<!ordering\s)cost\s*(?:is|are|=|:|of|\$)?\s*\$?([0-9]+(?:\.[0-9]+)?)",
                r"\$?([0-9]+(?:\.[0-9]+)?)\s*(?:purchase\s+cost|wholesale\s+cost|unit\s+cost)",
            ])

            # Salvage Value
            salvage = find_pattern([
                r"(?:salvage\s*value|salvage\s*price|clearance\s*price|clearance\s*value|liquidation\s*value|scrap\s*value)\s*(?:is|are|=|:|of|\$)?\s*\$?([0-9]+(?:\.[0-9]+)?)",
                r"discounted\s*to\s*\$?([0-9]+(?:\.[0-9]+)?)",
                r"salvaged\s*at\s*\$?([0-9]+(?:\.[0-9]+)?)",
                r"salvage\s*(?:is|are|=|:|of|\$)?\s*\$?([0-9]+(?:\.[0-9]+)?)",
            ])

            # Mean Demand
            mean_demand = find_pattern([
                r"(?:mean|average|expected)\s*demand\s*(?:is|are|=|:|of)?\s*([0-9]+(?:\.[0-9]+)?)",
                r"demand\s*(?:is\s*normal\s*with\s*mean|=|:|of)?\s*([0-9]+(?:\.[0-9]+)?)",
                r"mean\s*(?:is|are|=|:|of)\s*([0-9]+(?:\.[0-9]+)?)",
            ])

            # Demand Standard Deviation
            std_demand = find_pattern([
                r"demand\s+standard\s+deviation\s*(?:is|are|=|:|of)?\s*([0-9]+(?:\.[0-9]+)?)",
                r"standard\s+deviation\s+of\s+demand\s*(?:is|are|=|:|of)?\s*([0-9]+(?:\.[0-9]+)?)",
                r"(?:with\s+)?standard\s*deviation\s*(?:is|are|=|:|of)?\s*([0-9]+(?:\.[0-9]+)?)",
                r"std(?:\s*dev)?\s*(?:of\s+demand)?\s*(?:is|are|=|:|of)?\s*([0-9]+(?:\.[0-9]+)?)",
                r"sigma\s*(?:is|are|=|:)?\s*([0-9]+(?:\.[0-9]+)?)",
            ])

            if price is not None:
                inputs["selling_price"] = price
            # Only set purchase_cost from generic cost pattern if underage cost was not directly provided
            if cost is not None and underage is None:
                inputs["purchase_cost"] = cost
            elif cost is not None and underage is not None and price is not None:
                inputs["purchase_cost"] = cost
            if salvage is not None:
                inputs["salvage_value"] = salvage
            if underage is not None:
                inputs["underage_cost"] = underage
            if overage is not None:
                inputs["overage_cost"] = overage
            if penalty is not None:
                inputs["shortage_penalty"] = penalty
            if mean_demand is not None:
                inputs["mean_demand"] = mean_demand
            if std_demand is not None:
                inputs["std_demand"] = std_demand

        return inputs

    # =================================================================
    # STEP 3: Select Python Tool
    # =================================================================
    def select_tool(self, model_type: InventoryModelType) -> Tuple[str, Callable]:
        """Map the identified model to its corresponding calculation tool."""
        if model_type not in self.tool_mapping:
            raise ValueError(f"No registered calculation tool for model: {model_type}")
        return self.tool_mapping[model_type]

    # =================================================================
    # STEP 4: Calculate
    # =================================================================
    def calculate(self, tool_callable: Callable, inputs: Dict[str, Any]) -> Any:
        """Execute the mathematical calculation tool with the extracted inputs."""
        try:
            return tool_callable(**inputs)
        except TypeError as e:
            raise ValueError(f"Missing required parameters for calculation: {e}")
        except Exception as e:
            raise RuntimeError(f"Calculation failed: {e}")

    # =================================================================
    # STEP 5: Verify
    # =================================================================
    def verify(
        self,
        model_type: InventoryModelType,
        inputs: Dict[str, Any],
        result: Any,
    ) -> VerificationResult:
        """Perform automated mathematical, domain, and sanity checks on results."""
        checks: List[VerificationCheck] = []
        notes: List[str] = []

        if model_type == InventoryModelType.EOQ:
            eoq_res: EOQResult = result
            # Check 1: Non-negative order quantity
            checks.append(
                VerificationCheck(
                    name="Positive Order Quantity Check",
                    passed=eoq_res.optimal_order_quantity > 0,
                    details=f"Q* = {eoq_res.optimal_order_quantity} units",
                )
            )
            # Check 2: Optimality condition: Annual Ordering Cost ≈ Annual Holding Cost
            cost_diff = abs(eoq_res.annual_ordering_cost - eoq_res.annual_holding_cost)
            is_optimal_balance = cost_diff <= 1.0 or (cost_diff / max(1.0, eoq_res.annual_holding_cost)) < 0.01
            checks.append(
                VerificationCheck(
                    name="Cost Equivalence at EOQ",
                    passed=is_optimal_balance,
                    details=f"Ordering Cost (${eoq_res.annual_ordering_cost}) vs Holding Cost (${eoq_res.annual_holding_cost}), delta=${round(cost_diff, 2)}",
                )
            )

        elif model_type == InventoryModelType.QUANTITY_DISCOUNT:
            qd_res: QuantityDiscountResult = result
            # Check 1: Optimal tier cost is <= all other evaluated tiers
            all_costs = [t.total_annual_cost for t in qd_res.tier_evaluations]
            is_global_min = math.isclose(qd_res.minimum_total_annual_cost, min(all_costs), abs_tol=0.01)
            checks.append(
                VerificationCheck(
                    name="Global Cost Minimization Check",
                    passed=is_global_min,
                    details=f"Selected Tier {qd_res.optimal_tier_index} total cost ${qd_res.minimum_total_annual_cost:,.2f} is the minimum across all {len(qd_res.tier_evaluations)} tiers",
                )
            )
            # Check 2: Order quantity satisfies tier boundary
            opt_eval = qd_res.tier_evaluations[qd_res.optimal_tier_index - 1]
            meets_min = opt_eval.realized_order_quantity >= opt_eval.min_quantity
            checks.append(
                VerificationCheck(
                    name="Tier Minimum Quantity Compliance",
                    passed=meets_min,
                    details=f"Realized Q = {opt_eval.realized_order_quantity} >= Min tier threshold {opt_eval.min_quantity}",
                )
            )

        elif model_type == InventoryModelType.SAFETY_STOCK_ROP:
            ss_res: SafetyStockROPResult = result
            # Check 1: ROP >= Safety Stock
            checks.append(
                VerificationCheck(
                    name="Reorder Point Dominance Check",
                    passed=ss_res.reorder_point >= ss_res.safety_stock,
                    details=f"ROP ({ss_res.reorder_point}) >= Safety Stock ({ss_res.safety_stock})",
                )
            )
            # Check 2: Z-score is positive for service levels > 50%
            is_z_valid = True
            if ss_res.z_score is not None and ss_res.target_service_level is not None:
                is_z_valid = ss_res.z_score > 0 if ss_res.target_service_level > 0.5 else True
            z_details = f"Target CSL = {ss_res.target_service_level*100:.1f}%, Z = {ss_res.z_score}" if ss_res.target_service_level is not None else "Deterministic scenario (no Z-score needed)"
            checks.append(
                VerificationCheck(
                    name="Service Level Z-Score Check",
                    passed=is_z_valid,
                    details=z_details,
                )
            )

        elif model_type == InventoryModelType.PERIODIC_REVIEW:
            pr_res: PeriodicReviewResult = result
            # Check 1: Base Stock S >= Safety Stock + Expected Demand
            expected_total = pr_res.expected_demand_during_protection_interval + pr_res.safety_stock
            checks.append(
                VerificationCheck(
                    name="Base Stock Conservation Check",
                    passed=math.isclose(pr_res.target_inventory_level_S, expected_total, abs_tol=0.1),
                    details=f"S ({pr_res.target_inventory_level_S}) = Expected Protection Demand ({pr_res.expected_demand_during_protection_interval}) + SS ({pr_res.safety_stock})",
                )
            )
            # Check 2: Recommended Q >= 0
            checks.append(
                VerificationCheck(
                    name="Non-Negative Order Quantity",
                    passed=pr_res.recommended_order_quantity_Q >= 0,
                    details=f"Order quantity Q = {pr_res.recommended_order_quantity_Q} units",
                )
            )

        elif model_type == InventoryModelType.NEWSVENDOR:
            nv_res: NewsvendorResult = result
            # Check 1: Critical Ratio is within (0, 1)
            cr_valid = 0.0 < nv_res.critical_ratio < 1.0
            checks.append(
                VerificationCheck(
                    name="Critical Ratio Bounds Check",
                    passed=cr_valid,
                    details=f"CR = {nv_res.critical_ratio:.4f} in (0, 1)",
                )
            )
            # Check 2: Expected Profit <= EVPI Profit
            if nv_res.expected_profit_perfect_information is not None:
                evpi_valid = nv_res.expected_profit <= (nv_res.expected_profit_perfect_information + 1e-2)
                checks.append(
                    VerificationCheck(
                        name="Profit Bound Check (EVPI)",
                        passed=evpi_valid,
                        details=f"Expected Profit (${nv_res.expected_profit:,.2f}) <= Perfect Info Profit (${nv_res.expected_profit_perfect_information:,.2f})",
                    )
                )

        all_passed = all(c.passed for c in checks)
        return VerificationResult(is_valid=all_passed, checks=checks, notes=notes)

    # =================================================================
    # STEP 6: Explain
    # =================================================================
    def explain(
        self,
        model_type: InventoryModelType,
        inputs: Dict[str, Any],
        result: Any,
        verification: VerificationResult,
    ) -> str:
        """Generate an executive-grade supply chain decision explanation."""
        sections: List[str] = []

        # Model Header
        sections.append(f"## 📦 Supply Chain Decision Analysis: {model_type.value}\n")

        # 1. Executive Decision Summary
        if model_type == InventoryModelType.EOQ:
            eoq: EOQResult = result
            sections.append(
                f"### 🎯 Optimal Decision Summary\n"
                f"- **Recommended Order Quantity (EOQ):** `{eoq.optimal_order_quantity:,.2f} units`\n"
                f"- **Optimal Order Frequency:** `{eoq.orders_per_year:.2f} orders/year` (every `{eoq.cycle_time_days:.1f} days`)\n"
                f"- **Total Annual Inventory Cost:** `${eoq.total_inventory_cost:,.2f}`\n"
            )
            if eoq.reorder_point_deterministic is not None:
                sections.append(f"- **Deterministic Reorder Point:** `{eoq.reorder_point_deterministic:,.2f} units` (based on {eoq.lead_time_days} days lead time)\n")

            # Metrics Table
            headers = ["Metric", "Value", "Operational Significance"]
            rows = [
                ["Optimal Order Quantity (Q*)", f"{eoq.optimal_order_quantity:,.2f} units", "Minimizes combined ordering & holding costs"],
                ["Annual Demand (D)", f"{eoq.annual_demand:,.0f} units", "Baseline operational requirement"],
                ["Annual Ordering Cost", f"${eoq.annual_ordering_cost:,.2f}", f"Based on {eoq.orders_per_year:.1f} annual orders @ ${eoq.order_cost}/order"],
                ["Annual Holding Cost", f"${eoq.annual_holding_cost:,.2f}", f"Holding {eoq.average_cycle_inventory:,.1f} avg units @ ${eoq.holding_cost_per_unit}/unit/yr"],
                ["Total Annual Inventory Cost", f"${eoq.total_inventory_cost:,.2f}", "Sum of setup and carrying costs"],
            ]
            if eoq.total_annual_cost and eoq.annual_purchasing_cost:
                rows.append(["Annual Purchasing Cost", f"${eoq.annual_purchasing_cost:,.2f}", "Total material cost"])
                rows.append(["Total Annual Supply Chain Cost", f"${eoq.total_annual_cost:,.2f}", "Total cost including procurement"])
            sections.append(format_table(headers, rows) + "\n")

            # Strategic Insights
            sections.append(
                "### 💡 Supply Chain Trade-Offs & Insights\n"
                "- **Cost Balance:** Notice that annual ordering costs and annual holding costs are virtually identical. This mathematical parity proves that the lot size is optimal.\n"
                "- **Sensitivity:** The EOQ curve is relatively flat around the minimum. A ±10% variation in order quantity increases total inventory costs by less than 1%.\n"
            )

        elif model_type == InventoryModelType.QUANTITY_DISCOUNT:
            qd: QuantityDiscountResult = result
            sections.append(
                f"### 🎯 Optimal Decision Summary\n"
                f"- **Recommended Order Quantity:** `{qd.optimal_order_quantity:,.2f} units`\n"
                f"- **Optimal Price Tier:** `Tier {qd.optimal_tier_index}` at `${qd.optimal_unit_price:,.2f}/unit`\n"
                f"- **Minimum Total Annual Cost:** `${qd.minimum_total_annual_cost:,.2f}`\n"
                f"- **Annual Savings vs Baseline Tier:** `${qd.annual_cost_savings:,.2f}`\n"
            )

            # Tier Evaluation Table
            headers = ["Tier", "Bracket", "Unit Price", "Raw EOQ", "Feasible Order Qty", "Annual Purchase", "Annual Order + Hold", "Total Annual Cost", "Status"]
            rows = []
            for t in qd.tier_evaluations:
                max_str = f"{t.max_quantity:,.0f}" if t.max_quantity is not None else "Above"
                bracket = f"{t.min_quantity:,.0f} - {max_str}"
                status = "⭐ Optimal" if t.tier_index == qd.optimal_tier_index else "Suboptimal"
                order_hold = t.annual_ordering_cost + t.annual_holding_cost
                rows.append([
                    f"Tier {t.tier_index}",
                    bracket,
                    f"${t.unit_price:,.2f}",
                    f"{t.raw_eoq:,.1f}",
                    f"{t.realized_order_quantity:,.1f}",
                    f"${t.annual_purchase_cost:,.2f}",
                    f"${order_hold:,.2f}",
                    f"${t.total_annual_cost:,.2f}",
                    status,
                ])
            sections.append(format_table(headers, rows) + "\n")

            sections.append(
                "### 💡 Supply Chain Trade-Offs & Insights\n"
                "- **Bulk Discount Justification:** Ordering at the price break incurs higher average holding costs, but the purchase price discount across annual demand far outweighs carrying costs.\n"
                "- **Working Capital & Warehouse Impact:** Ensure warehouse capacity and working capital limits accommodate the larger batch size.\n"
            )

        elif model_type == InventoryModelType.SAFETY_STOCK_ROP:
            ss: SafetyStockROPResult = result
            sections.append(
                f"### 🎯 Optimal Decision Summary\n"
                f"- **Recommended Reorder Point (ROP):** `{ss.reorder_point:,.2f} units`\n"
                f"- **Required Safety Stock:** `{ss.safety_stock:,.2f} units`\n"
                f"- **Target Cycle Service Level (CSL):** `{ss.target_service_level*100:.1f}%` (Z = `{ss.z_score:.4f}`)\n"
                f"- **Expected Demand During Lead Time:** `{ss.expected_demand_during_lead_time:,.2f} units`\n"
            )

            headers = ["Inventory Parameter", "Value", "Operational Description"]
            rows = [
                ["Reorder Point (ROP)", f"{ss.reorder_point:,.2f} units", "Place a replenishment order when inventory drops to this level"],
                ["Safety Stock (SS)", f"{ss.safety_stock:,.2f} units", "Buffer stock absorbing demand and lead time variations"],
                ["Expected Lead Time Demand (DDLT)", f"{ss.expected_demand_during_lead_time:,.2f} units", "Average consumption during replenishment window"],
                ["Combined Lead Time Std Dev", f"{ss.combined_lead_time_std_dev:,.2f} units", f"Variability scenario: {ss.variability_scenario.replace('_', ' ').title()}"],
                ["Target Service Level (CSL)", f"{ss.target_service_level*100:.1f}%", f"Z-score = {ss.z_score}"],
            ]
            if ss.annual_safety_stock_holding_cost is not None:
                rows.append(["Annual SS Carrying Cost", f"${ss.annual_safety_stock_holding_cost:,.2f}", "Cost of holding safety buffer"])
            if ss.expected_units_short_per_cycle is not None:
                rows.append(["Expected Shortage / Cycle", f"{ss.expected_units_short_per_cycle:.3f} units", "Anticipated stockout units per replenishment"])
            sections.append(format_table(headers, rows) + "\n")

            sections.append(
                "### 💡 Supply Chain Trade-Offs & Insights\n"
                f"- **Stockout Protection:** Under this policy, only `{(1.0 - ss.target_service_level)*100:.1f}%` of replenishment cycles will experience a stockout.\n"
                "- **Lead Time Impact:** Because safety stock scales with the square root of lead time, negotiating shorter lead times with suppliers will exponentially reduce required buffer inventory.\n"
            )

        elif model_type == InventoryModelType.PERIODIC_REVIEW:
            pr: PeriodicReviewResult = result
            sections.append(
                f"### 🎯 Optimal Decision Summary\n"
                f"- **Target Inventory Level (Order-Up-To Level S):** `{pr.target_inventory_level_S:,.2f} units`\n"
                f"- **Recommended Order Quantity (Q):** `{pr.recommended_order_quantity_Q:,.2f} units`\n"
                f"- **Required Safety Stock:** `{pr.safety_stock:,.2f} units`\n"
                f"- **Protection Interval (T + L):** `{pr.protection_interval_T_plus_L:.1f} periods` (Review {pr.review_period_T} + Lead Time {pr.lead_time_L})\n"
            )

            headers = ["Periodic Parameter", "Value", "Operational Function"]
            rows = [
                ["Target Base Stock (S)", f"{pr.target_inventory_level_S:,.2f} units", "Target order-up-to ceiling level"],
                ["Recommended Replenishment (Q)", f"{pr.recommended_order_quantity_Q:,.2f} units", f"Order to place now (S - Inventory Position)"],
                ["Current Inventory Position", f"{pr.current_inventory_position:,.2f} units", f"On-Hand ({pr.current_on_hand}) + On-Order ({pr.current_on_order}) - Backorders ({pr.current_backorders})"],
                ["Protection Interval", f"{pr.protection_interval_T_plus_L:.1f} periods", "Duration vulnerable to stockouts before next delivery arrives"],
                ["Safety Stock", f"{pr.safety_stock:,.2f} units", f"Absorbs variability over {pr.protection_interval_T_plus_L:.1f} periods"],
                ["Average Cycle Stock", f"{pr.average_cycle_stock:,.2f} units", "Expected cycle stock held between reviews"],
                ["Average Total Inventory", f"{pr.expected_average_inventory:,.2f} units", "Expected average on-hand stock"],
            ]
            sections.append(format_table(headers, rows) + "\n")

            sections.append(
                "### 💡 Supply Chain Trade-Offs & Insights\n"
                "- **Vulnerability Window:** Unlike continuous review where risk is limited to lead time L, periodic review systems are exposed over `T + L`. Shortening the review period T directly lowers required safety buffer.\n"
                "- **Ordering Policy:** If the current inventory position exceeds the target S, no replenishment order is placed (Q = 0).\n"
            )

        elif model_type == InventoryModelType.NEWSVENDOR:
            nv: NewsvendorResult = result
            sections.append(
                f"### 🎯 Optimal Decision Summary\n"
                f"- **Recommended Order Quantity (Q*):** `{nv.optimal_order_quantity:,.2f} units`\n"
                f"- **Critical Ratio (Optimal Service Level):** `{nv.critical_ratio*100:.2f}%`\n"
                f"- **Expected Profit:** `${nv.expected_profit:,.2f}`\n"
                f"- **Expected Leftover Inventory:** `{nv.expected_leftover_units:,.2f} units`\n"
                f"- **Expected Lost Sales:** `{nv.expected_lost_sales:,.2f} units`\n"
            )

            headers = ["Economic Factor", "Value", "Explanation"]
            rows = [
                ["Selling Price (P)", f"${nv.selling_price:,.2f}", "Retail / realization price per unit"],
                ["Purchase Cost (C)", f"${nv.purchase_cost:,.2f}", "Procurement / production cost per unit"],
                ["Salvage Value (V)", f"${nv.salvage_value:,.2f}", "Liquidation / salvage recovery per leftover unit"],
                ["Cost of Underage (Cu)", f"${nv.cost_of_underage_Cu:,.2f}", "Profit foregone per unmet unit of demand"],
                ["Cost of Overage (Co)", f"${nv.cost_of_overage_Co:,.2f}", "Net loss per unsold unit"],
                ["Critical Ratio (CR)", f"{nv.critical_ratio:.4f}", "Target probability of meeting all demand"],
                ["Optimal Order Quantity (Q*)", f"{nv.optimal_order_quantity:,.2f} units", "Maximizes expected economic profit"],
                ["Expected Profit", f"${nv.expected_profit:,.2f}", "Expected profit factoring all uncertainty"],
            ]
            if nv.expected_value_of_perfect_information is not None:
                rows.append(["Expected Value of Perfect Info (EVPI)", f"${nv.expected_value_of_perfect_information:,.2f}", "Maximum price worth paying for perfect demand forecast"])
            sections.append(format_table(headers, rows) + "\n")

            sections.append(
                "### 💡 Supply Chain Trade-Offs & Insights\n"
                f"- **Critical Ratio Mechanics:** Since Cu (${nv.cost_of_underage_Cu:,.2f}) vs Co (${nv.cost_of_overage_Co:,.2f}) yields CR = {nv.critical_ratio:.2%}, "
                + ("ordering above the mean demand is optimal because understocking is more costly than overstocking." if nv.critical_ratio > 0.5 else "ordering below the mean is optimal because overstocking penalty exceeds understocking margin.")
                + "\n- **Salvage Value Sensitivity:** Increasing salvage value (e.g. through secondary markets or buyback contracts) reduces Co and enables ordering more units with less downside risk.\n"
            )

        # Student Verification Section (Deterministic 8-Step Walkthrough from Python Tool)
        if hasattr(result, "format_student_verification_markdown"):
            sections.append(result.format_student_verification_markdown() + "\n")

        # Verification Section
        sections.append("### ✅ Automated Model Verification\n")
        v_headers = ["Verification Test", "Result", "Diagnostic Details"]
        v_rows = [
            [c.name, "PASSED ✅" if c.passed else "FAILED ❌", c.details]
            for c in verification.checks
        ]
        sections.append(format_table(v_headers, v_rows) + "\n")

        return "\n".join(sections)

    # =================================================================
    # Missing Inputs Verification & Explanation
    # =================================================================
    def identify_missing_inputs(self, model_type: InventoryModelType, inputs: Dict[str, Any]) -> List[str]:
        """Examine inputs extracted from the full question and return any genuinely missing parameters."""
        missing: List[str] = []

        if model_type == InventoryModelType.EOQ:
            if inputs.get("demand") is None:
                missing.append("demand")
            if inputs.get("order_cost") is None:
                missing.append("order_cost")
            if inputs.get("holding_cost") is None and inputs.get("holding_rate") is None:
                missing.append("holding_cost")

        elif model_type == InventoryModelType.QUANTITY_DISCOUNT:
            if inputs.get("demand") is None:
                missing.append("demand")
            if inputs.get("order_cost") is None:
                missing.append("order_cost")
            if not inputs.get("discount_tiers"):
                missing.append("discount_tiers")
            if (
                inputs.get("holding_cost_rate") is None
                and inputs.get("holding_rate") is None
                and inputs.get("fixed_holding_cost") is None
                and inputs.get("holding_cost") is None
            ):
                missing.append("holding_cost_or_rate")

        elif model_type == InventoryModelType.SAFETY_STOCK_ROP:
            if inputs.get("avg_demand_per_period") is None and inputs.get("avg_demand") is None:
                missing.append("avg_demand")
            if inputs.get("lead_time_periods") is None and inputs.get("lead_time") is None:
                missing.append("lead_time")
            has_var = (
                (inputs.get("std_demand_per_period") or 0) > 0
                or (inputs.get("std_demand") or 0) > 0
                or (inputs.get("std_lead_time_periods") or 0) > 0
                or (inputs.get("std_lead_time") or 0) > 0
            )
            if has_var and inputs.get("service_level") is None:
                missing.append("service_level")

        elif model_type == InventoryModelType.PERIODIC_REVIEW:
            if inputs.get("avg_demand_per_period") is None and inputs.get("avg_demand") is None:
                missing.append("avg_demand")
            if inputs.get("review_period_T") is None and inputs.get("review_period") is None:
                missing.append("review_period")
            if inputs.get("lead_time_L") is None and inputs.get("lead_time") is None:
                missing.append("lead_time")
            has_var = (inputs.get("std_demand_per_period") or 0) > 0 or (inputs.get("std_demand") or 0) > 0
            if has_var and inputs.get("service_level") is None:
                missing.append("service_level")

        elif model_type == InventoryModelType.NEWSVENDOR:
            has_direct_margin = (
                (inputs.get("underage_cost") is not None or inputs.get("shortage_cost") is not None)
                and (inputs.get("overage_cost") is not None or inputs.get("excess_cost") is not None)
            )
            has_prices = (
                inputs.get("selling_price") is not None and inputs.get("purchase_cost") is not None
            )
            if not has_direct_margin and not has_prices:
                if inputs.get("selling_price") is None and inputs.get("underage_cost") is None:
                    missing.append("selling_price_or_underage_cost")
                if inputs.get("purchase_cost") is None and inputs.get("overage_cost") is None:
                    missing.append("purchase_cost_or_overage_cost")
            has_continuous = (
                inputs.get("mean_demand") is not None and inputs.get("std_demand") is not None
            )
            has_discrete = (
                inputs.get("discrete_demand") is not None and inputs.get("discrete_probs") is not None
            )
            if not has_continuous and not has_discrete:
                if inputs.get("mean_demand") is None and inputs.get("discrete_demand") is None:
                    missing.append("mean_demand_or_discrete_distribution")
                if inputs.get("std_demand") is None and inputs.get("discrete_probs") is None:
                    missing.append("std_demand")

        return missing

    def explain_missing_inputs(
        self,
        model_type: InventoryModelType,
        inputs: Dict[str, Any],
        missing: List[str],
    ) -> str:
        """Generate clear supply chain explanation of what required inputs are missing and why."""
        param_descriptions = {
            "demand": "Annual Demand (D): The expected units required or sold per year.",
            "order_cost": "Ordering / Setup Cost (S): Fixed administrative and handling cost incurred per order (synonyms: setup expense, cost per order, ordering/setup cost).",
            "holding_cost": "Holding Cost (H): Annual cost to carry one unit in inventory ($/unit/year).",
            "holding_cost_or_rate": "Holding Cost: Either a fixed carrying cost ($/unit/yr) or a carrying rate (% of price).",
            "discount_tiers": "Quantity Breaks: Tier boundaries (e.g. 0-499 units, 500+ units) and unit prices for each tier.",
            "avg_demand": "Average Demand (d): Mean demand rate per period (daily, weekly, etc.).",
            "lead_time": "Average Lead Time (L): Expected supplier replenishment duration in the same periods.",
            "service_level": "Cycle Service Level (CSL): Target probability of not stocking out during replenishment (e.g., 95% or 0.95).",
            "review_period": "Review Period (T): Time interval between inventory reviews and order placements.",
            "selling_price_or_underage_cost": "Selling Price (P) or Underage Cost (Cu): Revenue earned per unit or profit margin foregone per stockout.",
            "purchase_cost_or_overage_cost": "Purchase Cost (C) or Overage Cost (Co): Procurement cost per unit or net loss per unsold leftover unit.",
            "mean_demand_or_discrete_distribution": "Mean Demand (mu): Expected demand for the single selling period.",
            "std_demand": "Demand Standard Deviation (sigma): Measure of demand variability/uncertainty.",
        }

        lines = [
            f"## ⚠️ Missing Required Information for {model_type.value}\n",
            "To calculate deterministic and mathematically sound inventory recommendations, the agent never invents missing numerical values.\n",
            "### 📋 Successfully Extracted Parameters:",
        ]
        if inputs:
            for k, v in inputs.items():
                if k != "discount_tiers":
                    lines.append(f"- **{k}:** `{v}`")
                else:
                    lines.append(f"- **{k}:** `{len(v)} tiers extracted`")
        else:
            lines.append("- *(No valid numerical parameters could be identified from the question)*")

        lines.append("\n### ❓ Missing Required Parameters:")
        for m in missing:
            desc = param_descriptions.get(m, m)
            lines.append(f"- **{m}:** {desc}")

        lines.append(
            "\n💡 *Please rephrase your question with the missing values to receive optimal replenishment quantities, total costs, and safety stock recommendations.*"
        )
        return "\n".join(lines)

    # =================================================================
    # FULL PIPELINE: Process Customer Question
    # =================================================================
    def process_question(self, question: str) -> AgentDecisionResponse:
        """Run the end-to-end 7-step pipeline on a natural language customer query."""
        # 1. Identify Model
        model_type = self.identify_model(question)
        if model_type == InventoryModelType.UNKNOWN:
            raise ValueError(
                "Could not identify inventory decision model from question. "
                "Please specify details for EOQ, Quantity Discount, Safety Stock & ROP, Periodic Review, or Newsvendor."
            )

        # 2. Extract Inputs
        inputs = self.extract_inputs(question, model_type)
        missing = self.identify_missing_inputs(model_type, inputs)

        # 3. Select Tool
        tool_name, tool_func = self.select_tool(model_type)

        if missing:
            explanation = self.explain_missing_inputs(model_type, inputs, missing)
            return AgentDecisionResponse(
                question=question,
                model_type=model_type,
                extracted_inputs=inputs,
                selected_tool=tool_name,
                calculation_result=None,
                verification=None,
                explanation=explanation,
                missing_inputs=missing,
            )

        # 4. Calculate
        result = self.calculate(tool_func, inputs)

        # 5. Verify
        verification = self.verify(model_type, inputs, result)

        # 6. Explain
        explanation = self.explain(model_type, inputs, result, verification)

        return AgentDecisionResponse(
            question=question,
            model_type=model_type,
            extracted_inputs=inputs,
            selected_tool=tool_name,
            calculation_result=result,
            verification=verification,
            explanation=explanation,
            missing_inputs=[],
        )
