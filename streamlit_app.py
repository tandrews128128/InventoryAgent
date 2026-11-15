"""Supply Chain Analytics: Inventory Decision Agent - Web Dashboard.

A Streamlit web application providing intelligent inventory decision support:
1. Natural Language Inventory Decision Agent (Question -> Model -> Inputs -> Tool -> Calculate -> Verify -> Explain)
2. Economic Order Quantity (EOQ) with Interactive Cost Trade-off Curves
3. Quantity Discount Model with All-Units Schedule Comparison
4. Safety Stock and Reorder Point (ROP) with Lead Time Demand Distributions
5. Fixed-Period Review Model (Order-Up-To Level) with Protection Interval Analysis
6. Newsvendor Model with Profit Curves and Critical Ratio Optimization
7. Formula and Methodology Reference Guide
"""

from __future__ import annotations

import math
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from inventory_agent import (
    InventoryDecisionAgent,
    InventoryModelType,
)
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
    norm_cdf,
    norm_pdf,
    norm_ppf,
)

# -----------------------------------------------------------------------------
# Streamlit Page Configuration & Styling
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Supply Chain Analytics | Inventory Decision Agent",
    page_icon="📦",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E293B;
        margin-bottom: 0.2rem;
    }
    .subtitle {
        font-size: 1.05rem;
        color: #64748B;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 8px;
        padding: 1rem;
        margin-bottom: 0.8rem;
    }
    .stMetric {
        background: #F8FAFC;
        padding: 12px;
        border-radius: 8px;
        border: 1px solid #E2E8F0;
    }
    .step-pill {
        display: inline-block;
        padding: 4px 10px;
        border-radius: 12px;
        font-size: 0.85rem;
        font-weight: 600;
        background-color: #E0E7FF;
        color: #3730A3;
        margin-right: 6px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def get_agent() -> InventoryDecisionAgent:
    return InventoryDecisionAgent()


agent = get_agent()

# -----------------------------------------------------------------------------
# Sidebar Navigation
# -----------------------------------------------------------------------------
with st.sidebar:
    st.image(
        "https://images.unsplash.com/photo-1586528116311-ad8dd3c8310d?auto=format&fit=crop&w=600&q=80",
        use_container_width=True,
    )
    st.markdown("### 📦 Inventory Decision Agent")
    st.markdown(
        "**Decision Pipeline:**  \n"
        "`Question` → `Identify` → `Extract` → `Calculate` → `Verify` → `Explain`"
    )

    st.markdown("---")
    selected_view = st.selectbox(
        "Select Decision Workspace:",
        [
            "🤖 Natural Language Decision Agent",
            "📦 1. Economic Order Quantity (EOQ)",
            "🏷️ 2. Quantity Discount Model",
            "🛡️ 3. Safety Stock & Reorder Point",
            "🔄 4. Fixed-Period Review Model",
            "📰 5. Newsvendor Model",
            "📖 Calculations & Formulas Reference",
        ],
        index=0,
    )

    st.markdown("---")
    st.markdown("#### 💡 Quick Supply Chain Tips")
    st.info(
        "- **EOQ**: Balances ordering and holding costs.\n"
        "- **Discounts**: Compare total acquisition cost, not just purchase price.\n"
        "- **Safety Stock**: Scales with $\\sqrt{L}$; shorter lead times exponentially lower buffer stock.\n"
        "- **Newsvendor**: $CR = C_u / (C_u + C_o)$ dictates whether to stock above or below mean."
    )
    st.caption("Supply Chain Analytics Decision Support System v2.0")


# =============================================================================
# VIEW 1: Natural Language Decision Agent
# =============================================================================
if selected_view == "🤖 Natural Language Decision Agent":
    st.markdown('<div class="main-title">🤖 Natural Language Inventory Decision Agent</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="subtitle">Ask any inventory management or replenishment question in plain English. The agent identifies the model, extracts parameters, executes the verified Python math, and provides executive recommendations.</div>',
        unsafe_allow_html=True,
    )

    # Preset examples
    sample_queries = {
        "Select a template question or type your own below...": "",
        "EOQ: Annual demand 10,000 units, order cost $50, holding cost $2/unit": (
            "Our annual demand is 10,000 units, ordering cost is $50 per order, and holding cost is $2 per unit per year. What is our EOQ and total inventory cost?"
        ),
        "Quantity Discount: 10,000 units demand, 20% holding, 3 price tiers": (
            "Annual demand is 10,000 units, order cost is $50, and holding cost rate is 20%. Price tiers: 0 to 499 units @ $10.00, 500 to 999 units @ $9.50, 1000+ units @ $9.00. What quantity should we order?"
        ),
        "Safety Stock & ROP: Daily demand 100 (std 20), lead time 4 days, 95% CSL": (
            "Daily demand is 100 units with standard deviation of 20 units. Lead time is 4 days. We require a 95% service level. What safety stock and reorder point should we set?"
        ),
        "Periodic Review: Review every 2 wks, lead time 1 wk, weekly demand 50 (std 10), on-hand 40, on-order 20": (
            "We review inventory periodically every 2 weeks. Supplier lead time is 1 week. Weekly demand is 50 units with std dev of 10. Target service level is 95%. Current on-hand is 40 and on-order is 20. What is our order-up-to level and how much should we order?"
        ),
        "Newsvendor: Retail $100, cost $40, salvage $10, normal demand mean 1000, std 200": (
            "We sell a seasonal fashion item for $100 retail price. Purchase cost is $40, and salvage value is $10 for unsold units. Demand is normal with mean 1000 and standard deviation 200. Using the newsvendor model, what is our optimal order quantity?"
        ),
    }

    selected_sample = st.selectbox("💡 Load an Example Question:", list(sample_queries.keys()))
    default_text = sample_queries[selected_sample] if sample_queries[selected_sample] else "Our annual demand is 10,000 units, ordering cost is $50 per order, and holding cost is $2 per unit per year. What is our EOQ?"

    user_query = st.text_area(
        "Enter your inventory or supply chain question in plain English:",
        value=default_text,
        height=90,
        help="Specify demand, costs, lead time, review periods, or probabilities.",
    )

    col_btn1, col_btn2 = st.columns([1, 4])
    with col_btn1:
        run_btn = st.button("🚀 Analyze with Agent", type="primary", use_container_width=True)

    if run_btn and user_query.strip():
        with st.spinner("Executing 7-step decision pipeline..."):
            try:
                response = agent.process_question(user_query.strip())
                st.success("✅ Decision Pipeline Executed Successfully!")

                # Step-by-step pipeline trace
                with st.expander("🔍 View 7-Stage Agent Pipeline Trace", expanded=False):
                    st.markdown(f"**1. Input Question:** `{response.question}`")
                    st.markdown(f"**2. Identified Model:** `{response.model_type.value}`")
                    st.markdown(f"**3. Extracted Inputs:**")
                    st.json(response.extracted_inputs)
                    st.markdown(f"**4. Selected Tool:** `inventory_tools.{response.selected_tool}()`")
                    st.markdown(f"**5. Python Calculation:** Completed successfully.")
                    st.markdown(f"**6. Verification Status:** `{'PASSED ALL CHECKS' if response.verification.is_valid else 'WARNING'}`")
                    for c in response.verification.checks:
                        st.write(f"- {'✅' if c.passed else '❌'} **{c.name}:** {c.details}")

                # Display the executive explanation
                st.markdown(response.explanation)

                # Contextual Interactive Chart depending on identified model
                st.markdown("### 📊 Interactive Visual Model Analysis")
                if response.model_type == InventoryModelType.EOQ:
                    res: EOQResult = response.calculation_result
                    q_vals = np.linspace(max(10, res.optimal_order_quantity * 0.2), res.optimal_order_quantity * 2.5, 200)
                    ord_costs = (res.annual_demand / q_vals) * res.order_cost
                    hold_costs = (q_vals / 2.0) * res.holding_cost_per_unit
                    tot_costs = ord_costs + hold_costs

                    fig = go.Figure()
                    fig.add_trace(go.Scatter(x=q_vals, y=tot_costs, mode="lines", name="Total Inventory Cost", line=dict(color="#2563EB", width=3)))
                    fig.add_trace(go.Scatter(x=q_vals, y=hold_costs, mode="lines", name="Annual Holding Cost", line=dict(color="#10B981", dash="dash")))
                    fig.add_trace(go.Scatter(x=q_vals, y=ord_costs, mode="lines", name="Annual Ordering Cost", line=dict(color="#F59E0B", dash="dash")))
                    fig.add_vline(x=res.optimal_order_quantity, line_dash="dot", line_color="#EF4444", annotation_text=f"Q* = {res.optimal_order_quantity:,.1f}", annotation_position="top right")
                    fig.update_layout(title="EOQ Cost Trade-off & Minimization", xaxis_title="Order Quantity (Q)", yaxis_title="Annual Cost ($)", template="plotly_white", hovermode="x unified")
                    st.plotly_chart(fig, use_container_width=True)

                elif response.model_type == InventoryModelType.SAFETY_STOCK_ROP:
                    res_ss: SafetyStockROPResult = response.calculation_result
                    mu = res_ss.expected_demand_during_lead_time
                    sigma = max(1.0, res_ss.combined_lead_time_std_dev)
                    x_axis = np.linspace(max(0, mu - 4 * sigma), mu + 4 * sigma, 300)
                    y_axis = [norm_pdf((x - mu) / sigma) / sigma for x in x_axis]

                    fig = go.Figure()
                    fig.add_trace(go.Scatter(x=x_axis, y=y_axis, mode="lines", name="Lead Time Demand PDF", line=dict(color="#2563EB", width=2)))
                    # Shaded safety stock buffer
                    x_fill = [x for x in x_axis if mu <= x <= res_ss.reorder_point]
                    y_fill = [norm_pdf((x - mu) / sigma) / sigma for x in x_fill]
                    if x_fill:
                        fig.add_trace(go.Scatter(x=x_fill + [x_fill[-1], x_fill[0]], y=y_fill + [0, 0], fill="toself", fillcolor="rgba(16, 185, 129, 0.3)", line=dict(color="rgba(255,255,255,0)"), name="Safety Stock Buffer"))
                    fig.add_vline(x=mu, line_color="#6B7280", line_dash="dash", annotation_text=f"Exp Demand ({mu:.1f})")
                    fig.add_vline(x=res_ss.reorder_point, line_color="#EF4444", line_width=2.5, annotation_text=f"ROP = {res_ss.reorder_point:.1f} (CSL={res_ss.target_service_level*100:.0f}%)", annotation_position="top right")
                    fig.update_layout(title="Stochastic Demand During Lead Time & Buffer Protection", xaxis_title="Demand During Lead Time (units)", yaxis_title="Probability Density", template="plotly_white")
                    st.plotly_chart(fig, use_container_width=True)

                elif response.model_type == InventoryModelType.NEWSVENDOR:
                    res_nv: NewsvendorResult = response.calculation_result
                    if res_nv.mean_demand and res_nv.std_demand:
                        mu, sigma = res_nv.mean_demand, res_nv.std_demand
                        q_candidates = np.linspace(max(0, mu - 3 * sigma), mu + 3 * sigma, 150)
                        profits = []
                        for q in q_candidates:
                            z = (q - mu) / sigma
                            loss = sigma * (norm_pdf(z) - z * (1.0 - norm_cdf(z)))
                            exp_s = mu - loss
                            exp_left = q - exp_s
                            prof = (res_nv.selling_price * exp_s) + (res_nv.salvage_value * exp_left) - (res_nv.purchase_cost * q)
                            profits.append(prof)

                        fig = go.Figure()
                        fig.add_trace(go.Scatter(x=q_candidates, y=profits, mode="lines", name="Expected Profit ($)", line=dict(color="#10B981", width=3)))
                        fig.add_vline(x=res_nv.optimal_order_quantity, line_dash="dot", line_color="#EF4444", annotation_text=f"Q* = {res_nv.optimal_order_quantity:,.1f} (Max Profit)", annotation_position="top left")
                        fig.update_layout(title="Newsvendor Expected Profit Curve", xaxis_title="Order Quantity (Q)", yaxis_title="Expected Profit ($)", template="plotly_white")
                        st.plotly_chart(fig, use_container_width=True)

            except Exception as e:
                st.error(f"❌ Error during decision execution: {e}")
                st.info("Tip: Make sure to include core parameters like demand, costs, or lead times.")


# =============================================================================
# VIEW 2: 1. Economic Order Quantity (EOQ)
# =============================================================================
elif selected_view == "📦 1. Economic Order Quantity (EOQ)":
    st.markdown('<div class="main-title">📦 Economic Order Quantity (EOQ) Workspace</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="subtitle">The Wilson EOQ model identifies the optimal lot size that minimizes the combined annual costs of placing orders and carrying cycle inventory.</div>',
        unsafe_allow_html=True,
    )

    col_in, col_calc = st.columns([1, 2], gap="large")

    with col_in:
        st.markdown("### ⚙️ Input Parameters")
        demand = st.number_input("Annual Demand ($D$ units/year):", min_value=1.0, value=10000.0, step=500.0)
        order_cost = st.number_input("Ordering / Setup Cost ($S$ $/order):", min_value=0.1, value=50.0, step=5.0)

        holding_mode = st.radio("Holding Cost Definition:", ["Fixed per Unit ($/unit/year)", "% of Unit Purchase Price"])
        if holding_mode == "Fixed per Unit ($/unit/year)":
            holding_cost = st.number_input("Holding Cost ($H$ $/unit/year):", min_value=0.01, value=2.0, step=0.5)
            unit_price = st.number_input("Unit Purchase Price ($/unit, optional):", min_value=0.0, value=10.0, step=1.0)
        else:
            unit_price = st.number_input("Unit Purchase Price ($/unit):", min_value=0.01, value=10.0, step=1.0)
            holding_rate_pct = st.slider("Annual Carrying Cost Rate (%):", min_value=1.0, max_value=60.0, value=20.0, step=1.0)
            holding_cost = (holding_rate_pct / 100.0) * unit_price
            st.caption(f"Calculated Holding Cost $H = {holding_rate_pct}\\% \\times \\${unit_price:.2f} = \\${holding_cost:.2f}$/unit/year")

        lead_time_days = st.number_input("Lead Time (days, optional):", min_value=0.0, value=5.0, step=1.0)
        work_days = st.number_input("Operating Days / Year:", min_value=100.0, max_value=365.0, value=250.0, step=10.0)

    # Perform calculation
    res_eoq = calculate_eoq(
        demand=demand,
        order_cost=order_cost,
        holding_cost=holding_cost,
        unit_cost=unit_price if unit_price > 0 else None,
        lead_time_days=lead_time_days if lead_time_days > 0 else None,
        work_days_per_year=work_days,
    )

    with col_calc:
        st.markdown("### 🎯 Optimal Decision Metrics")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Optimal Order Size (Q*)", f"{res_eoq.optimal_order_quantity:,.1f} units")
        m2.metric("Orders per Year", f"{res_eoq.orders_per_year:.2f} orders")
        m3.metric("Cycle Time", f"{res_eoq.cycle_time_days:.1f} days")
        m4.metric("Total Inventory Cost", f"${res_eoq.total_inventory_cost:,.2f}")

        if res_eoq.reorder_point_deterministic is not None:
            st.info(f"📍 **Deterministic Reorder Point (ROP):** `{res_eoq.reorder_point_deterministic:,.1f} units` (Place order when inventory hits this level to cover {lead_time_days} days lead time)")

        # Interactive Cost Curve Plot
        q_min = max(5.0, res_eoq.optimal_order_quantity * 0.25)
        q_max = res_eoq.optimal_order_quantity * 2.5
        q_grid = np.linspace(q_min, q_max, 250)
        ordering_curve = (demand / q_grid) * order_cost
        holding_curve = (q_grid / 2.0) * holding_cost
        total_curve = ordering_curve + holding_curve

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=q_grid, y=total_curve, mode="lines", name="Total Inventory Cost (TOC + THC)", line=dict(color="#1D4ED8", width=3.5)))
        fig.add_trace(go.Scatter(x=q_grid, y=holding_curve, mode="lines", name="Annual Holding Cost (Q/2 * H)", line=dict(color="#059669", dash="dash", width=2)))
        fig.add_trace(go.Scatter(x=q_grid, y=ordering_curve, mode="lines", name="Annual Ordering Cost (D/Q * S)", line=dict(color="#D97706", dash="dash", width=2)))
        fig.add_vline(x=res_eoq.optimal_order_quantity, line_dash="dot", line_color="#DC2626", annotation_text=f"Q* = {res_eoq.optimal_order_quantity:,.1f}", annotation_position="top right")

        fig.update_layout(
            title="EOQ Cost Minimization Trade-Off",
            xaxis_title="Order Quantity (Q in units)",
            yaxis_title="Annual Cost ($/year)",
            template="plotly_white",
            hovermode="x unified",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("#### 📋 Cost Breakdown & Parity Check")
        df_breakdown = pd.DataFrame(
            [
                {"Metric": "Optimal Order Quantity (Q*)", "Value": f"{res_eoq.optimal_order_quantity:,.2f} units", "Notes": "Wilson lot size formula"},
                {"Metric": "Annual Ordering Cost (TOC)", "Value": f"${res_eoq.annual_ordering_cost:,.2f}", "Notes": f"{res_eoq.orders_per_year:.1f} orders @ ${order_cost}/order"},
                {"Metric": "Annual Holding Cost (THC)", "Value": f"${res_eoq.annual_holding_cost:,.2f}", "Notes": f"Holding {res_eoq.average_cycle_inventory:,.1f} avg units"},
                {"Metric": "Total Inventory Cost (TIC)", "Value": f"${res_eoq.total_inventory_cost:,.2f}", "Notes": "TOC + THC (Minimization target)"},
                {"Metric": "Annual Acquisition Cost (TAC)", "Value": f"${res_eoq.annual_purchasing_cost:,.2f}" if res_eoq.annual_purchasing_cost else "N/A", "Notes": "Procurement cost ($D \\times C$)"},
                {"Metric": "Total Supply Chain Cost", "Value": f"${res_eoq.total_annual_cost:,.2f}", "Notes": "TAC + TOC + THC"},
            ]
        )
        st.table(df_breakdown)

        with st.expander("🎓 Student Verification Walkthrough (8-Step Audit)", expanded=False):
            st.markdown(res_eoq.format_student_verification_markdown())


# =============================================================================
# VIEW 3: 2. Quantity Discount Model
# =============================================================================
elif selected_view == "🏷️ 2. Quantity Discount Model":
    st.markdown('<div class="main-title">🏷️ Quantity Discount Model Workspace</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="subtitle">Evaluates all-units volume discount schedules. Balances purchase price savings against increased inventory carrying costs to determine the true global cost-minimizing order size.</div>',
        unsafe_allow_html=True,
    )

    col_in, col_calc = st.columns([1, 2], gap="large")

    with col_in:
        st.markdown("### ⚙️ Operational Parameters")
        qd_demand = st.number_input("Annual Demand ($D$):", min_value=1.0, value=10000.0, step=500.0)
        qd_order_cost = st.number_input("Order Cost ($S$):", min_value=0.1, value=50.0, step=5.0)
        qd_holding_rate = st.slider("Annual Carrying Rate ($h$ % of unit price):", min_value=1.0, max_value=50.0, value=20.0, step=1.0) / 100.0

        st.markdown("### 📋 Discount Tiers Schedule")
        st.caption("Customize the brackets and price points:")

        t1_price = st.number_input("Tier 1 (0 - 499 units) Price ($):", min_value=0.1, value=10.00, step=0.25)
        t2_min = st.number_input("Tier 2 Min Quantity:", min_value=1, value=500, step=50)
        t2_price = st.number_input(f"Tier 2 ({t2_min} - 999 units) Price ($):", min_value=0.1, value=9.50, step=0.25)
        t3_min = st.number_input("Tier 3 Min Quantity (Bulk Breakpoint):", min_value=t2_min + 1, value=1000, step=100)
        t3_price = st.number_input(f"Tier 3 ({t3_min}+ units) Price ($):", min_value=0.1, value=9.00, step=0.25)

        tiers = [
            {"min_qty": 0, "max_qty": t2_min - 1, "price": t1_price},
            {"min_qty": t2_min, "max_qty": t3_min - 1, "price": t2_price},
            {"min_qty": t3_min, "max_qty": None, "price": t3_price},
        ]

    res_qd = calculate_quantity_discount(
        demand=qd_demand,
        order_cost=qd_order_cost,
        discount_tiers=tiers,
        holding_cost_rate=qd_holding_rate,
    )

    with col_calc:
        st.markdown("### 🎯 Optimal Procurement Decision")
        q1, q2, q3, q4 = st.columns(4)
        q1.metric("Optimal Order Qty", f"{res_qd.optimal_order_quantity:,.1f} units")
        q2.metric("Optimal Tier", f"Tier {res_qd.optimal_tier_index}")
        q3.metric("Optimal Price", f"${res_qd.optimal_unit_price:,.2f}/unit")
        q4.metric("Net Annual Savings", f"${res_qd.annual_cost_savings:,.2f}", delta=f"${res_qd.annual_cost_savings:,.2f}")

        # Tier breakdown table
        st.markdown("#### 📊 Comprehensive Tier Evaluations")
        tier_rows = []
        for t in res_qd.tier_evaluations:
            bracket = f"{t.min_quantity:,.0f} - {t.max_quantity:,.0f}" if t.max_quantity else f"{t.min_quantity:,.0f}+"
            tier_rows.append({
                "Tier": f"Tier {t.tier_index}",
                "Bracket": bracket,
                "Price": f"${t.unit_price:.2f}",
                "Raw EOQ": f"{t.raw_eoq:,.1f}",
                "Feasible Order Qty": f"{t.realized_order_quantity:,.1f}",
                "Annual Purchase Cost": f"${t.annual_purchase_cost:,.2f}",
                "Annual Order + Hold": f"${t.annual_ordering_cost + t.annual_holding_cost:,.2f}",
                "Total Annual Cost": f"${t.total_annual_cost:,.2f}",
                "Optimal?": "⭐ Optimal" if t.tier_index == res_qd.optimal_tier_index else "Suboptimal",
            })
        st.dataframe(pd.DataFrame(tier_rows), use_container_width=True)

        # Plotly chart comparing total cost across tiers
        fig_qd = go.Figure()
        q_sweep = np.linspace(100, max(2000.0, res_qd.optimal_order_quantity * 1.5), 300)

        # Curve for each price
        colors = ["#94A3B8", "#3B82F6", "#10B981"]
        for idx, t in enumerate(res_qd.tier_evaluations):
            p = t.unit_price
            h = qd_holding_rate * p
            tc_curve = (qd_demand * p) + (qd_demand / q_sweep) * qd_order_cost + (q_sweep / 2.0) * h
            fig_qd.add_trace(go.Scatter(x=q_sweep, y=tc_curve, mode="lines", name=f"Cost Curve @ ${p:.2f} (Tier {t.tier_index})", line=dict(color=colors[idx % len(colors)], dash="dot")))

        # Add marker for realized order quantity
        for idx, t in enumerate(res_qd.tier_evaluations):
            is_best = t.tier_index == res_qd.optimal_tier_index
            fig_qd.add_trace(
                go.Scatter(
                    x=[t.realized_order_quantity],
                    y=[t.total_annual_cost],
                    mode="markers+text",
                    name=f"Tier {t.tier_index} Decision",
                    text=[f"Tier {t.tier_index}: Q={t.realized_order_quantity:.0f}"],
                    textposition="top center",
                    marker=dict(size=14 if is_best else 10, color="#DC2626" if is_best else colors[idx % len(colors)], symbol="star" if is_best else "circle"),
                )
            )

        fig_qd.update_layout(
            title="Total Annual Acquisition & Inventory Cost Across Quantity Breaks",
            xaxis_title="Order Quantity (Q)",
            yaxis_title="Total Annual Cost ($)",
            template="plotly_white",
            legend=dict(orientation="h", y=-0.2),
        )
        st.plotly_chart(fig_qd, use_container_width=True)

        with st.expander("🎓 Student Verification Walkthrough (8-Step Audit)", expanded=False):
            st.markdown(res_qd.format_student_verification_markdown())


# =============================================================================
# VIEW 4: 3. Safety Stock & Reorder Point
# =============================================================================
elif selected_view == "🛡️ 3. Safety Stock & Reorder Point":
    st.markdown('<div class="main-title">🛡️ Safety Stock & Reorder Point (ROP) Workspace</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="subtitle">Calculates the optimal replenishment trigger level (ROP) and safety stock buffer under stochastic demand and supplier lead time uncertainty.</div>',
        unsafe_allow_html=True,
    )

    col_in, col_calc = st.columns([1, 2], gap="large")

    with col_in:
        st.markdown("### ⚙️ Uncertainty Inputs")
        avg_d = st.number_input("Average Demand per Period ($d$ units/day):", min_value=0.1, value=100.0, step=10.0)
        lt_periods = st.number_input("Replenishment Lead Time ($L$ days):", min_value=0.1, value=4.0, step=0.5)
        csl = st.slider("Target Cycle Service Level (CSL %):", min_value=50.0, max_value=99.9, value=95.0, step=0.5) / 100.0

        st.markdown("#### 🌪️ Sources of Variance")
        std_d = st.number_input("Demand Standard Deviation ($\\sigma_d$ units/day):", min_value=0.0, value=20.0, step=2.0)
        std_lt = st.number_input("Lead Time Standard Deviation ($\\sigma_L$ days):", min_value=0.0, value=0.0, step=0.5)

        unit_h = st.number_input("Unit Holding Cost per Year ($/unit/yr, optional):", min_value=0.0, value=5.0, step=1.0)
        order_q = st.number_input("Replenishment Lot Size ($Q$, optional for fill rate):", min_value=0.0, value=500.0, step=50.0)

    res_ss = calculate_safety_stock_and_rop(
        avg_demand_per_period=avg_d,
        lead_time_periods=lt_periods,
        service_level=csl,
        std_demand_per_period=std_d,
        std_lead_time_periods=std_lt,
        unit_holding_cost=unit_h if unit_h > 0 else None,
        order_quantity=order_q if order_q > 0 else None,
    )

    with col_calc:
        st.markdown("### 🎯 Reorder Point Policy")
        s1, s2, s3, s4 = st.columns(4)
        s1.metric("Reorder Point (ROP)", f"{res_ss.reorder_point:,.1f} units")
        s2.metric("Required Safety Stock", f"{res_ss.safety_stock:,.1f} units")
        s3.metric("Normal Z-Score", f"{res_ss.z_score:.4f}")
        s4.metric("Lead Time Demand", f"{res_ss.expected_demand_during_lead_time:,.1f} units")

        if res_ss.implied_fill_rate is not None:
            st.success(f"📈 **Implied Fill Rate (Type 2 Service Level):** `{res_ss.implied_fill_rate*100:.2f}%` (Expected Shortage: `{res_ss.expected_units_short_per_cycle:.3f} units/cycle`)")

        # Distribution Bell Curve
        mu = res_ss.expected_demand_during_lead_time
        sigma = max(1.0, res_ss.combined_lead_time_std_dev)
        x_pts = np.linspace(max(0, mu - 4 * sigma), mu + 4 * sigma, 300)
        y_pts = [norm_pdf((x - mu) / sigma) / sigma for x in x_pts]

        fig_ss = go.Figure()
        fig_ss.add_trace(go.Scatter(x=x_pts, y=y_pts, mode="lines", name="Demand During Lead Time PDF", line=dict(color="#1E40AF", width=2.5)))

        # Fill Safety stock zone
        x_ss = [x for x in x_pts if mu <= x <= res_ss.reorder_point]
        y_ss = [norm_pdf((x - mu) / sigma) / sigma for x in x_ss]
        if x_ss:
            fig_ss.add_trace(go.Scatter(x=x_ss + [x_ss[-1], x_ss[0]], y=y_ss + [0, 0], fill="toself", fillcolor="rgba(16, 185, 129, 0.35)", line=dict(color="rgba(255,255,255,0)"), name=f"Safety Stock ({res_ss.safety_stock:.1f} units)"))

        # Fill Stockout risk zone
        x_stockout = [x for x in x_pts if x >= res_ss.reorder_point]
        y_stockout = [norm_pdf((x - mu) / sigma) / sigma for x in x_stockout]
        if x_stockout:
            fig_ss.add_trace(go.Scatter(x=x_stockout + [x_stockout[-1], x_stockout[0]], y=y_stockout + [0, 0], fill="toself", fillcolor="rgba(239, 68, 68, 0.35)", line=dict(color="rgba(255,255,255,0)"), name=f"Stockout Risk ({ (1.0 - csl)*100:.1f}%)"))

        fig_ss.add_vline(x=mu, line_dash="dash", line_color="#4B5563", annotation_text=f"Expected DDLT: {mu:.1f}")
        fig_ss.add_vline(x=res_ss.reorder_point, line_color="#DC2626", line_width=3, annotation_text=f"ROP = {res_ss.reorder_point:.1f}", annotation_position="top right")

        fig_ss.update_layout(
            title=f"Lead Time Demand Distribution (Scenario: {res_ss.variability_scenario.replace('_', ' ').title()})",
            xaxis_title="Demand During Lead Time (units)",
            yaxis_title="Probability Density",
            template="plotly_white",
        )
        st.plotly_chart(fig_ss, use_container_width=True)

        # Sensitivity curve: Safety Stock vs Service Level
        csl_sweep = np.linspace(0.80, 0.999, 100)
        ss_sweep = [norm_ppf(p) * sigma for p in csl_sweep]
        fig_sens = go.Figure()
        fig_sens.add_trace(go.Scatter(x=csl_sweep * 100, y=ss_sweep, mode="lines", name="Safety Stock Required", line=dict(color="#6366F1", width=3)))
        fig_sens.add_vline(x=csl * 100, line_dash="dot", line_color="#EF4444", annotation_text=f"Selected CSL: {csl*100:.1f}%")
        fig_sens.update_layout(title="Exponential Carrying Cost of Higher Service Levels", xaxis_title="Cycle Service Level (CSL %)", yaxis_title="Required Safety Stock (units)", template="plotly_white")
        st.plotly_chart(fig_sens, use_container_width=True)

        with st.expander("🎓 Student Verification Walkthrough (8-Step Audit)", expanded=False):
            st.markdown(res_ss.format_student_verification_markdown())


# =============================================================================
# VIEW 5: 4. Fixed-Period Review Model
# =============================================================================
elif selected_view == "🔄 4. Fixed-Period Review Model":
    st.markdown('<div class="main-title">🔄 Fixed-Period Review Model Workspace</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="subtitle">A periodic review $(T, S)$ system reviews stock every $T$ periods and orders up to target level $S$. Must protect against stockouts across the full protection interval $(T + L)$.</div>',
        unsafe_allow_html=True,
    )

    col_in, col_calc = st.columns([1, 2], gap="large")

    with col_in:
        st.markdown("### ⚙️ Periodic Parameters")
        pr_demand = st.number_input("Periodic Demand ($d$ units/week):", min_value=0.1, value=50.0, step=5.0)
        pr_review_T = st.number_input("Review Period ($T$ weeks):", min_value=0.1, value=2.0, step=0.5)
        pr_lead_L = st.number_input("Supplier Lead Time ($L$ weeks):", min_value=0.0, value=1.0, step=0.5)
        pr_std_d = st.number_input("Demand Std Dev per Period ($\\sigma_d$):", min_value=0.0, value=10.0, step=1.0)
        pr_csl = st.slider("Target Service Level (CSL %):", min_value=50.0, max_value=99.9, value=95.0, step=0.5) / 100.0

        st.markdown("#### 🏢 Current Stock Status")
        on_hand = st.number_input("Current On-Hand Stock:", min_value=0.0, value=40.0, step=5.0)
        on_order = st.number_input("Current On-Order / In-Transit:", min_value=0.0, value=20.0, step=5.0)
        backorders = st.number_input("Current Backorders:", min_value=0.0, value=0.0, step=5.0)

    res_pr = calculate_periodic_review(
        avg_demand_per_period=pr_demand,
        review_period_T=pr_review_T,
        lead_time_L=pr_lead_L,
        service_level=pr_csl,
        std_demand_per_period=pr_std_d,
        on_hand=on_hand,
        on_order=on_order,
        backorders=backorders,
    )

    with col_calc:
        st.markdown("### 🎯 Order-Up-To Replenishment Decision")
        p1, p2, p3, p4 = st.columns(4)
        p1.metric("Target Level (S)", f"{res_pr.target_inventory_level_S:,.1f} units")
        p2.metric("Order to Place Now (Q)", f"{res_pr.recommended_order_quantity_Q:,.1f} units")
        p3.metric("Safety Stock Buffer", f"{res_pr.safety_stock:,.1f} units")
        p4.metric("Inventory Position (IP)", f"{res_pr.current_inventory_position:,.1f} units")

        # Visual bar chart showing IP bridging to S
        fig_pr = go.Figure()
        fig_pr.add_trace(go.Bar(name="On-Hand Inventory", x=["Inventory Composition"], y=[on_hand], marker_color="#3B82F6"))
        fig_pr.add_trace(go.Bar(name="On-Order / In-Transit", x=["Inventory Composition"], y=[on_order], marker_color="#60A5FA"))
        fig_pr.add_trace(go.Bar(name="Recommended Order Q", x=["Inventory Composition"], y=[res_pr.recommended_order_quantity_Q], marker_color="#10B981"))
        fig_pr.add_hline(y=res_pr.target_inventory_level_S, line_dash="dash", line_color="#DC2626", annotation_text=f"Target Base Stock S = {res_pr.target_inventory_level_S:,.1f}")

        fig_pr.update_layout(
            barmode="stack",
            title=f"Inventory Bridge to Target Order-Up-To Level S (Protection Interval: {res_pr.protection_interval_T_plus_L:.1f} weeks)",
            yaxis_title="Units",
            template="plotly_white",
        )
        st.plotly_chart(fig_pr, use_container_width=True)

        st.markdown("#### 📋 Protection Interval Economics")
        pr_df = pd.DataFrame([
            {"Parameter": "Protection Interval ($T + L$)", "Value": f"{res_pr.protection_interval_T_plus_L:.1f} periods", "Significance": "Vulnerability duration before next order arrives"},
            {"Parameter": "Expected Protection Demand", "Value": f"{res_pr.expected_demand_during_protection_interval:,.1f} units", "Significance": f"$d \\times (T + L) = {pr_demand} \\times {res_pr.protection_interval_T_plus_L:.1f}$"},
            {"Parameter": "Protection Interval Std Dev", "Value": f"{res_pr.std_dev_protection_interval:,.2f} units", "Significance": "$\\sigma_d \\sqrt{T + L}$"},
            {"Parameter": "Safety Stock", "Value": f"{res_pr.safety_stock:,.2f} units", "Significance": "$Z \\times \\sigma_{T+L}$"},
            {"Parameter": "Average Cycle Inventory", "Value": f"{res_pr.average_cycle_stock:,.1f} units", "Significance": "$d \\times T / 2$"},
            {"Parameter": "Average Total Inventory", "Value": f"{res_pr.expected_average_inventory:,.1f} units", "Significance": "Safety Stock + Cycle Stock"},
        ])
        st.table(pr_df)

        with st.expander("🎓 Student Verification Walkthrough (8-Step Audit)", expanded=False):
            st.markdown(res_pr.format_student_verification_markdown())


# =============================================================================
# VIEW 6: 5. Newsvendor Model
# =============================================================================
elif selected_view == "📰 5. Newsvendor Model":
    st.markdown('<div class="main-title">📰 Newsvendor Model Workspace</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="subtitle">Solves single-period stochastic inventory problems for perishable, fashion, seasonal, or short lifecycle products by optimizing the Critical Ratio trade-off between underage and overage costs.</div>',
        unsafe_allow_html=True,
    )

    col_in, col_calc = st.columns([1, 2], gap="large")

    with col_in:
        st.markdown("### ⚙️ Unit Financials")
        price = st.number_input("Selling / Retail Price ($P$):", min_value=1.0, value=100.0, step=5.0)
        cost = st.number_input("Purchase / Wholesale Cost ($C$):", min_value=0.1, value=40.0, step=5.0)
        salvage = st.number_input("Salvage / Clearance Value ($V$):", min_value=0.0, max_value=cost, value=10.0, step=2.0)
        penalty = st.number_input("Shortage / Goodwill Penalty ($g$, optional):", min_value=0.0, value=0.0, step=1.0)

        st.markdown("### 📊 Demand Distribution")
        dist_choice = st.radio("Distribution Type:", ["Normal Distribution", "Discrete Scenarios"])

        if dist_choice == "Normal Distribution":
            mean_d = st.number_input("Mean Demand ($\\mu$):", min_value=1.0, value=1000.0, step=50.0)
            std_d = st.number_input("Demand Standard Deviation ($\\sigma$):", min_value=1.0, value=200.0, step=10.0)
            disc_dem, disc_prob = None, None
        else:
            st.caption("Discrete Demand Points & Probabilities:")
            disc_dem = [800.0, 900.0, 1000.0, 1100.0, 1200.0]
            disc_prob = [0.15, 0.20, 0.30, 0.20, 0.15]
            st.write(pd.DataFrame({"Demand": disc_dem, "Probability": disc_prob}))
            mean_d, std_d = None, None

    res_nv = calculate_newsvendor(
        selling_price=price,
        purchase_cost=cost,
        salvage_value=salvage,
        shortage_penalty=penalty,
        mean_demand=mean_d,
        std_demand=std_d,
        discrete_demand=disc_dem,
        discrete_probs=disc_prob,
    )

    with col_calc:
        st.markdown("### 🎯 Optimal Stocking Decision")
        n1, n2, n3, n4 = st.columns(4)
        n1.metric("Optimal Order Size (Q*)", f"{res_nv.optimal_order_quantity:,.1f} units")
        n2.metric("Critical Ratio (Target CSL)", f"{res_nv.critical_ratio*100:.1f}%")
        n3.metric("Expected Profit", f"${res_nv.expected_profit:,.2f}")
        n4.metric("Value of Perfect Info (EVPI)", f"${res_nv.expected_value_of_perfect_information:,.2f}" if res_nv.expected_value_of_perfect_information else "N/A")

        st.markdown(
            f"**Economic Trade-Offs:**  \n"
            f"- **Cost of Underage ($C_u = P - C + g$):** `\\${res_nv.cost_of_underage_Cu:.2f}` (Margin lost per unmet unit)  \n"
            f"- **Cost of Overage ($C_o = C - V$):** `\\${res_nv.cost_of_overage_Co:.2f}` (Net loss per leftover unit)  \n"
            f"- **Critical Ratio ($CR = C_u / (C_u + C_o)$):** `{res_nv.critical_ratio:.4f}`"
        )

        # Expected Profit Curve
        if res_nv.distribution_type == "normal" and mean_d and std_d:
            q_grid = np.linspace(max(0, mean_d - 3 * std_d), mean_d + 3 * std_d, 200)
            profit_curve = []
            for q in q_grid:
                z_q = (q - mean_d) / std_d
                loss_q = std_d * (norm_pdf(z_q) - z_q * (1.0 - norm_cdf(z_q)))
                exp_s_q = mean_d - loss_q
                exp_left_q = q - exp_s_q
                prof_q = (price * exp_s_q) + (salvage * exp_left_q) - (cost * q) - (penalty * loss_q)
                profit_curve.append(prof_q)

            fig_nv = go.Figure()
            fig_nv.add_trace(go.Scatter(x=q_grid, y=profit_curve, mode="lines", name="Expected Profit ($)", line=dict(color="#059669", width=3)))
            fig_nv.add_vline(x=res_nv.optimal_order_quantity, line_dash="dash", line_color="#DC2626", annotation_text=f"Q* = {res_nv.optimal_order_quantity:,.1f}")
            fig_nv.update_layout(title="Expected Profit as a Function of Order Quantity", xaxis_title="Order Quantity (Q)", yaxis_title="Expected Profit ($)", template="plotly_white")
            st.plotly_chart(fig_nv, use_container_width=True)

        with st.expander("🎓 Student Verification Walkthrough (8-Step Audit)", expanded=False):
            st.markdown(res_nv.format_student_verification_markdown())


# =============================================================================
# VIEW 7: Calculations & Formulas Reference Guide
# =============================================================================
elif selected_view == "📖 Calculations & Formulas Reference":
    st.markdown('<div class="main-title">📖 Calculations & Formulas Reference Guide</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="subtitle">Complete mathematical documentation, derivations, and supply chain governance rules implemented in the decision agent.</div>',
        unsafe_allow_html=True,
    )

    with st.expander("📦 1. Economic Order Quantity (Wilson EOQ Formulation)", expanded=True):
        st.markdown(
            r"""
            ### Objective & Assumptions
            The classical EOQ model determines the lot size $Q$ that minimizes the sum of setup and inventory carrying costs under continuous, deterministic demand.
            
            #### Total Cost Function:
            $$TC(Q) = \text{Annual Ordering Cost} + \text{Annual Holding Cost} = \frac{D}{Q} S + \frac{Q}{2} H$$
            
            #### First-Order Condition for Optimality:
            $$\frac{dTC}{dQ} = -\frac{DS}{Q^2} + \frac{H}{2} = 0 \implies Q^* = \sqrt{\frac{2DS}{H}}$$
            
            #### Key Economic Properties:
            - **Cost Equivalence at $Q^*$:** Annual Ordering Cost equals Annual Holding Cost ($\frac{D}{Q^*}S = \frac{Q^*}{2}H = \sqrt{\frac{DSH}{2}}$).
            - **Cycle Time:** $T = \frac{Q^*}{D}$ years, or $T_{\text{days}} = \frac{Q^*}{D} \times \text{Workdays}$.
            - **Deterministic Reorder Point:** $ROP = d_{\text{daily}} \times L$.
            """
        )

    with st.expander("🏷️ 2. Quantity Discount Model (All-Units Schedule)", expanded=False):
        st.markdown(
            r"""
            ### Algorithm for All-Units Quantity Discounts
            Suppliers offer price tiers:
            $$C(Q) = P_i \quad \text{for } Q_{\min, i} \le Q \le Q_{\max, i}$$
            
            1. For each price tier $i$, calculate the unconstrained $EOQ_i$:
               $$EOQ_i = \sqrt{\frac{2DS}{h \cdot P_i}}$$
            2. **Feasibility Check:**
               - If $Q_{\min, i} \le EOQ_i \le Q_{\max, i}$, the candidate order quantity is $Q_i = EOQ_i$.
               - If $EOQ_i < Q_{\min, i}$, the candidate quantity is the breakpoint minimum $Q_i = Q_{\min, i}$.
               - If $EOQ_i > Q_{\max, i}$, this tier is dominated.
            3. Compute Total Annual Acquisition & Inventory Cost for each candidate:
               $$TC(Q_i) = D \cdot P_i + \frac{D}{Q_i} S + \frac{Q_i}{2} (h \cdot P_i)$$
            4. Choose $Q^*$ corresponding to $\min_i TC(Q_i)$.
            """
        )

    with st.expander("🛡️ 3. Safety Stock & Reorder Point (Continuous Review)", expanded=False):
        st.markdown(
            r"""
            ### Continuous Review $(s, Q)$ System
            Protects against stockouts during supplier lead time $L$.
            
            #### Lead Time Demand Variance:
            - **Variable Demand, Constant Lead Time:** $\sigma_{DL} = \sigma_D \sqrt{L}$
            - **Constant Demand, Variable Lead Time:** $\sigma_{DL} = d \cdot \sigma_L$
            - **Both Variable:** $\sigma_{DL} = \sqrt{L \sigma_D^2 + d^2 \sigma_L^2}$
            
            #### Decision Variables:
            - **Safety Stock ($SS$):** $SS = Z \times \sigma_{DL} \quad \text{where } Z = \Phi^{-1}(\text{CSL})$
            - **Reorder Point ($ROP$):** $ROP = \mathbb{E}[\text{DDLT}] + SS = (d \cdot L) + Z \sigma_{DL}$
            - **Expected Shortage per Replenishment Cycle ($ESC$):**
              $$ESC = \sigma_{DL} \left[ \phi(Z) - Z(1 - \Phi(Z)) \right]$$
            - **Fill Rate (Type 2 Service Level):** $\text{Fill Rate} = 1 - \frac{ESC}{Q}$
            """
        )

    with st.expander("🔄 4. Fixed-Period Review Model (Periodic Review)", expanded=False):
        st.markdown(
            r"""
            ### Periodic Review $(T, S)$ Policy
            Inventory is reviewed every $T$ periods. Order arrives after lead time $L$.
            
            #### Protection Interval:
            The system is exposed to stockouts over $T + L$ periods.
            $$\sigma_{T+L} = \sigma_d \sqrt{T + L}$$
            $$SS = Z \times \sigma_d \sqrt{T + L}$$
            
            #### Target Base Stock (Order-Up-To Level $S$):
            $$S = d(T + L) + SS$$
            
            #### Current Replenishment Quantity ($Q$):
            $$\text{Inventory Position (IP)} = \text{On-Hand} + \text{On-Order} - \text{Backorders}$$
            $$Q = \max(0, S - IP)$$
            """
        )

    with st.expander("📰 5. Newsvendor Model (Single-Period Stochastic)", expanded=False):
        st.markdown(
            r"""
            ### Single-Period Marginal Analysis
            Balances the marginal benefit of ordering one more unit against its marginal cost:
            - **Underage Cost ($C_u$):** Profit foregone if demand exceeds order ($C_u = P - C + g$).
            - **Overage Cost ($C_o$):** Loss incurred if unit remains unsold ($C_o = C - V$).
            
            #### Critical Ratio Condition:
            $$F(Q^*) = P(D \le Q^*) = \frac{C_u}{C_u + C_o} = CR$$
            
            #### For Normally Distributed Demand $D \sim \mathcal{N}(\mu, \sigma^2)$:
            $$Q^* = \mu + \Phi^{-1}(CR) \cdot \sigma$$
            
            #### Expected Value of Perfect Information (EVPI):
            $$\text{EVPI} = \mathbb{E}[\text{Profit}_{\text{Perfect Info}}] - \mathbb{E}[\text{Profit}_{\text{Optimal Newsvendor}}]$$
            """
        )
