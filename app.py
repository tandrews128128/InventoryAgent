"""Supply Chain Analytics Inventory Decision Agent Application.

Interactive CLI application providing decision support for:
1. Economic Order Quantity (EOQ)
2. Quantity Discount Model
3. Safety Stock and Reorder Point (ROP)
4. Fixed-Period Review Model
5. Newsvendor Model

Architecture Pipeline:
Customer Question -> Identify Model -> Extract Inputs -> Select Python Tool -> Calculate -> Verify -> Explain
"""

import argparse
import json
import sys
from typing import Optional

from inventory_agent import InventoryDecisionAgent, InventoryModelType

BANNER = r"""
================================================================================
   SUPPLY CHAIN ANALYTICS: INVENTORY DECISION AGENT
   Orchestration: Question -> Model -> Inputs -> Tool -> Calculate -> Verify -> Explain
================================================================================
"""

SAMPLE_QUESTIONS = [
    {
        "model": "Economic Order Quantity (EOQ)",
        "question": "Our annual demand is 10,000 units, ordering cost is $50 per order, and holding cost is $2 per unit per year. What is our EOQ and total inventory cost?",
    },
    {
        "model": "Quantity Discount Model",
        "question": "Annual demand is 10,000 units, order cost is $50, and holding cost rate is 20%. Price tiers: 0 to 499 units @ $10.00, 500 to 999 units @ $9.50, 1000+ units @ $9.00. What quantity should we order?",
    },
    {
        "model": "Safety Stock & Reorder Point",
        "question": "Daily demand is 100 units with standard deviation of 20 units. Lead time is 4 days. We require a 95% service level. What safety stock and reorder point should we set?",
    },
    {
        "model": "Fixed-Period Review Model",
        "question": "We review inventory periodically every 2 weeks. Supplier lead time is 1 week. Weekly demand is 50 units with std dev of 10. Target service level is 95%. Current on-hand is 40 and on-order is 20. What is our order-up-to level and how much should we order?",
    },
    {
        "model": "Newsvendor Model",
        "question": "We sell a seasonal fashion item for $100 retail price. Purchase cost is $40, and salvage value is $10 for unsold units. Demand is normal with mean 1000 and standard deviation 200. Using the newsvendor model, what is our optimal order quantity?",
    },
]


def print_response(agent_response, json_output: bool = False):
    """Print the decision analysis either as JSON or formatted report."""
    if json_output:
        print(json.dumps(agent_response.to_dict(), indent=2))
        return

    print("\n" + "=" * 80)
    print(f"🔎 INVENTORY DECISION PIPELINE EXECUTION")
    print("=" * 80)
    print(f"1. 📥 Customer Question:\n   \"{agent_response.question}\"")
    print(f"\n2. 🏷️  Identified Model:\n   {agent_response.model_type.value}")
    print(f"\n3. ⚙️  Extracted Inputs:")
    for k, v in agent_response.extracted_inputs.items():
        print(f"   - {k}: {v}")
    print(f"\n4. 🛠️  Selected Python Tool:\n   inventory_tools.{agent_response.selected_tool}()")
    print(f"\n5. 🔢 Numerical Calculation:\n   [Completed successfully in Python]")
    print(f"\n6. 🛡️  Verification:\n   Status: {'PASSED ALL CHECKS ✅' if agent_response.verification.is_valid else 'CHECK FAILED ⚠️'}")
    for c in agent_response.verification.checks:
        status_icon = "✅" if c.passed else "❌"
        print(f"   {status_icon} {c.name}: {c.details}")
    print("\n7. 📊 Executive Explanation:")
    print("-" * 80)
    print(agent_response.explanation)
    print("=" * 80 + "\n")


def run_demo(agent: InventoryDecisionAgent, json_output: bool = False):
    """Run through all 5 inventory decision scenarios."""
    print("\n🚀 Running Supply Chain Decision Agent Demo across all 5 Models...\n")
    for i, sample in enumerate(SAMPLE_QUESTIONS, start=1):
        print(f"\n>>> Scenario {i}/5: {sample['model']}")
        try:
            resp = agent.process_question(sample["question"])
            print_response(resp, json_output=json_output)
        except Exception as e:
            print(f"❌ Error processing scenario {i}: {e}")


def interactive_loop(agent: InventoryDecisionAgent, json_output: bool = False):
    """Interactive CLI loop for free-form user inventory questions."""
    print(BANNER)
    print("Type your supply chain question in plain English.")
    print("Commands:")
    print("  'demo'      -> Run all 5 model scenarios")
    print("  'examples'  -> Show sample questions for each model")
    print("  'exit'/'q'  -> Exit the application\n")

    while True:
        try:
            user_input = input("\nInventory Agent > ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nExiting. Good luck with your supply chain decisions!")
            break

        if not user_input:
            continue
        if user_input.lower() in ["exit", "quit", "q"]:
            print("Exiting. Good luck with your supply chain decisions!")
            break
        if user_input.lower() == "demo":
            run_demo(agent, json_output=json_output)
            continue
        if user_input.lower() == "examples":
            print("\nSample Questions by Inventory Model:")
            for s in SAMPLE_QUESTIONS:
                print(f"\n[{s['model']}]:\n  \"{s['question']}\"")
            continue

        try:
            resp = agent.process_question(user_input)
            print_response(resp, json_output=json_output)
        except Exception as e:
            print(f"\n❌ Could not resolve question: {e}")
            print("Tip: Ensure key parameters (demand, costs, lead time, or probabilities) are mentioned.")


def main():
    parser = argparse.ArgumentParser(
        description="Supply Chain Analytics Inventory Decision Agent"
    )
    parser.add_argument(
        "--query",
        "-q",
        type=str,
        help="Single natural-language inventory question to evaluate.",
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Run predefined demonstrations for all 5 inventory models.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output structured JSON instead of formatted Markdown.",
    )

    args = parser.parse_args()
    agent = InventoryDecisionAgent()

    if args.demo:
        run_demo(agent, json_output=args.json)
    elif args.query:
        try:
            resp = agent.process_question(args.query)
            print_response(resp, json_output=args.json)
        except Exception as e:
            print(f"Error: {e}", file=sys.stderr)
            sys.exit(1)
    else:
        interactive_loop(agent, json_output=args.json)


if __name__ == "__main__":
    main()
