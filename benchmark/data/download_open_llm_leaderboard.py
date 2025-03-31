import os
import json
from datasets import load_dataset
from pathlib import Path


def download_and_process_leaderboard():
    # Create leaderboard directory in the same folder as this script
    script_dir = Path(__file__).parent
    leaderboard_dir = script_dir / "leaderboard"
    leaderboard_dir.mkdir(exist_ok=True)

    # Download the dataset
    dataset = load_dataset("open-llm-leaderboard/contents")

    # Convert to list of records (each record is a dictionary with all fields)
    data = dataset["train"].to_list()

    # Save full dataset
    with open(leaderboard_dir / "open_llm_leaderboard.json", "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    # Extract specific fields
    cost_data = [{"eval_name": item["eval_name"], "CO2_cost_kg": item["CO₂ cost (kg)"]} for item in data]

    # Save extracted data
    with open(leaderboard_dir / "llm_cost_CO2.json", "w", encoding="utf-8") as f:
        json.dump(cost_data, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    download_and_process_leaderboard()
