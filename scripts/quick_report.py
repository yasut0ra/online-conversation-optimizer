from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from statistics import mean


def load_latest_file(log_dir: Path) -> list[dict]:
    files = sorted(log_dir.glob("turns-*.jsonl"))
    if not files:
        raise FileNotFoundError("No turn logs found in logs/ directory")
    latest = files[-1]
    records = []
    with latest.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return records


def aggregate(records: list[dict]) -> dict:
    by_turn: dict[tuple[str, str], dict] = {}
    for record in records:
        key = (record.get("session_id"), record.get("turn_id"))
        by_turn[key] = record

    final_records = list(by_turn.values())
    total = len(final_records)
    if total == 0:
        raise ValueError("No turn records available")

    rewards = [r["reward"] for r in final_records if r.get("reward") is not None]
    avg_reward = mean(rewards) if rewards else None

    propensity_values = [r["propensity"] for r in final_records if r.get("propensity") is not None]
    propensity_stats = (
        {
            "mean": mean(propensity_values),
            "min": min(propensity_values),
            "max": max(propensity_values),
        }
        if propensity_values
        else None
    )

    style_counter: Counter[str] = Counter()
    for record in final_records:
        candidates = record.get("candidates") or []
        chosen_idx = record.get("chosen_idx")
        if not isinstance(candidates, list) or chosen_idx is None:
            continue
        if 0 <= chosen_idx < len(candidates):
            style = candidates[chosen_idx].get("style", "unknown")
            style_counter[style] += 1

    style_win_rates = (
        {style: count / total for style, count in style_counter.items()}
        if style_counter
        else {}
    )

    unique_choices = len({record.get("chosen_idx") for record in final_records if record.get("chosen_idx") is not None})
    exploration_rate = unique_choices / total

    return {
        "turns": total,
        "avg_reward": avg_reward,
        "style_win_rates": style_win_rates,
        "exploration_rate": exploration_rate,
        "propensity_stats": propensity_stats,
    }


def main() -> None:
    log_dir = Path(__file__).resolve().parent.parent / "logs"
    try:
        records = load_latest_file(log_dir)
    except FileNotFoundError as exc:
        print(exc)
        return
    try:
        summary = aggregate(records)
    except ValueError as exc:
        print(exc)
        return

    print("Turn count:", summary["turns"])
    if summary["avg_reward"] is not None:
        print(f"Average reward: {summary['avg_reward']:.3f}")
    else:
        print("Average reward: n/a (no feedback yet)")

    if summary["style_win_rates"]:
        print("Style win rates:")
        for style, rate in sorted(summary["style_win_rates"].items(), key=lambda x: x[1], reverse=True):
            print(f"  {style}: {rate:.2%}")
    else:
        print("Style win rates: n/a")

    print(f"Exploration rate: {summary['exploration_rate']:.2%}")

    stats = summary["propensity_stats"]
    if stats:
        print(
            "Propensity stats: mean={mean:.3f}, min={min:.3f}, max={max:.3f}".format(
                **stats
            )
        )
    else:
        print("Propensity stats: n/a")


if __name__ == "__main__":
    main()

