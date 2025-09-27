from __future__ import annotations


def main() -> None:
    from pathlib import Path

    from src.metrics import compute_metrics

    log_dir = Path(__file__).resolve().parent.parent / "logs"
    summary = compute_metrics(log_dir)

    turn_count = summary["turn_count"]
    print("Turn count:", turn_count)
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

    mean_value = summary["propensity_mean"]
    std_value = summary["propensity_std"]
    if mean_value is not None:
        std_display = f", std={std_value:.3f}" if std_value is not None else ""
        print(f"Propensity stats: mean={mean_value:.3f}{std_display}")
    else:
        print("Propensity stats: n/a")


if __name__ == "__main__":
    main()

