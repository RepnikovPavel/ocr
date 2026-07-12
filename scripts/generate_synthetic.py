import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        default="output/fixtures/composite_chart.png",
    )
    args = parser.parse_args()

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.titleweight": "bold",
            "axes.grid": True,
            "grid.alpha": 0.25,
        }
    )

    rng = np.random.default_rng(20260712)
    fig, axes = plt.subplots(2, 2, figsize=(12, 7.5), dpi=100)
    fig.suptitle("Quarterly Operations Dashboard", fontsize=20, fontweight="bold")

    regions = ["North", "South", "West", "Central"]
    actual = np.array([84, 67, 76, 91])
    plan = np.array([80, 72, 74, 88])
    positions = np.arange(len(regions))
    width = 0.36
    bars_actual = axes[0, 0].bar(
        positions - width / 2,
        actual,
        width,
        label="Actual",
        color="#276FBF",
    )
    axes[0, 0].bar(
        positions + width / 2,
        plan,
        width,
        label="Plan",
        color="#F4A259",
    )
    axes[0, 0].bar_label(bars_actual, padding=3, fmt="%d")
    axes[0, 0].set_title("Revenue by Region")
    axes[0, 0].set_ylabel("Revenue, USD millions")
    axes[0, 0].set_xticks(positions, regions)
    axes[0, 0].set_ylim(0, 110)
    axes[0, 0].legend(loc="upper left")

    months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun"]
    throughput = np.array([42, 47, 45, 53, 58, 64])
    quality = np.array([91, 92, 94, 93, 96, 97])
    quality_scaled = quality - 45
    axes[0, 1].plot(
        months,
        throughput,
        marker="o",
        linewidth=2.5,
        label="Throughput",
        color="#2A9D8F",
    )
    axes[0, 1].plot(
        months,
        quality_scaled,
        marker="s",
        linewidth=2.5,
        label="Quality index",
        color="#E76F51",
    )
    axes[0, 1].annotate(
        "Peak 64",
        xy=("Jun", 64),
        xytext=("Apr", 68),
        arrowprops={"arrowstyle": "->", "color": "#333333"},
    )
    axes[0, 1].set_title("Monthly Production Trend")
    axes[0, 1].set_ylabel("Normalized score")
    axes[0, 1].set_ylim(35, 72)
    axes[0, 1].legend(loc="upper left")

    efficiency = rng.uniform(62, 98, 18)
    cycle_time = 18.5 - 0.12 * efficiency + rng.normal(0, 0.65, 18)
    volume = rng.integers(80, 420, 18)
    scatter = axes[1, 0].scatter(
        efficiency,
        cycle_time,
        s=volume,
        c=volume,
        cmap="viridis",
        alpha=0.78,
        edgecolors="white",
        linewidths=0.8,
    )
    axes[1, 0].set_title("Efficiency vs. Cycle Time")
    axes[1, 0].set_xlabel("Efficiency, percent")
    axes[1, 0].set_ylabel("Cycle time, hours")
    colorbar = fig.colorbar(scatter, ax=axes[1, 0], pad=0.02)
    colorbar.set_label("Batch volume")

    categories = ["Materials", "Labor", "Logistics", "Energy"]
    shares = [38, 31, 19, 12]
    colors = ["#264653", "#2A9D8F", "#E9C46A", "#E76F51"]
    _, _, values = axes[1, 1].pie(
        shares,
        labels=categories,
        colors=colors,
        autopct="%1.0f%%",
        startangle=90,
        pctdistance=0.72,
        wedgeprops={"width": 0.43, "edgecolor": "white"},
    )
    for value in values:
        value.set_fontweight("bold")
    axes[1, 1].set_title("Operating Cost Structure")

    fig.text(
        0.5,
        0.01,
        "Reporting period: Q2 2026 | Currency: USD | Status: Final",
        ha="center",
        fontsize=10,
    )
    fig.tight_layout(rect=(0, 0.04, 1, 0.95))
    fig.savefig(
        output,
        dpi=100,
        facecolor="white",
        metadata={"Software": "matplotlib"},
    )
    plt.close(fig)
    print(output)


if __name__ == "__main__":
    main()
