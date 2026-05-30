import os
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.lines import Line2D


def ensure_output_dir(path: str) -> None:
    """Ensure that the output directory exists."""
    os.makedirs(path, exist_ok=True)


def plot_weather_heatmap(df: pd.DataFrame, outdir: str) -> str:
    """Create a heatmap of average temperature by city and month."""
    monthly = (
        df.groupby(["city", "month"], as_index=False)["avg_temp"]
        .mean()
        .sort_values(["city", "month"])
    )
    heatmap_data = monthly.pivot(index="city", columns="month", values="avg_temp")
    heatmap_data = heatmap_data.reindex(sorted(heatmap_data.columns), axis=1)

    fig, ax = plt.subplots(figsize=(10, 4))
    sns.heatmap(
        heatmap_data,
        cmap="coolwarm",
        annot=True,
        fmt=".1f",
        cbar_kws={"label": "Average temperature"},
        ax=ax,
    )
    ax.set_title("Average monthly temperature by city")
    ax.set_xlabel("Month")
    ax.set_ylabel("City")

    outpath = os.path.join(outdir, "weather_heatmap.png")
    fig.savefig(outpath, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return outpath


def plot_weather_scatter(df: pd.DataFrame, outdir: str) -> str:
    """Create a scatter plot of humidity, temperature, and precipitation."""
    plot_df = df.copy()
    plot_df["precip"] = pd.to_numeric(plot_df["precip"], errors="coerce").fillna(0.0)

    fig, ax = plt.subplots(figsize=(9, 6))
    size_range = (20, 300)
    cities = sorted(plot_df["city"].dropna().unique())
    palette = dict(zip(cities, sns.color_palette("tab10", n_colors=len(cities))))

    sns.scatterplot(
        data=plot_df,
        x="avg_humidity",
        y="avg_temp",
        hue="city",
        size="precip",
        sizes=size_range,
        palette=palette,
        alpha=0.65,
        legend=False,
        ax=ax,
    )

    city_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            label=city,
            markerfacecolor=color,
            markeredgecolor=color,
            markersize=8,
        )
        for city, color in palette.items()
    ]
    city_legend = ax.legend(
        handles=city_handles,
        title="City",
        loc="upper left",
        bbox_to_anchor=(1.02, 1.0),
        borderaxespad=0,
    )
    ax.add_artist(city_legend)

    max_precip = float(plot_df["precip"].max())
    if max_precip > 0:
        precip_values = np.linspace(0, max_precip, 4)
        precip_sizes = np.interp(precip_values, [0, max_precip], size_range)
    else:
        precip_values = np.array([0.0])
        precip_sizes = np.array([size_range[0]])

    precip_handles = [
        plt.scatter([], [], s=size, color="gray", alpha=0.65, label=f"{value:.2g}")
        for value, size in zip(precip_values, precip_sizes)
    ]
    ax.legend(
        handles=precip_handles,
        title="Precipitation",
        loc="lower left",
        bbox_to_anchor=(1.02, 0.0),
        borderaxespad=0,
    )

    ax.set_title("Daily weather: temperature vs humidity with precipitation (size)")
    ax.set_xlabel("Average relative humidity (%)")
    ax.set_ylabel("Average temperature (F)")

    outpath = os.path.join(outdir, "weather_scatter.png")
    fig.savefig(outpath, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return outpath


def plot_global_temp_heatmap(df: pd.DataFrame, outdir: str) -> str:
    """Create a heatmap of global temperature anomalies by year and month."""
    months = [
        "Jan",
        "Feb",
        "Mar",
        "Apr",
        "May",
        "Jun",
        "Jul",
        "Aug",
        "Sep",
        "Oct",
        "Nov",
        "Dec",
    ]
    month_numbers = {month: idx for idx, month in enumerate(months, start=1)}

    long_df = df.melt(
        id_vars="Year",
        value_vars=months,
        var_name="Month",
        value_name="Anomaly",
    )
    long_df["MonthNum"] = long_df["Month"].map(month_numbers)
    heatmap_data = long_df.pivot(index="Year", columns="MonthNum", values="Anomaly")
    heatmap_data = heatmap_data.sort_index()

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        heatmap_data,
        cmap="coolwarm",
        vmin=-1.5,
        vmax=1.5,
        cbar_kws={"label": "Temperature anomaly (C relative to 1951-1980)"},
        linewidths=0,
        linecolor="white",
        ax=ax,
    )
    ax.set_xticklabels(months, rotation=45, ha="right")
    ax.set_title("Global land-ocean temperature anomalies (1880-2025)")
    ax.set_xlabel("Month")
    ax.set_ylabel("Year")

    outpath = os.path.join(outdir, "global_temp_heatmap.png")
    fig.savefig(outpath, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return outpath


def plot_minnesota_precip_line(df: pd.DataFrame, outdir: str) -> str:
    """Create a line chart of monthly precipitation by Minnesota site."""
    plot_df = df.copy()
    plot_df["date"] = pd.to_datetime(
        {"year": plot_df["year"], "month": plot_df["mo"], "day": 1}
    )

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.lineplot(data=plot_df, x="date", y="precip", hue="site", ax=ax)
    ax.set_title("Monthly precipitation by Minnesota site (1927-1936)")
    ax.set_xlabel("Year")
    ax.set_ylabel("Precipitation (inches)")
    ax.legend(
        bbox_to_anchor=(1.05, 1),
        loc="upper left",
        title="Site",
        borderaxespad=0,
    )

    outpath = os.path.join(outdir, "minnesota_precip_line.png")
    fig.savefig(outpath, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return outpath


def main() -> List[str]:
    """Run all visualizations and return a list of generated file paths."""
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(base_dir, "data")
    out_dir = os.path.join(base_dir, "output")
    ensure_output_dir(out_dir)
    figures: List[str] = []

    weather_path = os.path.join(data_dir, "weather_data.csv")
    weather_df = pd.read_csv(weather_path)
    figures.append(plot_weather_heatmap(weather_df, out_dir))
    figures.append(plot_weather_scatter(weather_df, out_dir))

    global_path = os.path.join(data_dir, "global_temp.csv")
    global_df = pd.read_csv(global_path, skiprows=1)
    global_df = global_df.replace("***", pd.NA)
    for col in global_df.columns[1:]:
        global_df[col] = pd.to_numeric(global_df[col], errors="coerce")
    figures.append(plot_global_temp_heatmap(global_df, out_dir))

    minn_path = os.path.join(data_dir, "minnesota_weather.csv")
    minn_df = pd.read_csv(minn_path)
    figures.append(plot_minnesota_precip_line(minn_df, out_dir))
    return figures


if __name__ == "__main__":
    generated = main()
    print("Generated figures:")
    for path in generated:
        print(os.path.join("output", os.path.basename(path)))
