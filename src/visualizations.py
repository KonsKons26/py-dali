import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import re
from scipy import stats

def create_protein_comparison_plot(csv_file_path, figsize=(14, 8)):
    """
    Create a plot comparing protein structures based on RMSD and MCA scores.

    Parameters
    ----------
    csv_file_path : str
        Path to the CSV file containing protein data.
    figsize : tuple, optional
        Size of the figure to create. Default is (14, 8).

    Returns
    -------
    matplotlib.figure.Figure
        The created figure object containing the plot.
    """

    # Set seaborn style for prettier plots
    sns.set_style("whitegrid")
    sns.set_context("notebook", font_scale=1.2)

    # Read the CSV file
    df = pd.read_csv(csv_file_path)

    # Extract RMSD values from the tuple strings
    def extract_rmsd(rmsd_str):
        match = re.search(r"np\.float32\(([\d.e-]+)\)", rmsd_str)
        if match:
            return float(match.group(1))
        else:
            numbers = re.findall(r"[\d.e-]+", rmsd_str)
            return float(numbers[0]) if numbers else np.nan

    df["rmsd_value"] = df["rmsd"].apply(extract_rmsd)

    # Get query (first row) values
    query = df.iloc[0]
    query_class = query["class"]
    query_fold = query["fold"]
    query_superfamily = query["superfamily"]
    query_family = query["family"]
    query_mca_score = query["mca_score"]

    # Calculate relative MCA scores
    df["relative_mca_score"] = df["mca_score"] / query_mca_score
    df["rmsd_plot"] = df["rmsd_value"]

    # Classify entries into regions
    def classify_entry(row):
        if (row["class"] == query_class and 
            row["fold"] == query_fold and 
            row["superfamily"] == query_superfamily and 
            row["family"] == query_family):
            return "Same Family"
        elif (row["class"] == query_class and 
              row["fold"] == query_fold and 
              row["superfamily"] == query_superfamily):
            return "Same Superfamily"
        elif (row["class"] == query_class and 
              row["fold"] == query_fold):
            return "Same Fold"
        elif row["class"] == query_class:
            return "Same Class"
        else:
            return "Different Class"

    df["region"] = df.apply(classify_entry, axis=1)

    # Define region order and calculate statistics
    region_order = [
        "Same Family", "Same Superfamily", "Same Fold",
        "Same Class", "Different Class"
    ]

    # Calculate summary statistics for each region
    summary_stats = []
    for i, region in enumerate(region_order):
        region_data = df[df["region"] == region]
        if len(region_data) > 0:
            # MCA statistics
            mca_mean = region_data["relative_mca_score"].mean()
            mca_std = region_data["relative_mca_score"].std()
            mca_sem = mca_std / np.sqrt(len(region_data)) if len(region_data) > 1 else 0
            mca_ci = 1.96 * mca_sem  # 95% confidence interval

            # RMSD statistics
            rmsd_mean = region_data["rmsd_plot"].mean()
            rmsd_std = region_data["rmsd_plot"].std()
            rmsd_sem = rmsd_std / np.sqrt(len(region_data)) if len(region_data) > 1 else 0
            rmsd_ci = 1.96 * rmsd_sem  # 95% confidence interval

            summary_stats.append({
                "region": region,
                "x_pos": i,
                "n_entries": len(region_data),
                "mca_mean": mca_mean,
                "mca_ci": mca_ci,
                "rmsd_mean": rmsd_mean,
                "rmsd_ci": rmsd_ci
            })

    stats_df = pd.DataFrame(summary_stats)

    fig, ax1 = plt.subplots(figsize=figsize)
    ax2 = ax1.twinx()

    mca_color = "#2E86AB"
    rmsd_color = "#A23B72"

    # Plot MCA line with confidence intervals
    ax1.plot(
        stats_df["x_pos"], stats_df["mca_mean"], 
        color=mca_color, linewidth=3, marker="o", markersize=10,
        markerfacecolor=mca_color, markeredgecolor="white", 
        markeredgewidth=2, label="Relative MCA Score", zorder=3
    )

    # Add confidence intervals for MCA
    ax1.fill_between(
        stats_df["x_pos"], 
        stats_df["mca_mean"] - stats_df["mca_ci"],
        stats_df["mca_mean"] + stats_df["mca_ci"],
        alpha=0.3, color=mca_color, label="95% CI (MCA)"
    )

    # Plot RMSD line with confidence intervals
    ax2.plot(
        stats_df["x_pos"], stats_df["rmsd_mean"], 
        color=rmsd_color, linewidth=3, marker="^", markersize=10,
        markerfacecolor=rmsd_color, markeredgecolor="white", 
        markeredgewidth=2, label="RMSD", zorder=3
    )

    # Add confidence intervals for RMSD
    ax2.fill_between(
        stats_df["x_pos"], 
        stats_df["rmsd_mean"] - stats_df["rmsd_ci"],
        stats_df["rmsd_mean"] + stats_df["rmsd_ci"],
        alpha=0.2, color=rmsd_color, label="95% CI (RMSD)"
        )

    # Customize axes
    ax1.set_xlabel(
        "Structural Similarity Level",
        fontsize=14, fontweight="bold"
    )
    ax1.set_ylabel(
        "Relative MCA Score (ratio to query)",
        fontsize=14, fontweight="bold", color=mca_color
    )
    ax2.set_ylabel(
        "Sliding window RMSD (Å)",
        fontsize=14, fontweight="bold", color=rmsd_color
    )

    # Color the y-axis labels
    ax1.tick_params(axis="y", labelcolor=mca_color, labelsize=12)
    ax2.tick_params(axis="y", labelcolor=rmsd_color, labelsize=12)
    ax1.tick_params(axis="x", labelsize=11)

    # Set x-axis labels
    region_labels = [
        f"{row['region']}\n(n={row["n_entries"]})"
        for _, row in stats_df.iterrows()
    ]
    ax1.set_xticks(stats_df["x_pos"])
    ax1.set_xticklabels(region_labels, fontsize=11, ha="center")

    # Add horizontal reference line at y=1 for relative MCA score
    ax1.axhline(y=1.0, color=mca_color, linestyle="--", alpha=0.6, linewidth=2)

    # Add horizontal reference line at y=0 for RMSD
    ax2.axhline(y=0, color=rmsd_color, linestyle="--", alpha=0.6, linewidth=2)

    # Set axis limits with some padding
    ax1.set_xlim(-0.3, len(stats_df) - 0.7)

    # Add beautiful title
    plt.suptitle("Protein Structure Similarity Analysis", 
                fontsize=18, fontweight="bold", y=0.98)
    ax1.set_title(
        f"Query: {query['domain_id']} | Class: {query_class} | Fold: {query_fold} | Superfamily: {query_superfamily} | Family: {query_family}", 
        fontsize=12, pad=20, color="#333333"
        )

    # Create combined legend - fixed version
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()

    # Filter out the query protein from labels and lines
    filtered_lines1 = []
    filtered_labels1 = []
    for line, label in zip(lines1, labels1):
        if "Query Protein" not in label:
            filtered_lines1.append(line)
            filtered_labels1.append(label)

    # Include all RMSD lines and labels (both line and CI)
    all_lines = filtered_lines1 + lines2
    all_labels = filtered_labels1 + labels2

    legend = ax1.legend(
        all_lines, all_labels, loc="center left",
        frameon=True, fancybox=False, shadow=False,
        fontsize=11, bbox_to_anchor=(1.05, 0.1)
    )
    legend.get_frame().set_facecolor("white")
    legend.get_frame().set_alpha(0.95)

    # Improve grid
    ax1.grid(True, alpha=0.3)
    ax2.grid(False)

    # Adjust layout
    plt.tight_layout()

    return fig