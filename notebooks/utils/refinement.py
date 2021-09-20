import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def plot_refinement_improvement(accuracy_df: pd.DataFrame, refinement_df: pd.DataFrame, reference_refinement_type: str="original"):
    grouped_df = (
        pd.merge(refinement_df, accuracy_df, left_on="info_run_id", right_on="run_id")
        .groupby(
            [
                "refinement_type", 
                "refinement_run",
                "info_run_id",
                "type",
                "percentile",
            ],
            as_index=False,
        )
        .agg({"accuracy": max,})
    )
    mean_grouped_df = (
        grouped_df.groupby(
            [
                "refinement_run",
                "refinement_type",
                "type",
                "percentile",
            ],
            as_index=True,
        )
        .agg({"accuracy": np.mean,})
        .reset_index(drop=False)
        .pivot(
            index=["type", "percentile"],
            columns=["refinement_run", "refinement_type"],
            values="accuracy",
        )
    )

    knowledge_types = [x[0] for x in mean_grouped_df.columns]
    for knowledge_type in set(knowledge_types):
        mean_grouped_df[knowledge_type] = mean_grouped_df.loc[:, (knowledge_type)].sub(
            mean_grouped_df[(knowledge_type, reference_refinement_type)], axis=0
        )
    groupeds = []
    for knowledge_type in set(knowledge_types):
        gdf = mean_grouped_df[knowledge_type].copy()
        gdf["knowledge"] = knowledge_type

        groupeds.append(gdf)

    mean_grouped_df = pd.concat(groupeds).reset_index().melt(
        id_vars=["type", "percentile", "knowledge"],
        value_vars=list(set(refinement_df["refinement_type"])),
        value_name="mean_accuracy_diff"
    )
    mean_grouped_df = pd.merge(
        mean_grouped_df, refinement_df, how="left", 
        left_on="knowledge", right_on="refinement_run", suffixes=("", "_other"))

    g = sns.relplot(
        data=mean_grouped_df[mean_grouped_df["percentile"] > -1],
        x="percentile",
        y="mean_accuracy_diff",
        row="type",
        hue="refinement_type",
        col="data_params_RefinementConfigoriginal_file_knowledge",
        kind="line",
        palette=None,
        facet_kws={"sharey": False, "sharex": True},
    )
    g.set_titles("Type: {row_name}, Knowledge: {col_name}").set_axis_labels(
        "", "mean_accuracy_diff"
    )
    g.map(plt.axhline, y=0, color=".7", dashes=(2, 1), zorder=0)
    for ax in g.axes.flatten():
        ax.tick_params(labelbottom=True)
    plt.show()