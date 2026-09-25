"""
team_rankings.py
    produce dataframes for top 10 and bottom 10 teams by:
        season (2021-2025)
        model (srs, gc, awp)
"""

import os
from ratings_models import SRSModel, GameControlModel, AWPModel

# Make datasets for best/worst teams by season for each model
def export_top_bottom(ratings_by_season, value_col, out_dir, model_label, n=10):

    os.makedirs(out_dir, exist_ok=True)

    # drop intercept/HFA row before ranking teams
    clean = ratings_by_season[~ratings_by_season['team'].isin(['(Intercept)', 'const'])]

    for season, sdf in clean.groupby('season'):
        sdf_sorted = sdf.sort_values(value_col, ascending=False)

        top10 = sdf_sorted.head(n)
        bottom10 = sdf_sorted.tail(n).sort_values(value_col, ascending=True)

        top_path = os.path.join(out_dir, f"{model_label}_top10_{season}.csv")
        bottom_path = os.path.join(out_dir, f"{model_label}_bottom10_{season}.csv")

        top10.to_csv(top_path, index=False)
        bottom10.to_csv(bottom_path, index=False)

        print(f"[{model_label}] season {season}: wrote {top_path} and {bottom_path}", flush=True)


def export_all_results(results_root="results"):
    export_top_bottom(
        SRSModel.ratings_by_season,
        value_col="SRS",
        out_dir=os.path.join(results_root, "srs_results"),
        model_label="srs",
    )
    export_top_bottom(
        GameControlModel.ratings_by_season,
        value_col="Adj_GC",
        out_dir=os.path.join(results_root, "gc_results"),
        model_label="gc",
    )
    export_top_bottom(
        AWPModel.ratings_by_season,
        value_col="Adj_AWP_prob",
        out_dir=os.path.join(results_root, "awp_results"),
        model_label="awp",
    )


if __name__ == "__main__":
    export_all_results()
