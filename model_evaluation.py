"""
model_evaluation.py
    Predictive out-of-sample and repeatability testing for the rating models
"""

import numpy as np
import pandas as pd
import os
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score, log_loss, brier_score_loss
from scipy.stats import pearsonr
from scipy import stats

from ratings_models import seasons, SRSModel, GameControlModel, AWPModel


print("--- Model Evaluation ---")

print("Preparing datasets...", flush=True)


# build one dataset per model
srs_df = SRSModel.df.copy()
srs_df['home_win'] = (srs_df['home_points'] > srs_df['away_points']).astype(int)

games_meta = srs_df[['game_id', 'week', 'home_win']]

gc_df = GameControlModel.game_control_by_game.merge(games_meta, on='game_id', how='left')
awp_df = AWPModel.awp_by_game.merge(games_meta, on='game_id', how='left')


# ---------------------------------------------------------------------
# OUT-OF-SAMPLE PERFORMANCE
# ---------------------------------------------------------------------

def evaluate_oos(df, fit_func, value_col, home_col, away_col,
                                       week_col='week', outcome_col='home_win', train_frac=0.7):

    results = []

    # chronological split (early season train, late season test)
    for season, sdf in df.groupby('season'):
        cutoff = sdf[week_col].quantile(train_frac)
        train = sdf[sdf[week_col] <= cutoff]
        test = sdf[sdf[week_col] > cutoff]

        # shouldn't catch, but in case a half is left empty
        if train.empty or test.empty or test[outcome_col].nunique() < 2:
            continue

        # fit ratings on train games for that season
        ratings_df = fit_func(train)
        ratings = ratings_df.set_index('team')[value_col].to_dict()

        # pull home-field advantage term
        hfa = ratings.pop('const', None)
        if hfa is None:
            hfa = ratings.pop('(Intercept)', 0.0)

        def rating_diff(row):
            # predicted rating gap for a game: HFA + home rating - away rating
            return hfa + ratings.get(row[home_col], 0.0) - ratings.get(row[away_col], 0.0)

        X_train = train.apply(rating_diff, axis=1).values.reshape(-1, 1)
        X_test = test.apply(rating_diff, axis=1).values.reshape(-1, 1)
        y_train = train[outcome_col].values
        y_test = test[outcome_col].values

        # calibrate rating diff -> win prob
        model = LogisticRegression().fit(X_train, y_train)

        # calculate values and append
        pred_probs = model.predict_proba(X_test)[:, 1]
        pred_class = (pred_probs >= 0.5).astype(int)

        metrics_df = pd.DataFrame({
            "season": [season],
            "accuracy": [accuracy_score(y_test, pred_class)],
            "roc_auc": [roc_auc_score(y_test, pred_probs)],
            "log_loss": [log_loss(y_test, pred_probs, labels=[0, 1])],
            "brier score": [brier_score_loss(y_test, pred_probs)]
        })
        results.append(metrics_df)

    return pd.concat(results, ignore_index=True)


# -------------
# REPEATABILITY
# -------------

# for each season split games at median week and check correlation
def evaluate_repeatability(df, fit_func, value_col, week_col='week'):
    results = []
    for season, sdf in df.groupby('season'):
        median_week = sdf[week_col].median()
        h1 = sdf[sdf[week_col] <= median_week]
        h2 = sdf[sdf[week_col] > median_week]

        # fit each half, pull out team: rating and drop the intercept row
        ratings_h1 = fit_func(h1).set_index('team')[value_col].to_dict()
        ratings_h1.pop('const', None)
        ratings_h1.pop('(Intercept)', None)

        ratings_h2 = fit_func(h2).set_index('team')[value_col].to_dict()
        ratings_h2.pop('const', None)
        ratings_h2.pop('(Intercept)', None)

        # drop teams not in both halves
        wide = pd.DataFrame({
            'H1': pd.Series(ratings_h1),
            'H2': pd.Series(ratings_h2)
        }).dropna()

        # get values and append
        r, p_val = stats.pearsonr(wide['H1'], wide['H2'])

        results.append({
            'season': season,
            'n_teams': len(wide),
            'r': r,
            'p_value': p_val,
            'r_squared': r ** 2
        })

    return pd.DataFrame(results)


# ----------------
# CREATE DATASETS
# ----------------

RESULTS_DIR = "results"

def export_performance_tables(pred_all, rep_all, results_dir=RESULTS_DIR):
    os.makedirs(results_dir, exist_ok=True)

    # per-season detail
    pred_all.to_csv(os.path.join(results_dir, "oos_performance_by_season.csv"), index=False)
    rep_all.to_csv(os.path.join(results_dir, "repeatability_by_season.csv"), index=False)

    # across season averages
    oos_summary = (
        pred_all.groupby('model')[['accuracy', 'roc_auc', 'log_loss', 'brier score']]
        .mean()
        .sort_values('accuracy', ascending=False)
        .reset_index()
    )
    rep_summary = (
        rep_all.groupby('model')[['r', 'r_squared', 'p_value']]
        .mean()
        .sort_values('r', ascending=False)
        .reset_index()
    )

    oos_summary.to_csv(os.path.join(results_dir, "oos_performance_summary.csv"), index=False)
    rep_summary.to_csv(os.path.join(results_dir, "repeatability_summary.csv"), index=False)

    print(f"\nWrote performance tables to {results_dir}/:")
    print("  oos_performance_by_season.csv")
    print("  oos_performance_summary.csv")
    print("  repeatability_by_season.csv")
    print("  repeatability_summary.csv")

    return oos_summary, rep_summary

# evaluate models and send results to results folder
if __name__ == "__main__":
    print("Building Model Specifications")

    MODELS = {
        "SRS": dict(df=srs_df, fit_func=SRSModel.calc_srs_scratch, value_col='SRS',
                    home_col='home_team', away_col='away_team'),
        "Adj_GC": dict(df=gc_df, fit_func=GameControlModel.calc_adj_gc_scratch, value_col='Adj_GC',
                       home_col='homeTeamName', away_col='awayTeamName'),
        "Adj_AWP": dict(df=awp_df, fit_func=AWPModel.calc_adj_awp_scratch, value_col='Adj_AWP_logit',
                        home_col='homeTeamName', away_col='awayTeamName'),
    }

    pred_all, rep_all = [], []
    for name, spec in MODELS.items():

        p = evaluate_oos(spec['df'], spec['fit_func'], spec['value_col'], spec['home_col'], spec['away_col'])
        p['model'] = name
        pred_all.append(p)

        r = evaluate_repeatability(spec['df'], spec['fit_func'], spec['value_col'])
        r['model'] = name
        rep_all.append(r)

    pred_all = pd.concat(pred_all, ignore_index=True)
    rep_all = pd.concat(rep_all, ignore_index=True)

    oos_summary, rep_summary = export_performance_tables(pred_all, rep_all)

    print("Out-Of-Sample Performance (avg across seasons):")
    print(oos_summary)
    print("\nRepeatability (avg across seasons):")
    print(rep_summary)