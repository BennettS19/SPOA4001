"""
ratings_models.py
    SRS
    GameControl
    Avg Win Prob
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
import sportsdataverse as sdv
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from scipy.special import logit, expit


seasons = [2021, 2022, 2023, 2024, 2025]

# predictors for AWP model
AWP_PREDICTORS = [
    "start.adj_TimeSecsRem",
    "start.homeScore",
    "start.awayScore",
    "start.homeTeamTimeouts",
    "start.awayTeamTimeouts",
    "under_2",
    "down",
    "distance",
    "start.yardsToEndzone",
    "start.is_home",
    "gameSpread",
]


# ------------
# LOAD DATA
# ------------

def load_data_srs(seasons):

    print("[SRS] Loading schedule...", flush=True)

    games_df_polars = sdv.cfb.load_cfb_schedule(seasons=seasons, return_as_pandas=False)

    games_df = games_df_polars.to_pandas(use_pyarrow_extension_array=False)

    # only fbs v fbs games (regular season)
    fbs_srs = games_df[
        (games_df['home_division'] == 'fbs') &
        (games_df['away_division'] == 'fbs') &
        (games_df['season_type'] == 'regular')
    ].copy()

    fbs_srs["score_diff"] = (fbs_srs["home_points"] - fbs_srs["away_points"])
    fbs_srs = fbs_srs.dropna(subset=["score_diff"]).copy()

    print(f"[SRS] {len(fbs_srs):,} games loaded.", flush=True)

    return fbs_srs


def load_data(seasons):
    # PBP data merged with schedule info
    # Used for Game Control and AWP models

    print("[PBP] Loading PBP and schedule...", flush=True)

    plays_df_polars = sdv.cfb.load_cfb_pbp(seasons=seasons, return_as_pandas=False)
    games_df_polars = sdv.cfb.load_cfb_schedule(seasons=seasons, return_as_pandas=False)

    pbp_raw = plays_df_polars.to_pandas(use_pyarrow_extension_array=False)
    games_df = games_df_polars.to_pandas(use_pyarrow_extension_array=False)

    # Pull only game level fields needed
    games_meta = games_df[
        ['game_id', 'season_type', 'home_division', 'away_division', 'home_points', 'away_points', 'home_team', 'away_team']
    ]

    final_pbp = pd.merge(pbp_raw, games_meta, on='game_id', how='left')

    final_pbp['home_win'] = (final_pbp['home_points'] > final_pbp['away_points']).astype(int)

    # filter fbs v fbs regular season again (no OT plays)
    fbs_pbp = final_pbp[
        (final_pbp['home_division'] == 'fbs') &
        (final_pbp['away_division'] == 'fbs') &
        (final_pbp['season_type'] == 'regular') &
        (final_pbp['drive.start.period.number'] <= 4)
    ].copy()

    # drop rows missing important features
    fbs_pbp = fbs_pbp.dropna(subset=['home_win'] + AWP_PREDICTORS)

    # compute play length
    prev_seconds = (fbs_pbp.groupby('game_id')['start.adj_TimeSecsRem'].shift(1).fillna(3600))

    fbs_pbp['play_duration'] = (prev_seconds - fbs_pbp['start.adj_TimeSecsRem'])
    fbs_pbp['play_duration'] = np.maximum(0, fbs_pbp['play_duration']) # make sure no negative durations

    print(f"[PBP] Finished.", flush=True)

    return fbs_pbp


# ---------------------------------------------
# BUILD FUNCTIONS FOR WIN PROB PREDICTION (AWP)
# ---------------------------------------------

def train_wp_model(game_df, predictors=None):
    # fit logistic regression model later applied for AWP
    if predictors is None:
        predictors = AWP_PREDICTORS

    print("[WP] Training model...", flush=True)

    X = game_df[predictors]
    y = game_df["home_win"]

    # stratify by season
    strata = game_df["season"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=925,
        stratify=strata
    )

    # standardize preds before fitting
    scaler = StandardScaler()

    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train),
        columns=predictors,
        index=X_train.index
    )

    X_train_sm = sm.add_constant(X_train_scaled)

    model = sm.Logit(y_train, X_train_sm).fit(disp=False)

    print("[WP] Model trained.", flush=True)

    return model, scaler


def get_game_wp(game_id, model, scaler, predictors, df):
    # get wp for every play of every game and add value to dataset

    g = df[df['game_id'] == game_id].copy()

    # ensure sorted chronologically
    g = g.sort_values('start.adj_TimeSecsRem', ascending=False)

    x_game = g[predictors]
    x_game_scaled = pd.DataFrame(scaler.transform(x_game), columns=predictors, index=x_game.index)

    x_game_scaled = sm.add_constant(x_game_scaled, has_constant='add')
    g['home_wp'] = model.predict(x_game_scaled)

    return g


# ---------------
# SRS MODEL
# ---------------

class SRSModel():

    print("[SRS] Initializing...", flush=True)

    df = load_data_srs(seasons)

    def calc_srs_scratch(df):
        teams = np.unique(np.concatenate([df['home_team'].unique(), df['away_team'].unique()]))
        n_games = len(df)

        # build design matrix (+1 home, -1 away, 0 else)
        X = pd.DataFrame(0.0, index=range(n_games), columns=teams)

        for i, (_, row) in enumerate(df.iterrows()):
            X.loc[i, row['home_team']] = 1.0
            X.loc[i, row['away_team']] = -1.0

        margin = (df['home_points'] - df['away_points'])
        X = sm.add_constant(X)
        fit = sm.OLS(margin.values, X).fit()
        srs_vec = fit.params.fillna(0)
        # center ratings around 0
        srs_centered = (srs_vec - srs_vec.mean())

        return pd.DataFrame({
            'team': srs_centered.index,
            'SRS': np.round(srs_centered.values, 2)
        })

    # fit ratings seperately for each season
    ratings_list = []

    for yr, season_df in df.groupby('season'):
        r = calc_srs_scratch(season_df)
        r['season'] = yr
        ratings_list.append(r)

    ratings_by_season = pd.concat(ratings_list, ignore_index=True)


# --------------------
# GAME CONTROL MODEL
# --------------------

class GameControlModel():

    print("[GC] Initializing...", flush=True)

    df_raw = load_data(seasons)

    # exclude garbage time plays
    df = df_raw[
        (df_raw['wp_before'] >= 0.05) &
        (df_raw['wp_before'] <= 0.95)
    ].copy()

    def calc_game_gc(group):
        # time weighted avg score differential for one game
        total_time = group['play_duration'].sum()
        denom = (total_time if total_time > 0 else 1.0) # dont divide by zero

        home_gc = ((group['HA_score_diff'] * group['play_duration']).sum() / denom)

        return pd.Series({
            'total_time': total_time,
            'home_gc': home_gc,
            'away_gc': -home_gc
        })

    game_control_by_game = (df.groupby(['season','game_id','homeTeamName','awayTeamName'], group_keys=False)
                            .apply(calc_game_gc).reset_index())

    def calc_adj_gc_scratch(df):
        # adjust for opponent
        teams = np.unique(np.concatenate([df['homeTeamName'].unique(), df['awayTeamName'].unique()]))
        n_games = len(df)
        season = df['season'].iloc[0]

        X = pd.DataFrame(0.0, index=range(n_games), columns=teams)

        for i, (_, row) in enumerate(df.iterrows()):
            X.loc[i, row['homeTeamName']] = 1.0
            X.loc[i, row['awayTeamName']] = -1.0

        margin = df['home_gc'].values

        X_design = sm.add_constant(X)
        fit = sm.OLS(margin, X_design).fit()

        srs_vec = fit.params.fillna(0)
        srs_vec = srs_vec.rename(index={'const': '(Intercept)'})
        srs_centered = (srs_vec - srs_vec.mean())

        return pd.DataFrame({
            'team': srs_centered.index,
            'season': season,
            'Adj_GC': np.round(srs_centered.values, 2)
        })

    ratings_list = []

    for yr, season_df in game_control_by_game.groupby('season'):
        ratings_list.append(calc_adj_gc_scratch(season_df))

    ratings_by_season = pd.concat(ratings_list, ignore_index=True)

    print("[GC} Finished GC.")


# ---------------
# AWP MODEL
# ---------------

class AWPModel():

    print("[AWP] Initializing...", flush=True)

    # reuse dataset from GC model so that it doesn't load twice
    df = GameControlModel.df_raw.copy()

    wp_model, wp_scaler = train_wp_model(df, AWP_PREDICTORS)

    print("[AWP] Generating win probabilities...", flush=True)

    # score every play, game by game
    wp_frames = []

    for i, game_id in enumerate(df['game_id'].unique(), start=1):

        game_wp = get_game_wp(game_id, wp_model, wp_scaler, AWP_PREDICTORS, df)

        if not game_wp.empty:
            wp_frames.append(game_wp)

    df = pd.concat(wp_frames, ignore_index=True)

    print("[AWP] Calculating game-level AWP...", flush=True)

    def compute_game_awp(group, wp_col="home_wp", dur_col="play_duration", eps=1e-4):
        # time weighted avg win probabilty for one game (logit scale)
        denom = (group[dur_col].sum() or 1.0)

        home_awp = np.clip((group[wp_col] * group[dur_col]).sum() / denom, eps, 1 - eps)

        return pd.Series({
            "home_awp": home_awp,
            "home_awp_logit": logit(home_awp)
        })

    awp_by_game = (df.groupby(["season","game_id", "homeTeamName", "awayTeamName"], group_keys=False)
                   .apply(compute_game_awp).reset_index()
    )

    def calc_adj_awp_scratch(df):
        # opponent adjusted
        
        teams = np.unique(np.concatenate([df['homeTeamName'].unique(), df['awayTeamName'].unique()]))
        n_games = len(df)
        season = df['season'].iloc[0]

        X = pd.DataFrame(0.0, index=range(n_games), columns=teams)

        for i, (_, row) in enumerate(df.iterrows()):
            X.loc[i, row['homeTeamName']] = 1.0
            X.loc[i, row['awayTeamName']] = -1.0

        margin = df['home_awp_logit'].values
        X_design = sm.add_constant(X)
        fit = sm.OLS(margin, X_design).fit()

        srs_vec = fit.params.fillna(0)
        srs_vec = srs_vec.rename(index={'const': '(Intercept)'})

        srs_centered = (srs_vec - srs_vec.mean())

        return pd.DataFrame({
            'team': srs_centered.index,
            'season': season,
            'Adj_AWP_logit': np.round(srs_centered.values, 2)
        })

    ratings_list = []

    for yr, season_df in awp_by_game.groupby('season'):
        ratings_list.append(calc_adj_awp_scratch(season_df))

    ratings_by_season = pd.concat(ratings_list, ignore_index=True)

    # convert back to probability scale
    ratings_by_season['Adj_AWP_prob'] = expit(ratings_by_season['Adj_AWP_logit'])

    print("[AWP] Finished.", flush=True)