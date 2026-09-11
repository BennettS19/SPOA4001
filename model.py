"""
model.py:
    build functions to load data, train models, and setup WP plots; these will be implemented in app.py
    - load_data()
    - train_model()
    - get_game_wp()
    - plot_game_wp()
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
import sportsdataverse as sdv
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import plotly.express as px
import plotly.graph_objects as go

# predictors used from the home team's perspective
HOME_PREDICTORS = [
    "start.adj_TimeSecsRem", "start.homeScore", "start.awayScore",
    "start.homeTeamTimeouts", "start.awayTeamTimeouts",
    "under_2", "down", "distance", "start.yardsToEndzone",
    "start.is_home", "home_team_spread",
]


def load_data(seasons=None):
    """Load and merge play-by-play, schedule, and betting data for the given seasons"""
    # 2025 data for this assignment
    if seasons is None:
        seasons = [2025]

    # load season data (pbp, schedule, betting lines)
    plays_df_polars = sdv.cfb.load_cfb_pbp(seasons=seasons, return_as_pandas=False)
    games_df_polars = sdv.cfb.load_cfb_schedule(seasons=seasons, return_as_pandas=False)
    betting_df_polars = sdv.cfb.load_cfb_betting(seasons=seasons, return_as_pandas=False)

    # convert polar files to pandas df
    pbp_raw = plays_df_polars.to_pandas(use_pyarrow_extension_array=False)
    games_df = games_df_polars.to_pandas(use_pyarrow_extension_array=False)
    betting_df = betting_df_polars.to_pandas(use_pyarrow_extension_array=False)

    keep_cols = ['game_id', 'season'] + HOME_PREDICTORS
    pbp_raw[[c for c in keep_cols if c in pbp_raw.columns]]
    
    # Filter game metadata to isolate the home/away division tags
    games_meta = games_df[
        ['game_id', 'season_type', 'home_division', 'away_division',
         'home_points', 'away_points', 'home_team', 'away_team']
    ]
    # Filter betting data for spread and favorite
    betting_meta = betting_df[['game_id', 'home_team_spread', 'home_favorite']]

    # Merge the division tags onto pbp data
    games_pbp = pd.merge(pbp_raw, games_meta, on='game_id', how='left')
    final_pbp = pd.merge(games_pbp, betting_meta, on='game_id', how='left')

    # New column for home_win indicator 
    final_pbp['home_win'] = final_pbp['home_points'] > final_pbp['away_points']

    # filter for only fbs v fbs games
    fbs_pbp = final_pbp[
        (final_pbp['home_division'] == 'fbs') & (final_pbp['away_division'] == 'fbs')
    ].copy()

    # only games where home_win and predictors aren't missing
    fbs_pbp = fbs_pbp.dropna(subset=["home_win"] + HOME_PREDICTORS)

    return fbs_pbp


def train_model(game_df, predictors=None):
    """Fit the logistic regression win probability model"""
    # use predictors from load_data()
    if predictors is None:
        predictors = HOME_PREDICTORS

    # predicting home_win
    X = game_df[predictors]
    y = game_df["home_win"]
    strata = game_df["season"]  # plays no impact for this assignment since only 2025 data is used

    # split train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=2026, stratify=strata
    )

    # create and fit scaler
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train), columns=predictors, index=X_train.index
    )
    # add win/loss constant
    X_train_sm = sm.add_constant(X_train_scaled)

    # train the log reg
    model = sm.Logit(y_train, X_train_sm).fit(disp=False)

    return model, scaler


def get_game_wp(game_id, model, scaler, predictors, df):
    """Return a single game's plays with predicted home win probability attached"""
    # df filtered to the individual game
    g = df[df['game_id'] == game_id].copy()
    if g.empty:
        return g

    # ensure df sorted by time remaining
    g = g.sort_values('start.adj_TimeSecsRem', ascending=False)

    # grab variables model was trained on
    x_game = g[predictors]

    # scale predictors & add constant to df
    x_game_scaled = pd.DataFrame(
        scaler.transform(x_game), columns=predictors, index=x_game.index
    )
    x_game_scaled = sm.add_constant(x_game_scaled, has_constant='add')

    # generate play win proabilities
    g['home_wp'] = model.predict(x_game_scaled)

    return g


def plot_game_wp(game_id, model, scaler, predictors, df, home_team=None, away_team=None):

    # grab wp using get_game_wp(); new df holds home_wp for each play
    g = get_game_wp(game_id, model, scaler, predictors, df)

    # check that game exists
    if g.empty:
        return None

    # create title
    title = f"Win Probability — Game {game_id}"

    if home_team and away_team:
        title = f"{away_team} @ {home_team} — Win Probability"

    # create Plotly figure
    fig = go.Figure()

    # 50% Baselne
    fig.add_trace(
        go.Scatter(
            x=g["start.adj_TimeSecsRem"],   # x-axis: time remaining
            y=[0.5] * len(g),               # y-axis: value at 50% wp for entirety of game
            line_dash="dash",
            line_color="gray",
            showlegend=False,
            hoverinfo="skip"                # don't allow hover info on 50% line
        )
    )

    # WP Line
    fig.add_trace(
        go.Scatter(
            x=g["start.adj_TimeSecsRem"],   # x-axis: time remaining
            y=g["home_wp"],                 # y-axis: home wp
            mode="lines",
            fill="tonexty",                 # fills area between this line (home wp) and previous line (50% baseline)
            line=dict(width=3),
            name="Home WP",                 # legend name
            customdata=np.column_stack((g["start.awayScore"],g["start.homeScore"])),    # bring in team score data

            # info displayed when hovering over line
            hovertemplate=(
                "Seconds Remaining: %{x}<br>"
                f"{away_team}: %{{customdata[0]}}<br>"
                f"{home_team}: %{{customdata[1]}}<br>"
                "Home WP: %{y:.1%}"
                "<extra></extra>"           # remove extra info displayed (the name: "Home WP")
            )
        )
    )

    # Layout
    fig.update_layout(
        title=title,
        xaxis_title="Seconds Remaining in Game",
        yaxis_title="Home Win Probability",
        hovermode="x"   # hover data based on x value you're in line with
    )

    # Ensure plot shows 0% - 100% range
    fig.update_yaxes(
        range=[0, 1],
        tickformat=".0%"
    )

    # Flip x-axis since time goes down
    fig.update_xaxes(
        range=[0,3600],
        autorange="reversed"
    )

    return fig