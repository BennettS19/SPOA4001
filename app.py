"""
app.py:
    interactive streamlit app; input game ID to receive WP plot throughout the game
    - Sidebar display with team dropdown - can grab game IDs from here
    - WP play-by-play plot with hovering abilities
    - Game basic info - Teams, Location, Final Score, Week, Spread
"""

import streamlit as st
from model import HOME_PREDICTORS, load_data, train_model, plot_game_wp
import pandas as pd
import numpy as np
import plotly.graph_objects as go

st.set_page_config(page_title="CFB Win Probability", layout="wide")

st.title("CFB Win Probability Viewer")
st.write("Enter a `game_id` from the 2025 FBS season to see the home team's win probability over time.")

# Load data and model using model.py functions
game_df = st.cache_data(load_data)()
model, scaler = st.cache_resource(train_model)(game_df)


################################################

# SIDEBAR DISPLAY TO FIND GAME ID'S VIA SCHEDULE

################################################

# Team Schedules Sidebar
st.sidebar.header("Team Schedule")

schedule_df = (
    game_df[["game_id", "week", "season_type", "home_team", "away_team", "home_points", "away_points", "home_team_spread"]]
    .drop_duplicates(subset="game_id")
    .sort_values("week")
)

# Get all teams
teams = sorted(pd.concat([
    schedule_df["home_team"],
    schedule_df["away_team"]
]).dropna().unique())

# Team dropdown
selected_team = st.sidebar.selectbox(
    "Select Team",
    teams
)

# Filter gamesfor selected team
team_schedule = schedule_df[
    (schedule_df["home_team"] == selected_team) |
    (schedule_df["away_team"] == selected_team)
].copy()

# Week column to adjust postseason week
team_schedule["Week"] = np.where(
    team_schedule["season_type"] == "postseason",
    "PS",
    team_schedule["week"].astype(str)
)
# New sorting value since sorting by Week sorts by string value
team_schedule["week_sort"] = np.where(
    team_schedule["season_type"] == "postseason",
    99,
    pd.to_numeric(team_schedule["week"], errors="coerce")
)
team_schedule = team_schedule.sort_values("week_sort")

# Create location column (home/away)
team_schedule["Location"] = np.where(
    team_schedule["home_team"] == selected_team,
    "vs",
    "@"
)

# Create opponnet column
team_schedule["Opponent"] = np.where(
    team_schedule["home_team"] == selected_team,
    team_schedule["away_team"],
    team_schedule["home_team"]
)

# Show score from selected team's perspective
team_schedule["Score"] = np.where(
    team_schedule["home_team"] == selected_team,
    team_schedule["home_points"].astype(str) + "-" +
    team_schedule["away_points"].astype(str),
    team_schedule["away_points"].astype(str) + "-" +
    team_schedule["home_points"].astype(str)
)

# Column for selected team's spread
team_schedule["Spread"] = np.where(
    team_schedule["home_team"] == selected_team,
    team_schedule["home_team_spread"],
    -team_schedule["home_team_spread"]
)

# Display sidebar schedule
st.sidebar.dataframe(
    team_schedule[["game_id", "Week", "Location", "Opponent", "Score"]]
    .rename(columns={"game_id": "Game ID"}),
    hide_index=True,
    width='stretch'
)


##############################

# DISPLAY INDIVIDUAL GAME INFO

##############################

# Game ID input
game_id_input = st.text_input("Game ID", placeholder="e.g. 401756893")

if game_id_input:
    # make sure input is a number
    try:
        game_id = int(game_id_input)
    except ValueError:
        st.error("Game ID should be a number.")
        game_id = None

    if game_id is not None:
        # grab df for game id if it exists
        match = game_df[game_df['game_id'] == game_id]
        if match.empty:
            st.error(f"No plays found for game_id {game_id}")
        # grab home and away teams
        else:
            home_team = match['home_team'].iloc[0] if 'home_team' in match.columns else None
            away_team = match['away_team'].iloc[0] if 'away_team' in match.columns else None

            # get info for matchup display
            game_info = match.iloc[0]

            home_team = game_info["home_team"]
            away_team = game_info["away_team"]
            home_score = game_info["home_points"]
            away_score = game_info["away_points"]
            home_spread = game_info["home_team_spread"]

            # Adjust week display for postseason
            if game_info["season_type"] == "postseason":
                week_display = "PS"
            else:
                week_display = f"Week {game_info['week']}"

            # Away team's spread is the opposite of home team's spread
            away_spread = -home_spread

            # Win probability plot (using function from model.py)
            fig = plot_game_wp(
                game_id, model, scaler, HOME_PREDICTORS, game_df,
                home_team=home_team, away_team=away_team
            )
            if fig is not None:
                st.plotly_chart(fig, width='stretch')


            ###############################
            # CLEAN DISPLAY OF MATCHUP INFO
            ###############################

            # Matchup header (centered)
            st.markdown(
                f"<h3 style='text-align: center;'>{away_team} @ {home_team}</h3>",
                unsafe_allow_html=True
            )

            # Create columns (teams, score, week)
            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.metric("AWAY", away_team)
            with col2:
                st.metric("FINAL SCORE", f"{away_score} - {home_score}")
            with col3:
                st.metric("HOME", home_team)
            with col4:
                st.metric("WEEK", week_display)


            # Columns for spreads
            odds_col1, odds_col2 = st.columns(2)

            # display team spreads (always show "+"/"-")
            with odds_col1:
                st.metric(
                    f"{away_team} Spread",
                    f"{away_spread:+.1f}"
                )
            with odds_col2:
                st.metric(
                    f"{home_team} Spread",
                    f"{home_spread:+.1f}"
                )