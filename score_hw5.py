import pandas as pd

games_orig = pd.read_csv("data/BigDataBowl_2024/games.csv")
games_new = pd.read_csv("data/BigDataBowl_2024/games2.csv")

orig_lac_den_6 = games_orig[
    (games_orig["week"] == 6) &
    (games_orig["homeTeamAbbr"] == "LAC") &
    (games_orig["visitorTeamAbbr"] == "DEN")
]

new_lac_den_6 = games_new[
    (games_new["week"] == 6) &
    (games_new["homeTeamAbbr"] == "LAC") &
    (games_new["visitorTeamAbbr"] == "DEN")
]

print(f"Original score: {orig_lac_den_6['homeFinalScore'].iloc[0]} - "
    f"{orig_lac_den_6['visitorFinalScore'].iloc[0]}"
)

print(f"New score: {new_lac_den_6['homeFinalScore'].iloc[0]} - "
    f"{new_lac_den_6['visitorFinalScore'].iloc[0]}"
)