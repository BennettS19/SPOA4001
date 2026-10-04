"""
Used https://www.kaggle.com/code/dariussingh/nfl-visualizing-player-tracking-data
to help animate play
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib import animation
from matplotlib.markers import MarkerStyle

game_id = 2022100210
play_id = 2735

df = pd.read_csv("data/BigDataBowl_2024/tracking_week_4.csv")
play = df[(df["gameId"] == game_id) & (df["playId"] == play_id)].copy()

# for home/away coloring
games = pd.read_csv("data/BigDataBowl_2024/games.csv")
g = games[games["gameId"] == game_id].iloc[0]
home_team, away_team = g["homeTeamAbbr"], g["visitorTeamAbbr"]

def create_football_field(fig, ax, line_color='black', field_color='white'):
    """
    Function that plots the football field for viewing players.
    """

    # set field dimensions
    plt.xlim(0,120)
    plt.ylim(0,53.3)

    # adding rectangles to the field
    for i in range(12):
        rect = patches.Rectangle((10*i,0), 10, 53.3, linewidth=1, edgecolor=line_color, facecolor=field_color)
        ax.add_patch(rect)

    # configure axes
    ax.tick_params(
        axis='both',
        which='both',
        direction='in',
        pad=-40,
        length = 5,
        bottom=True,
        top=True,
        labeltop=True,
        labelbottom=True,
        left=False,
        right=False,
        labelleft=False,
        labelright=False,
        color=line_color)

    # set ticks on the side of the field
    ax.set_xticks([i for i in range(10,111)])

    # setting yard marking
    label_set = []
    for i in range(1,10):
        if i<=5:
            label_set += [" " for j in range(9)] + [str(i*10)]
        else:
            label_set += [" " for j in range(9)] + [str((10-i)*10)]
    label_set =  [" "] + label_set + [" " for j in range(10)]
    ax.set_xticklabels(label_set, fontsize=20, color=line_color)

    return fig, ax


def populate_field(game_id:int, play_id:int, frame_id:int, player_tracking_data:pd.DataFrame,
                   home_team:str, away_team:str,
                   home_color:str='violet', away_color:str='coral'):
    """
    populates the field with player tracking data of a game_id, play_id and frame_id.
    """
    # subset data to current play and frame
    frame_info = player_tracking_data.query(
        "gameId==@game_id and playId==@play_id and frameId==@frame_id"
    ).copy()

    # create new field
    fig, ax = plt.subplots(figsize=(12, 5.33))
    fig, ax = create_football_field(fig, ax)

    # set title
    ax.set_title(f'Tracking data for game {game_id}, play {play_id} at frame {frame_id}')

    # populate field with players
    for row in frame_info.iterrows():
        if row[1]['club'] == 'football':
            ax.scatter(row[1]['x'], row[1]['y'], marker='o', s=60, color='saddlebrown')
            continue
        if row[1]['club'] == home_team:
            color = home_color
        else:
            color = away_color
        marker1 = MarkerStyle(r'$\spadesuit$')
        marker1._transform.rotate_deg(360-row[1]['o'])
        ax.scatter(row[1]['x'], row[1]['y'], marker=marker1, s=150, color=color)

    plt.close()
    return fig, ax


# total number of frames at 10hz
frames = play['frameId'].nunique()

# frequency modifier factor [min_val:1, max_val:10] (reduce this for smoother tracking but longer rendering time)
freq_mod_fac = 2
frames = int(frames/freq_mod_fac)
interval_ms = 100*freq_mod_fac

# initialize figure
fig, ax = plt.subplots(figsize=(12, 5.33))
fig, ax = create_football_field(fig, ax)


def animate(i:int, game_id:int, play_id:int, player_tracking_data:pd.DataFrame, frames,
            home_team:str, away_team:str, home_color:str='violet', away_color:str='coral'):
    """
    Function to animate player tracking data
    """
    # create fresh field
    ax.clear()
    create_football_field(fig, ax)

    # find appropriate frame
    play_info = player_tracking_data.query("gameId==@game_id and playId==@play_id").copy()
    frame_list = np.linspace(play_info['frameId'].min(), play_info['frameId'].max(), frames)
    frame_id = int(frame_list[i])

    # subset data to frame info
    frame_info = play_info.query("frameId==@frame_id").copy()

    # iterate frame info to populate field
    for row in frame_info.iterrows():
        # draw football
        if row[1]['club'] == 'football':
            ax.scatter(row[1]['x'], row[1]['y'], marker='o', s=60, color='saddlebrown')
            continue
        # draw players
        if row[1]['club'] == home_team:
            color = home_color
        else:
            color = away_color
        marker1 = MarkerStyle(r'$\spadesuit$')
        marker1._transform.rotate_deg(360-row[1]['o'])
        ax.scatter(row[1]['x'], row[1]['y'], marker=marker1, s=150, color=color)

    # set axis title
    ax.set_title(f'Tracking data for game {game_id}, play {play_id} at frame {frame_id}')


# animate
anim = animation.FuncAnimation(
    fig, animate,
    fargs=(game_id, play_id, play, frames, home_team, away_team),
    frames=frames, repeat=False, interval=interval_ms
)

# save as gif
anim.save("HW5/play.gif", writer="pillow", fps=10 / freq_mod_fac)