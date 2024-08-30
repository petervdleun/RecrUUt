import pandas as pd
import streamlit as st
import plotly.express as px
import matplotlib.pyplot as plt
import scipy.stats as stats
import textwrap
import urllib3
import base64
import numpy as np
import requests
import os
from bs4 import BeautifulSoup
from dateutil import parser
from matplotlib.patches import Patch
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from datetime import datetime
from difflib import get_close_matches
from PIL import Image
from io import BytesIO
from scipy.stats import percentileofscore, zscore
from mplsoccer import PyPizza
from streamlit_elements import elements, mui, dashboard, html
from reportlab.lib.pagesizes import landscape, A4
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader

st.set_page_config(layout='wide')

# Data import
data_wysc = os.path.join('data', 'wysc_test.csv')
db = pd.read_csv(data_wysc)
data_logos = os.path.join('data', 'logo_info.csv')
# db = pd.read_csv('data\wysc_test.csv')
logos = pd.read_csv(data_logos)
comps = pd.read_excel('data/wysc_comps.xlsx')
# colors = pd.read_excel('logo_colors.xlsx')
# transfermarkt = pd.read_csv('tm_try.csv')

db = db.merge(logos, left_on='last_club_name', right_on='current_team_name', how='left')
# db = db.merge(colors, left_on='last_club_name', right_on='current_team_name', how='left')
db = db.merge(comps, on='competition_id', how='left')

# # Function to match players from Wyscout and TM
# def find_match(row, transfermarkt_df):
#     dob = row['birth_date']
#     name = row['full_name']
    
#     # Filter transfermarkt data by DOB
#     matches = transfermarkt_df[transfermarkt_df['DOB'] == dob]
    
#     if not matches.empty:
#         # Find the closest name match
#         close_match = get_close_matches(name, matches['Player'], n=1, cutoff=0.8)
        
#         if close_match:
#             matched_row = matches[matches['Player'] == close_match[0]].iloc[0]
#             return pd.Series({'transfermarkt_id': matched_row['ID'], 'transfermarkt_player': matched_row['Player']})
    
#     return pd.Series({'transfermarkt_id': None, 'transfermarkt_player': None})

# # Apply the function to find matches for the selected player
# matches = db.apply(find_match, axis=1, transfermarkt_df=transfermarkt)
# db = pd.concat([db, matches], axis=1)

# Mapping function to assign templates based on positions
def map_position_to_template(position):
    if position in ["CB", "LCB", "RCB", "LCB3", "RCB3"]:
        return "CB"
    elif position in ["LB", "LB5", "RB", "RB5", "LWB", "RWB"]:
        return "FB"
    elif position in ["DMF", "LDMF", "RDMF"]:
        return "DM"
    elif position in ["AMF", "LCMF", "RCMF", "RCMF3", "LCMF3"]:
        return "CM"
    elif position in ["LAMF", "LW", "LWF", "RAMF", "RW", "RWF"]:
        return "W"
    elif position == "CF":
        return "ST"
    elif position == "GK":
        return "GK"
    else:
        return None

# Add new metrics
db['template_1'] = db['primary_position'].apply(map_position_to_template)
db['template_2'] = db['secondary_position'].apply(map_position_to_template)
db['minutes_per_match'] = db['minutes_on_field'] / db['total_matches']
db['non_penalty_xg'] = db['xg_shot'] - (db['penalties_taken'] * 0.76)
db['90s'] = db['minutes_on_field'] / 90
db['non_penalty_xg_90'] = db['non_penalty_xg'] / db['90s']
db['non_penalty_xg_shot'] = (db['non_penalty_xg'] / db['shots_avg']).fillna(0)
db['crosses_per_rp'] = db['crosses_avg'] / db['received_pass_avg']
db['successful_crosses_90'] = (db['accurate_crosses_percent']/100) * db['crosses_avg']
db['successful_dribbles_90'] = (db['successful_dribbles_percent']/100) * db['dribbles_avg']
db['received_long_rate'] = db['received_long_pass_avg'] / db['received_pass_avg']
db['forward_pass_rate'] = db['forward_passes_avg'] / db['passes_avg']
db['long_pass_rate'] = db['long_passes_avg'] / db['passes_avg']
db['progressive_pass_rate'] = db['progressive_pass_avg'] / db['passes_avg']
db['xa_90'] = db['xg_assist'] / db['90s']
db['xa_per_rp'] = db['xg_assist'] / db['received_pass_avg']
db['fouls_90_inv'] = 10 - db['fouls_avg']
db['successful_pass_entries_f3_90'] = (db['accurate_passes_to_final_third_percent']/100) * db['passes_to_final_third_avg']
db['successful_pass_entries_box_90'] = (db['accurate_pass_to_penalty_area_percent']/100) * db['pass_to_penalty_area_avg']
db['defensive_duels_won_90'] = (db['defensive_duels_won']/100) * db['defensive_duels_avg']
db['aerial_duels_won_90'] = (db['aerial_duels_won']/100) * db['aerial_duels_avg']
db['xg_conv'] = db['non_penalty_goal'] - db['non_penalty_xg']
db['gk_off_line'] = db['goalkeeper_exits_avg'] + db['aerial_duels_avg']

# Create a dictionary to map competition_id to competition name
db['display_name'] = db['country'] + " | " + db['competition']
# Clean the 'full_name' and 'last_club_name' columns
db['full_name'] = db['full_name'].apply(lambda x: ' '.join(x.split()))
db['last_club_name'] = db['last_club_name'].apply(lambda x: ' '.join(x.split()))

# Filter out GKs (temporary until fix)
db = db[db['template_1'] != "GK"]

def get_image_as_base64(image_path):
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode()

image_base64 = get_image_as_base64("RecrUUT.png")

st.markdown(
    f"""
    <div style="position: fixed; top: 10px; left: 10px;">
        <img src="data:image/png;base64,{image_base64}" width="150" style="border-radius: 10px;">
    </div>
    """,
    unsafe_allow_html=True
)

# Create league dictionary
comp_dict = db[['competition_id', 'display_name']].drop_duplicates().set_index('competition_id')['display_name'].to_dict()
# Reverse the dictionary to get a mapping from competition name to competition_id
comp_name_to_id = {v: k for k, v in comp_dict.items()}
# Sort competition names
sorted_comp_names = sorted(comp_name_to_id.keys())
# Find the index of the default competition
default_comp = 'NLD | Eredivisie'
default_index = sorted_comp_names.index(default_comp)
selected_comp = st.sidebar.selectbox('League', sorted_comp_names, index=default_index)
# Get the competition_id corresponding to the selected competition
selected_comp_id = comp_name_to_id[selected_comp]
# Filter the dataframe based on the selected competition_id
db_comp = db[db['competition_id'] == selected_comp_id]
db_comp = db_comp[db_comp['minutes_on_field'] > 50]

# Create a dictionary of seasons available for the selected competition
available_seasons = sorted(db_comp['season_id'].unique())
# Set '2024-25' as the default season if available
default_season = '2024-25'
default_season_index = available_seasons.index(default_season) if default_season in available_seasons else 0
# Create selectbox for season
selected_season = st.sidebar.selectbox('Season', available_seasons, index=default_season_index)
# Filter the DataFrame based on the selected season
db_comp = db_comp[db_comp['season_id'] == selected_season]

# Group by 'name' and '90s', and filter out groups where both 'name' and '90s' are the same
duplicate_glitch = db_comp.duplicated(subset=['received_pass_avg', 'passes_avg', '90s'], keep=False)
# For duplicated entries, remove the ones where 'birth_date' is None
db_comp = db_comp[~(duplicate_glitch & db_comp['birth_date'].isna())]

# Filter teams based on selected league
teams = db_comp['last_club_name'].drop_duplicates().tolist()
teams.sort()
teams.insert(0, 'All Teams')  # Default option when no team is selected
# Selectbox for team filter
selected_team = st.sidebar.selectbox('Team', teams)

# Filter players based on selected league and team
if selected_team == 'All Teams':
    db_team = db_comp  # Show all players if 'All Teams' is selected
else:
    db_team = db_comp[db_comp['last_club_name'] == selected_team]

# Define custom sort order for positions
pos_sort = {
    "GK": 1, "CB": 2, "FB": 3, "DM": 4, "CM": 5, "W": 6, "ST": 7
}

pos_disp = {
    "GK": "Goalkeeper", "CB": "Centre-Back", "FB": "Full-Back", "DM": "Defensive Midfield", "CM": "Centre-Midfield", "W": "Winger", "ST": "Striker"
}

positions = [pos for pos in db_team['template_1'].drop_duplicates().tolist() if pos is not None]
positions.sort(key=lambda x: pos_sort.get(x, float('inf')))
# Create a display list for the selectbox
positions_display = ["All Positions"] + [pos_disp.get(pos, pos) for pos in positions]

# Map display names back to original values for filtering
display_to_position = {v: k for k, v in pos_disp.items()}
display_to_position["All Positions"] = "All Positions"

# Selectbox for position filter
selected_display_position = st.sidebar.selectbox('Position', positions_display)
# Filter players based on selected league, team, and position
selected_position = display_to_position[selected_display_position]

# Filter players based on selected league, team, and position
if selected_position == 'All Positions':
    db_position = db_team
else:
    db_position = db_team[db_team['template_1'] == selected_position]

# Sort db_position by minutes_on_field
db_position_sort = db_position.sort_values(by='minutes_on_field', ascending=False)
# full_names = db_position_sort['full_name'].drop_duplicates().tolist()

# Create a mapping from full_name to id
name_to_id = db_position_sort.set_index('full_name')['id'].to_dict()

# Add a dashed line
st.sidebar.markdown("<hr style='border: 1px dashed grey;'>", unsafe_allow_html=True)
# Create a selectbox with full names
selected_name = st.sidebar.selectbox('Player [sorted by minutes]', list(name_to_id.keys()))
# Get the corresponding id
selected_id = name_to_id[selected_name]

# st.sidebar.text(f"Players: {len(db_comp)}")
# st.sidebar.text(f"Last Data Refresh: {db_comp['export_date'].iloc[0]}")

# Create the first filtered logs DataFrame with all logs for the selected player
db_player = db_comp[db_comp['id'] == selected_id]

# # Use current_team_color for hr border color
# team_color = db_player['color_code'].iloc[0]
# if team_color == '#000000' or pd.isna(team_color):
#         team_color = '#ffffff'

# Function to generate radar chart
def display_radar_chart(selected_player_data, template_players_data, selected_template, selected_league):
    # Dictionary mapping templates to sets of stats
    template_stats_mapping = {
        'GK': ['gk_off_line', 'xg_save_avg', 'long_pass_rate'],
        'CB': ['defensive_duels_avg', 'defensive_duels_won', 'aerial_duels_avg', 'aerial_duels_won', 'passes_avg', 'accurate_passes_percent',
               'progressive_pass_avg', 'successful_progressive_pass_percent', 'successful_pass_entries_f3_90', 'progressive_run_avg',
               'possession_adjusted_interceptions', 'shot_block_avg', 'fouls_90_inv'],
        'FB': ['non_penalty_xg_90', 'touch_in_box_avg', 'received_pass_avg', 'accurate_passes_percent', 'deep_completed_pass_avg',
               'crosses_per_rp', 'successful_pass_entries_box_90', 'xa_per_rp', 'progressive_run_avg', 'successful_dribbles_90',
               'defensive_duels_avg', 'defensive_duels_won', 'possession_adjusted_interceptions'],
        'DM': ['received_pass_avg', 'accurate_passes_percent', 'forward_pass_rate', 'progressive_pass_rate',
               'successful_pass_entries_f3_90', 'progressive_run_avg', 'defensive_duels_won_90', 'aerial_duels_won_90', 'possession_adjusted_interceptions'],
        'CM': ['non_penalty_xg_90', 'shots_avg', 'non_penalty_xg_shot', 'touch_in_box_avg', 'received_pass_avg', 'accurate_passes_percent',
               'successful_pass_entries_f3_90', 'successful_pass_entries_box_90', 'xa_per_rp', 'progressive_run_avg',
               'successful_dribbles_90', 'defensive_duels_won', 'possession_adjusted_interceptions'],
        'W':  ['non_penalty_xg_90', 'shots_avg', 'non_penalty_xg_shot', 'touch_in_box_avg', 
               'received_pass_avg', 'deep_completed_pass_avg', 'successful_pass_entries_box_90', 'xa_per_rp', 
               'foul_suffered_avg', 'successful_dribbles_90', 'progressive_run_avg', 'successful_pass_entries_f3_90'],
        'ST': ['non_penalty_xg_90', 'shots_avg', 'non_penalty_xg_shot', 'xg_conv', 'touch_in_box_avg', 'aerial_duels_won',
               'foul_suffered_avg', 'xa_per_rp', 'successful_pass_entries_box_90', 'successful_dribbles_90']
    }

    # Dictionary mapping stats to their display names
    stat_display_names = {
        'gk_off_line': 'GK Off Line',
        'xg_save_avg': 'xG Save Avg',
        'long_pass_rate': 'Long Pass Rate',
        'defensive_duels_avg': 'Defensive Duels per 90',
        'defensive_duels_won': 'Defensive Duels Won (%)',
        'aerial_duels_avg': 'Aerial Duels per 90',
        'aerial_duels_won': 'Aerial Duels Won (%)',
        'passes_avg': 'Passes per 90',
        'accurate_passes_percent': 'Accurate Passes (%)',
        'progressive_pass_avg': 'Progressive Passes per 90',
        'successful_progressive_pass_percent': 'Successful Progressive Passes (%)',
        'successful_pass_entries_f3_90': 'Successful Pass Entries Final Third per 90',
        'progressive_run_avg': 'Progressive Runs per 90',
        'possession_adjusted_interceptions': 'Possession Adjusted Interceptions',
        'shot_block_avg': 'Shot Blocks per 90',
        'fouls_90_inv': 'Fouls Inv per 90',
        'non_penalty_xg_90': 'Non-Penalty xG per 90',
        'touch_in_box_avg': 'Touches in Box per 90',
        'received_pass_avg': 'Received Passes per 90',
        'deep_completed_pass_avg': 'Deep Completed Passes per 90',
        'crosses_per_rp': 'Crosses per Received Pass',
        'successful_pass_entries_box_90': 'Successful Pass Entries Box per 90',
        'xa_per_rp': 'xA per Received Pass',
        'successful_dribbles_90': 'Successful Dribbles per 90',
        'defensive_duels_won_90': 'Defensive Duels Won per 90',
        'aerial_duels_won_90': 'Aerial Duels Won per 90',
        'shots_avg': 'Shots per 90',
        'non_penalty_xg_shot': 'Non-Penalty xG per Shot',
        'xg_conv': 'xG Conversion Rate',
        'foul_suffered_avg': 'Fouls Suffered per 90',
        'forward_pass_rate': 'Forward Pass Rate (%)',
        'progressive_pass_rate': 'Progressive Pass Rate (%)'
    }

    # Define groups based on the selected template
    if selected_template == 'ST':
        finishing_stats = ['non_penalty_xg_90', 'shots_avg', 'non_penalty_xg_shot', 'xg_conv', 'touch_in_box_avg']
        holdup_stats = ['foul_suffered_avg', 'aerial_duels_won']
        playmaking_stats = ['xa_per_rp', 'successful_dribbles_90', 'successful_pass_entries_box_90']
        aggression_stats = [] 
        passing_stats = [] 
        progression_stats = []
        involvement_stats = []
        defensive_stats = []
    elif selected_template == 'W':
        holdup_stats = []
        finishing_stats = ['non_penalty_xg_90', 'shots_avg', 'non_penalty_xg_shot', 'xg_conv', 'touch_in_box_avg']
        playmaking_stats = ['foul_suffered_avg', 'successful_pass_entries_box_90', 'xa_per_rp', 'successful_dribbles_90']
        involvement_stats = ['received_pass_avg', 'deep_completed_pass_avg']
        progression_stats = ['progressive_run_avg', 'successful_pass_entries_f3_90']
    elif selected_template == 'DM':
        holdup_stats = []
        finishing_stats = []
        playmaking_stats = []
        involvement_stats = []
        progression_stats = ['progressive_run_avg', 'successful_pass_entries_f3_90']
        aggression_stats = []  # Initialize aggression_stats if needed
        passing_stats = ['received_pass_avg', 'accurate_passes_percent', 'forward_pass_rate', 'progressive_pass_rate']
        defensive_stats = ['defensive_duels_won_90', 'aerial_duels_won_90', 'possession_adjusted_interceptions']
    elif selected_template == 'CM':
        holdup_stats = []
        involvement_stats = []
        progression_stats = []
        finishing_stats = ['non_penalty_xg_90', 'shots_avg', 'non_penalty_xg_shot', 'xg_conv', 'touch_in_box_avg']
        passing_stats = ['received_pass_avg', 'accurate_passes_percent', 'successful_pass_entries_f3_90']
        playmaking_stats = ['successful_pass_entries_box_90', 'progressive_run_avg', 'xa_per_rp', 'successful_dribbles_90']
        defensive_stats = ['defensive_duels_won', 'possession_adjusted_interceptions']
        aggression_stats = []  # Initialize aggression_stats if needed
    elif selected_template == 'FB':
        holdup_stats = []
        involvement_stats = []
        progression_stats = []
        finishing_stats = ['non_penalty_xg_90', 'shots_avg', 'non_penalty_xg_shot', 'xg_conv', 'touch_in_box_avg']
        passing_stats = ['received_pass_avg', 'accurate_passes_percent', 'deep_completed_pass_avg']
        playmaking_stats = ['successful_pass_entries_box_90', 'crosses_per_rp', 'progressive_run_avg', 'xa_per_rp', 'successful_dribbles_90']
        defensive_stats = ['defensive_duels_avg', 'defensive_duels_won', 'possession_adjusted_interceptions']
        aggression_stats = []  # Initialize aggression_stats if needed
    elif selected_template == 'CB':
        holdup_stats = []
        involvement_stats = []
        finishing_stats = []
        playmaking_stats = []
        passing_stats = ['passes_avg', 'accurate_passes_percent']
        progression_stats = ['progressive_pass_avg', 'successful_progressive_pass_percent', 'progressive_run_avg', 'successful_pass_entries_f3_90']
        defensive_stats = ['defensive_duels_won', 'defensive_duels_avg', 'aerial_duels_avg', 'aerial_duels_won']
        aggression_stats = ['possession_adjusted_interceptions', 'shot_block_avg', 'fouls_90_inv']
    elif selected_template == 'GK':
        holdup_stats = ['gk_off_line', 'xg_save_avg', 'long_pass_rate']
        involvement_stats = []
        finishing_stats = []
        playmaking_stats = []
        passing_stats = []
        progression_stats = []
        defensive_stats = []
        aggression_stats = []

    # Get the set of stats for the selected template
    stats_to_percentile = template_stats_mapping.get(selected_template, [])

    # Calculate percentiles for each stat
    percentiles = {}
    for stat in stats_to_percentile:
        selected_player_stat = selected_player_data[stat].iloc[0]
        template_players_stat = template_players_data[stat]
        percentile_value = percentileofscore(template_players_stat, selected_player_stat)
        percentiles[stat] = round(percentile_value, 0)

    # Update params and group_labels with display names
    params = [stat_display_names.get(stat, stat) for stat in stats_to_percentile]

    # instantiate PyPizza class
    baker = PyPizza(
        params=params,
        background_color="#222222",
        straight_line_color="#000000",
        straight_line_lw=1.2,
        last_circle_color="#000000",
        last_circle_lw=1.2,
        other_circle_lw=0,
        inner_circle_size=5
    )

    # color for the slices and text
    slice_colors = []
    group_labels = []
    for stat in stats_to_percentile:
        # display_name = stat_display_names.get(stat, stat)
        if stat in finishing_stats:
            slice_colors.append("#FFCCCB")
            group_labels.append(("Finishing", "#FFCCCB"))
        elif stat in holdup_stats:
            slice_colors.append("#FFFEE0")
            group_labels.append(("Holdup", "#FFFEE0"))
        elif stat in playmaking_stats:
            slice_colors.append("#CBC3E3")
            group_labels.append(("Playmaking", "#CBC3E3"))
        elif stat in involvement_stats:
            slice_colors.append("#FFFEE0")
            group_labels.append(("Involvement", "#FFFEE0"))
        elif stat in progression_stats:
            slice_colors.append("#ADD8E6")
            group_labels.append(("Progression", "#ADD8E6"))
        elif stat in passing_stats:
            slice_colors.append("#FFFEE0")
            group_labels.append(("Passing", "#FFFEE0"))
        elif stat in defensive_stats:
            slice_colors.append("#B2F3B2")
            group_labels.append(("Defense", "#B2F3B2"))
        elif stat in aggression_stats:
            slice_colors.append("#FFCCCB")
            group_labels.append(("Aggression", "#FFCCCB"))
    text_colors = ["#000000"] * len(stats_to_percentile)

    percentile_values = [int(value) for value in percentiles.values()]

    # plot pizza
    fig, ax = baker.make_pizza(
        percentile_values,
        figsize=(8, 8.5),
        color_blank_space="same",
        slice_colors=slice_colors,
        value_colors=text_colors,
        value_bck_colors=slice_colors,
        blank_alpha=0.4,
        kwargs_slices=dict(
            edgecolor="#000000", zorder=2, linewidth=1.2
        ),
        kwargs_params=dict(
            color="whitesmoke", fontsize=7, weight="bold", va="center",
            bbox=dict(
                edgecolor="darkgrey", facecolor="black", boxstyle="round,pad=0.4", lw=1.5
            )
        ),
        kwargs_values=dict(
            color="#000000", fontsize=9.5, zorder=3,
            bbox=dict(
                edgecolor="#000000", boxstyle="round,pad=0.2", lw=1
            )
        )
    )

    # Obtain selected player information
    selected_player_name = selected_player_data['full_name'].iloc[0]
    # selected_player_name = ' '.join(selected_player_name.split()) # removal of incidental double whitespaces in dataset
    selected_player_age = pd.to_datetime(selected_player_data['birth_day'].iloc[0]).year
    selected_player_team = selected_player_data['last_club_name'].iloc[0]
    # selected_player_team = ' '.join(selected_player_team.split()) # removal of incidental double whitespaces in dataset
    selected_player_minutes = int(selected_player_data['minutes_on_field'].iloc[0])

    # Function to convert image from URL to Matplotlib-compatible format
    def convert_image_from_url(image_url):
        # Initialize a PoolManager
        http = urllib3.PoolManager()
        # Send a GET request to the image URL
        response = http.request('GET', image_url)
        # Open the image from the response content
        img = Image.open(BytesIO(response.data)).convert("RGBA")  # Convert to RGBA format
        # Convert the image to a NumPy array
        img_array = np.array(img)
        return img_array

    # Set title with player name dynamically
    title_text = f"{selected_player_name}"

    # Get the image from PIL format to display in Matplotlib
    player_image = convert_image_from_url(image_url)

    # Add the image to the Matplotlib plot using AnnotationBbox
    imagebox = OffsetImage(player_image, zoom=0.24)
    ab = AnnotationBbox(imagebox, (1.05, 1.21), frameon=False, pad=0, xycoords='axes fraction')
    ax.add_artist(ab)
    ax.set_title(title_text, fontsize=14, pad=82, color='white', fontweight='bold')
    # Set subtitle with other player information
    subtitle_text = f"{selected_player_age} • {selected_player_team} • {selected_player_minutes} minutes • {season} \n {selected_league}"
    fig.suptitle(subtitle_text, fontsize=11, color='white')

    # Accessing the text labels, adjusting their positions, and setting rotation
    for text_label in ax.texts: 
        # Remove 'per 90' from the text label
        original_text = text_label.get_text()
        modified_text = original_text.replace('per 90', '')
        modified_text = modified_text.replace('Inv', '(Higher=Few)')
        # Break lines in the modified text label
        wrapped_text = textwrap.fill(modified_text, width=13)  # Adjust the width as needed
        # Set the wrapped text back to the label
        text_label.set_text(wrapped_text)
        text_label.set_rotation(0)  # Set rotation to 0 for horizontal text
        x, y = text_label.get_position()
        # Check if the label is positioned outside and add a white border
        if original_text in stats_to_percentile:
            text_label.set_bbox(dict(facecolor='black', edgecolor='white', boxstyle='round,pad=0.3'))
    
     # Create a legend
    legend_elements = [Patch(color=color, label=label) for label, color in set(group_labels)]
    legend = ax.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(-0.12, 1.25), facecolor='#0e1117', edgecolor='white',)
    # Set the legend text color to white
    for text in legend.get_texts():
        text.set_color('white')
        text.set_fontsize(9)

    # Annotation at the bottom center
    sample = len(template_players_data)
    annotation_text = f"Compared against {sample}\npositional peers"
    fig.text(0.5, 0.02, annotation_text, ha='center', va='center', color='white', fontsize=10)

    fig.patch.set_facecolor('#0e1117')

    # Display the figure using st.pyplot
    return fig

# Function to save radar chart as PNG
def save_radar_chart_as_png(fig):
    # Save the figure to a BytesIO object
    img_bytes = BytesIO()
    fig.savefig(img_bytes, format='png', bbox_inches='tight', pad_inches=0.2, dpi=300)
    plt.close(fig)
    return img_bytes.getvalue()

weights_dict = {
    'deep_striker': {'received_pass_avg': 0.3, 'xa_per_rp': 0.3, 'successful_pass_entries_box_90': 0.2, 'received_long_rate': -0.2},
    'poacher': {'non_penalty_xg_90': 0.3, 'non_penalty_xg_shot': 0.3, 'touch_in_box_avg': 0.2, 'xg_conv': 0.2},
    'target': {'aerial_duels_won_90': 0.3, 'received_long_rate': 0.2, 'forward_pass_rate': -0.15, 'successful_dribbles_90': -0.15, 'foul_suffered_avg': 0.2},
    'b2b': {'touch_in_box_avg': 0.15, 'non_penalty_xg_90': 0.1, 'progressive_run_avg': 0.25, 'defensive_duels_won': 0.25, 'long_pass_rate': -0.15},
    'drive': {'progressive_run_avg': 0.5, 'received_long_rate': 0.5},
    'ball_winner': {'defensive_duels_won_90': 0.25, 'possession_adjusted_interceptions': 0.25, 'long_pass_rate': -0.1, 'accurate_passes_percent': 0.25, 'deep_completed_pass_avg': -0.1},
    'playmaker': {'possession_adjusted_interceptions': 0.15, 'received_pass_avg': 0.2, 'forward_pass_rate': 0.1, 'received_long_rate': -0.15, 'successful_pass_entries_f3_90': 0.2, 'successful_progressive_pass_percent': 0.1, 'successful_dribbles_90': 0.1},
    'off': {'non_penalty_xg_90': 0.2, 'non_penalty_xg_shot': 0.15, 'touch_in_box_avg': 0.2, 'received_pass_avg': -0.15, 'successful_pass_entries_f3_90': -0.1, 'xg_conv': 0.1},
    'creator': {'successful_pass_entries_box_90': 0.2, 'xg_assist': 0.1, 'xa_per_rp': 0.15, 'successful_dribbles_90': 0.15, 'deep_completed_pass_avg': 0.1, 'foul_suffered_avg': 0.1, 'touch_in_box_avg': 0.1, 'accurate_passes_percent': -0.1},
    'deep': {'non_penalty_xg_90': 0.2, 'touch_in_box_avg': 0.2, 'received_pass_avg': -0.1, 'deep_completed_pass_avg': -0.1, 'received_long_rate': 0.15, 'forward_pass_rate': -0.15, 'successful_pass_entries_f3_90': -0.1},
    'wide': {'received_pass_avg': 0.1, 'crosses_per_rp': 0.25, 'accurate_crosses_percent': 0.1, 'xa_per_rp': 0.15, 'successful_pass_entries_box_90': 0.1, 'successful_pass_entries_f3_90': -0.1, 'foul_suffered_avg': 0.1},
    'inside': {'received_pass_avg': 0.2, 'successful_pass_entries_f3_90': 0.2, 'deep_completed_pass_avg': 0.1, 'crosses_per_rp': -0.1, 'pass_to_penalty_area_avg': 0.1, 'successful_dribbles_90': 0.15, 'forward_pass_rate': 0.1, 'received_long_rate': -0.05},
    'overlap': {'deep_completed_pass_avg': 0.1, 'crosses_per_rp': 0.2, 'accurate_crosses_percent': 0.05, 'successful_pass_entries_box_90': 0.2, 'xa_per_rp': 0.2, 'progressive_run_avg': 0.2, 'possession_adjusted_interceptions': -0.05},
    'buildup': {'received_pass_avg': 0.25, 'accurate_passes_percent': 0.2, 'successful_pass_entries_f3_90': 0.2, 'received_long_rate': -0.15, 'crosses_per_rp': -0.15, 'deep_completed_pass_avg': -0.05},
    'deep_inv': {'received_long_rate': 0.3, 'touch_in_box_avg': 0.25, 'shots_avg': 0.1, 'non_penalty_xg_shot': 0.1, 'foul_suffered_avg': 0.1, 'forward_pass_rate': -0.15},
    'prog': {'progressive_run_avg': 0.15, 'passes_avg': 0.05, 'successful_pass_entries_f3_90': 0.1, 'accurate_passes_to_final_third_percent': 0.05, 'successful_progressive_pass_percent': 0.15, 'progressive_pass_avg': 0.2, 'progressive_pass_rate': 0.2, 'accurate_passes_percent': 0.1},
    'duel': {'defensive_duels_avg': 0.15, 'defensive_duels_won': 0.25, 'aerial_duels_avg': 0.1, 'aerial_duels_won': 0.25, 'possession_adjusted_interceptions': -0.15, 'forward_pass_rate': 0.1},
    'agg': {'possession_adjusted_interceptions': 0.5, 'shot_block_avg': 0.25, 'fouls_avg': 0.25}
}

def calculate_and_normalize_ratings(z_scores, weights):
    ratings = (z_scores * list(weights.values())).sum(axis=1)
    positive_ratings = ratings - ratings.min() + 1
    transformed_ratings, _ = stats.boxcox(positive_ratings)

    # Normalize ratings to a scale from 0 to 100
    normalized_ratings = (transformed_ratings - transformed_ratings.min()) / \
                         (transformed_ratings.max() - transformed_ratings.min()) * 100

    return normalized_ratings

def ratings_df(template_players_data):
    # Extract the list of variables for each role from weights_dict
    roles_vars = {role: list(weights.keys()) for role, weights in weights_dict.items()}

    vars_deep_striker = roles_vars['deep_striker']
    vars_poacher = roles_vars['poacher']
    vars_target = roles_vars['target']
    vars_drive = roles_vars['drive']
    vars_b2b = roles_vars['b2b']
    vars_ball_winner = roles_vars['ball_winner']
    vars_playmaker = roles_vars['playmaker']
    vars_off = roles_vars['off']
    vars_creator = roles_vars['creator']
    vars_deep = roles_vars['deep']
    vars_wide = roles_vars['wide']
    vars_inside = roles_vars['inside']
    vars_overlap = roles_vars['overlap']
    vars_buildup = roles_vars['buildup']
    vars_deep_inv = roles_vars['deep_inv']
    vars_prog = roles_vars['prog']
    vars_duel = roles_vars['duel']
    vars_agg = roles_vars['agg']

    # Extract relevant columns from template_players_data
    template_deep_striker = template_players_data[vars_deep_striker]
    template_poacher = template_players_data[vars_poacher]
    template_target = template_players_data[vars_target]
    template_b2b = template_players_data[vars_b2b]
    template_drive = template_players_data[vars_drive]
    template_ball_winner = template_players_data[vars_ball_winner]
    template_playmaker = template_players_data[vars_playmaker]
    template_off = template_players_data[vars_off]
    template_creator = template_players_data[vars_creator]
    template_deep = template_players_data[vars_deep]
    template_wide = template_players_data[vars_wide]
    template_inside = template_players_data[vars_inside]
    template_overlap = template_players_data[vars_overlap]
    template_buildup = template_players_data[vars_buildup]
    template_deep_inv = template_players_data[vars_deep_inv]
    template_prog = template_players_data[vars_prog]
    template_duel = template_players_data[vars_duel]
    template_agg = template_players_data[vars_agg]

    # Calculate z-scores for each role
    z_scores_deep_striker = template_deep_striker.apply(zscore)
    z_scores_poacher = template_poacher.apply(zscore)
    z_scores_target = template_target.apply(zscore)
    z_scores_b2b = template_b2b.apply(zscore)
    z_scores_drive = template_drive.apply(zscore)
    z_scores_ball_winner = template_ball_winner.apply(zscore)
    z_scores_playmaker = template_playmaker.apply(zscore)
    z_scores_off = template_off.apply(zscore)
    z_scores_creator = template_creator.apply(zscore)
    z_scores_deep = template_deep.apply(zscore)
    z_scores_wide = template_wide.apply(zscore)
    z_scores_inside = template_inside.apply(zscore)
    z_scores_overlap = template_overlap.apply(zscore)
    z_scores_buildup = template_buildup.apply(zscore)
    z_scores_deep_inv = template_deep_inv.apply(zscore)
    z_scores_prog = template_prog.apply(zscore)
    z_scores_duel = template_duel.apply(zscore)
    z_scores_agg = template_agg.apply(zscore)

    # Calculate and normalize ratings for each role using the weights from the dictionary
    rating_deep_striker = calculate_and_normalize_ratings(z_scores_deep_striker, weights_dict['deep_striker'])
    rating_poacher = calculate_and_normalize_ratings(z_scores_poacher, weights_dict['poacher'])
    rating_target = calculate_and_normalize_ratings(z_scores_target, weights_dict['target'])
    rating_b2b = calculate_and_normalize_ratings(z_scores_b2b, weights_dict['b2b'])
    rating_drive = calculate_and_normalize_ratings(z_scores_drive, weights_dict['drive'])
    rating_ball_winner = calculate_and_normalize_ratings(z_scores_ball_winner, weights_dict['ball_winner'])
    rating_playmaker = calculate_and_normalize_ratings(z_scores_playmaker, weights_dict['playmaker'])
    rating_off = calculate_and_normalize_ratings(z_scores_off, weights_dict['off'])
    rating_creator = calculate_and_normalize_ratings(z_scores_creator, weights_dict['creator'])
    rating_deep = calculate_and_normalize_ratings(z_scores_deep, weights_dict['deep'])
    rating_wide = calculate_and_normalize_ratings(z_scores_wide, weights_dict['wide'])
    rating_inside = calculate_and_normalize_ratings(z_scores_inside, weights_dict['inside'])
    rating_overlap = calculate_and_normalize_ratings(z_scores_overlap, weights_dict['overlap'])
    rating_buildup = calculate_and_normalize_ratings(z_scores_buildup, weights_dict['buildup'])
    rating_deep_inv = calculate_and_normalize_ratings(z_scores_deep_inv, weights_dict['deep_inv'])
    rating_prog = calculate_and_normalize_ratings(z_scores_prog, weights_dict['prog'])
    rating_duel = calculate_and_normalize_ratings(z_scores_duel, weights_dict['duel'])
    rating_agg = calculate_and_normalize_ratings(z_scores_agg, weights_dict['agg'])

    # Create a DataFrame with ratings and roles
    df_ratings = pd.DataFrame({
        'Player': template_players_data['full_name'],
        'Team': template_players_data['last_club_name'],
        'Deep-Lying Striker': rating_deep_striker,
        'Poacher': rating_poacher,
        'Target': rating_target,
        'Box-to-Box': rating_b2b,
        'Ball-Winner': rating_ball_winner,
        'Forward Drive': rating_drive,
        'Deep-Lying Playmaker': rating_playmaker,
        'Shadow Striker/9,5': rating_off,
        'Advanced Creator': rating_creator,
        'Deep Threat': rating_deep,
        'Wide Winger': rating_wide,
        'Inside Forward': rating_inside,
        'Wide Overlap': rating_overlap,
        'Inside/Build-up': rating_buildup,
        'Deep Outlet': rating_deep_inv,
        'Progressor': rating_prog,
        'Man-to-Man': rating_duel,
        'Aggressor': rating_agg
    })

    return df_ratings

# # st.image(db_player['image'].values[0])
# if not db_player['transfermarkt_player'].isna().iloc[0]:
#     tm_name = db_player['transfermarkt_player'].iloc[0]
# else:
#     tm_name = db_player['full_name'].iloc[0]

# Define the team names that should have the brightness filter applied
dark_logo_teams = ['Helmond Sport', 'Gent', 'OH Leuven', 'RWD Molenbeek', 'Zulte-Waregem', 'Club Brugge II', 'Standard Liège II',
                    'Metz', 'Hradec Králové', 'Lokeren-Temse']

# Check if the current team name is in the dark logo teams list
if db_player['last_club_name'].iloc[0] in dark_logo_teams:
    brightness_filter = "brightness(1.8);"  # Apply brightness filter
else:
    brightness_filter = ""  # No brightness filter applied

# # Custom CSS styles for tabs
# custom_css = f"""
# <style>
#     /* Change text color on hover */
#     .stTabs [data-baseweb="tab"]:hover {{
#         color: {team_color};
#     }}

#     /* Text color for selected tab */
#     .stTabs [aria-selected="true"] {{
#         color: {team_color};
#     }}

#     /* Change text color on hover for data-baseweb="tab-highlight" */
#     .stTabs [data-baseweb="tab-highlight"] {{
#         background-color: {team_color};
#     }}

#     /* Adjust padding and margins for the tab container */
#     .stTabs {{
#         padding-top: 0px; /* Adjust top padding */
#         padding-bottom: 20px; /* Adjust bottom padding */
#     }}

# </style>
# """

# Extracting variables
image_url = db_player['logo'].values[0]
birth_date_str = db_player['birth_date'].iloc[0]
if not pd.isna(birth_date_str):
    # Convert birth_date_str to a datetime object
    birth_date = datetime.strptime(birth_date_str, '%Y-%m-%d')
    today = datetime.today()
    # Calculate the player's age
    player_age = today.year - birth_date.year - ((today.month, today.day) < (birth_date.month, birth_date.day))
    # Format birth_date
    birth_date = birth_date.strftime('%d-%m-%Y')
else:
    # Handle cases where birth_date_str is NaN, None, or not valid
    player_age = None
    birth_date = 'NA'

passport = db_player['passport_country_names'].iloc[0]
passport = [country.strip().strip("'").strip('"') for country in passport.strip('[]').split(',')]
passport = ', '.join(passport)
wysc_id = db_player['id'].iloc[0]
template = db_player['template_1'].iloc[0]
template_disp = pos_disp.get(template, "Unknown Position")
image_info = db_player['image'].values[0]
player_name = db_player['full_name'].values[0]
min_played = int(db_player['minutes_on_field'].values[0])
season = db_player['season_id'].values[0]
selected_player_data = db_player
selected_league = db_player['competition'].iloc[0]

# Number input for minimum minutes threshold
min_minutes_perc = st.sidebar.number_input('Minutes Threshold (%)', min_value=0, value=20, step=1)

# Get templates for the selected player
template_1 = db_player['template_1'].iloc[0]
template_2 = db_player['template_2'].iloc[0]

# Selectbox for choosing template with template_1 as default
template_options = [template_1]
if template_2 is not None and template_2 != template_1:
    template_options.append(template_2)

selected_template = st.sidebar.selectbox('Switch Template', template_options, index=0)

max_minutes = db_comp['minutes_on_field'].max()

# Calculate the minimum minutes threshold based on the percentage of max_minutes
min_minutes = int((min_minutes_perc / 100) * max_minutes)

template_players_data = db_comp[
    ((db_comp['template_1'] == selected_template)) & 
    (db_comp['minutes_on_field'] >= min_minutes)
]

selected_player_name = selected_player_data['full_name'].iloc[0]
# Check if the player's data is already included, if not, add it
player_data = db_comp[db_comp['full_name'] == selected_player_name]
# Check if the player's data is not already included in the filtered data
if not player_data['template_1'].values[0] == selected_template:
    template_players_data = pd.concat([template_players_data, player_data])

template_players_data_ratings = db_comp[
    ((db_comp['template_1'].isin(['DM', 'CM']))) & 
    (db_comp['minutes_on_field'] >= min_minutes)
]

# Define template-role mappings
template_role_mapping = {
    'CB': ['Progressor', 'Man-to-Man', 'Aggressor'],
    'FB': ['Wide Overlap', 'Inside/Build-up', 'Deep Outlet'],
    'DM': ['Deep-Lying Playmaker', 'Ball-Winner'],
    'CM': ['Shadow Striker/9,5', 'Advanced Creator', 'Box-to-Box', 'Deep-Lying Playmaker', 'Ball-Winner'],
    'W': ['Deep Threat', 'Wide Winger', 'Inside Forward'],
    'ST': ['Deep-Lying Striker', 'Poacher', 'Target']
}

tab1, tab2, tab3 = st.tabs(['Player Card', 'Glossary', 'Scheduling Tool'])

with tab1:

    # Calculate the percentage of minutes played by the selected player
    pctmin = int(db_player['minutes_on_field'].iloc[0] / max_minutes * 100)
    # Check if the player meets the minutes threshold
    if pctmin > min_minutes_perc:

        # Determine roles to display based on the selected template
        roles_to_display = template_role_mapping.get(selected_template, [])

        df_ratings = ratings_df(template_players_data)

        # Add a column to identify the selected player
        df_ratings['Selected'] = df_ratings['Player'] == selected_player_name
        df_ratings_filtered = df_ratings[['Player', 'Team', 'Selected'] + roles_to_display]

        # Step 1: Filter the DataFrame to get the row for the selected player
        selected_player_row = df_ratings_filtered[df_ratings_filtered['Selected'] == True]
        role_columns = df_ratings_filtered.columns.difference(['Player', 'Team', 'Selected'])
        main_role = selected_player_row[role_columns].idxmax(axis=1).values[0]

        # Melt the DataFrame to long format
        swarm_data = pd.melt(
            df_ratings_filtered,
            id_vars=["Player", "Team", "Selected"],
            var_name="Role",
            value_name="Rating"
        )

        # Create a Plotly scatter plot for non-selected players
        fig2 = px.scatter(
            swarm_data[swarm_data['Selected'] == False],
            x="Role",
            y="Rating",
            color="Selected",
            hover_name="Player",
            labels={"Rating": "Rating (0-100)"},
            template="plotly_dark"
        )

        # Add the trace for selected players with different size and marker
        fig2.add_trace(
            px.scatter(
                swarm_data[swarm_data['Selected'] == True],
                x="Role",
                y="Rating",
                color="Selected",
                hover_name="Player",
                labels={"Rating": "Rating (0-100)"},
                template="plotly_dark"
            ).update_traces(marker=dict(size=14, symbol='star', color='gold'), showlegend=False).data[0]
        )

        # Customize the layout and set a constant marker size for non-selected players
        fig2.update_layout(
            margin=dict(l=25, r=25, t=10, b=0),
            height=250,
            xaxis_title="",
            yaxis_title="",
            showlegend=False,
            plot_bgcolor="black",
            paper_bgcolor="black",
            font=dict(color="white"),
            yaxis=dict(gridcolor="#444444", tickfont=dict(size=9)),
            xaxis=dict(tickfont=dict(size=9))
        )

        fig2.update_traces(marker=dict(size=8, sizemode='diameter', color='#a31818'), selector=dict(marker_symbol='circle'))

        # Add annotations for each selected player
        for i, row in swarm_data[swarm_data['Selected'] == True].iterrows():
            fig2.add_annotation(
                x=row['Role'],
                y=row['Rating'],
                text=f"{row['Rating']:.0f}",  # Display the rating number with no decimals
                showarrow=False,
                font=dict(color="yellow", size=11),
                xshift=20,  # Shift text slightly to the right of the marker
                yshift=0,   # Align text with the marker vertically
                align="center",
            )

        # Add annotation for the player with the highest rating in each role
        for role in swarm_data['Role'].unique():
            highest_rated = swarm_data[swarm_data['Role'] == role].sort_values(by='Rating', ascending=False).iloc[0]
            # yshift = 20 if role in ["Advanced Creator", "Deep-Lying Playmaker", "Inside/Build-up", "Wide Winger"] else 10
            fig2.add_annotation(
                x=highest_rated['Role'],
                y=highest_rated['Rating'],
                text=highest_rated['Player'],  # Display the player's name
                showarrow=False,
                font=dict(color="lightgray", size=9),
                xshift=0,  # Shift text slightly to the right of the marker
                yshift=10, # Shift text slightly above the marker
                align="center",
                textangle=0
            )

        # Create a subset with the passing variables
        passing_data = template_players_data[['full_name', 'last_club_name', 'passes_avg', 
                                            'accurate_passes_percent', 'received_pass_avg', 
                                            'received_long_rate', 'long_pass_rate', 
                                            'forward_pass_rate', 'progressive_pass_rate']]

        # Calculate Z-Scores for the numerical columns
        numerical_columns = ['passes_avg', 'accurate_passes_percent', 'received_pass_avg',
                            'received_long_rate', 'long_pass_rate', 'forward_pass_rate', 
                            'progressive_pass_rate']

        # Calculate the z-scores and create a new DataFrame with them
        z_scores = passing_data[numerical_columns].apply(zscore)

        # Combine the original passing data with the z-scores
        passing_data_z = pd.concat([passing_data[['full_name', 'last_club_name']], z_scores], axis=1)

        # Define labels for the y-axis
        labels = ["VOLUME", "ACCURACY (%)", "RECEIVED", "RECEIVED LONG RATE", "LONG RATE", "FORWARD RATE", "PROGRESSIVE RATE"]

        # Extract the player's data (assuming you want to plot for a specific player)
        player_passing = passing_data_z.loc[passing_data_z['full_name'] == selected_player_name].iloc[0]

        # Plot
        fig, ax = plt.subplots(figsize=(6.67, 4))

        # Draw custom horizontal lines from -2 to 2.5
        for i in range(len(labels)):
            ax.hlines(y=i, xmin=-2.5, xmax=2.7, color='#595959', linestyle='-.', linewidth=1.3)

        # Loop through each metric and plot
        for i, label in enumerate(labels):
            ax.plot(player_passing[numerical_columns[::-1]].iloc[i], i, 'o', color='#a31818', markersize=12)  # Player's dot
            # ax.plot([0], i, 'o', color='pink', markersize=10, alpha=0.5)  # Team average (set to 0 for now)
            ax.axvline(0, color='gray', linestyle='--')  # League average line (z-score of 0)
            
            # Place the raw value next to the y-axis (on the left side)
            raw_value = passing_data.loc[passing_data['full_name'] == player_name, numerical_columns[::-1][i]].values[0]
            ax.text(-2.9, i - 0.04, f"{raw_value:.2f}", va='center', ha='right', color='white', fontsize=11, fontweight='bold')

        # Set y-axis labels and make them white
        ax.set_yticks(range(len(labels[::-1])))
        ax.set_yticklabels(labels[::-1], color='white', fontweight='bold')

        # Set limits and labels
        ax.set_xlim(-3.5, 3)  # Adjust limits based on expected z-scores
        ax.set_ylim(-0.5, len(labels) - 0.5)
        ax.set_xlabel("")
        # ax.set_title('PASSING STYLE', fontsize=16, fontweight='bold', color='white')
        ax.set_facecolor('#000000')

        # Set plot background color
        fig.patch.set_facecolor('#000000')

        # Add subtitle and notes
        fig.text(0.4, 0.95, "z-scores | dotted line = league average for position", 
                ha='center', fontsize=11, color='white', fontstyle='italic')

        img_data3 = save_radar_chart_as_png(fig)
        # Encode the image to base64 to display in HTML
        encoded_img3 = base64.b64encode(img_data3).decode()

        # Generate image of radar chart
        img_data = save_radar_chart_as_png(display_radar_chart(selected_player_data, template_players_data, selected_template, selected_league))
        # Encode the image to base64 to display in HTML
        encoded_img = base64.b64encode(img_data).decode()

        npg = int(db_player['non_penalty_goal'].iloc[0])
        ast = int(db_player['assists'].iloc[0])

        # Initialize the dashboard layout
        layout = [
            dashboard.Item("profile_card", x=0, y=0, w=10, h=1.5),
            dashboard.Item("radar_chart", x=0, y=1.5, w=5.25, h=3.8),
            dashboard.Item("swarm_plot", x=6, y=1.5, w=4.75, h=1.85),
            dashboard.Item("passing_profile", x=6, y=3.5, w=4.75, h=1.95)
        ]

        # Render the elements within the dashboard
        with elements("dashboard"):
            with dashboard.Grid(layout, draggableHandle=".draggable", resizable=True):
                
                # Profile Card
                with mui.Paper(key="profile_card", style={
                    "padding": "7px", 
                    "textAlign": "center", 
                    "border": "2px solid white",  
                    "borderRadius": "4px", 
                    "backgroundColor": "black"  
                }):
                    
                    # Header with Team Name and Logo
                    mui.Typography(variant="h5", style={"color": "white", "fontFamily": "Bahnschrift"})(
                        f"{player_name} ",
                        mui.IconButton(
                            html.img(
                                src=image_url,
                                style={
                                    "verticalAlign": "middle",
                                    "width": "45px",
                                    "filter": brightness_filter
                                }
                            )
                        )
                    )
                    
                    # Create a grid container for the profile image and details
                    with mui.Grid(container=True, spacing=-1.2, alignItems="center"):
                        
                        # Grid item for the profile image (left side)
                        with mui.Grid(item=True, xs=3):  # Adjust xs to control width
                            mui.Avatar(
                                src=image_info,
                                style={"width": "100px", "height": "100px", "margin": "5px auto", "marginBottom": "10px",
                                            "border": "3px solid #4f4f4f",  "borderRadius": "50%"}
                            )

                            # Define emojis
                            football_emoji = "᳂" 
                            boot_emoji = "➾" 
                            
                            # Use mui.Box to align the emojis and values
                            with mui.Box(style={"textAlign": "center", "color": "white", "marginTop": "10px"}):
                                # Display football emoji with the number of goals
                                with mui.Box(style={"display": "inline-flex", "alignItems": "center", "marginRight": "15px"}):
                                    mui.Typography(variant="body1", style={"fontFamily": "Bahnschrift", "fontSize": "20px", "marginRight": "8px"})(football_emoji)
                                    mui.Typography(variant="body1", style={"fontFamily": "Bahnschrift", "fontSize": "15px"})(npg)
                                
                                # Display boot emoji with the number of assists
                                with mui.Box(style={"display": "inline-flex", "alignItems": "center"}):
                                    mui.Typography(variant="body1", style={"fontFamily": "Bahnschrift", "fontSize": "20px", "marginRight": "8px"})(boot_emoji)
                                    mui.Typography(variant="body1", style={"fontFamily": "Bahnschrift", "fontSize": "15px"})(ast)

                        # Grid item for the profile text (right side)
                        with mui.Grid(item=True, xs=4.3):  # Adjust xs to control width
                            
                            # NATIONALITY row
                            with mui.Box(style={"display": "flex", "alignItems": "center"}):
                                mui.Typography(variant="h6", style={"color": "gray", "minWidth": "200px", "fontSize": "15px", "fontFamily": "Bahnschrift"})("NATIONALITY")
                                mui.Typography(variant="h5", style={"textAlign": "left", "fontSize": "15px", "fontFamily": "Bahnschrift"})(" " + passport)
                            
                            # DATE OF BIRTH row
                            with mui.Box(style={"display": "flex", "alignItems": "center"}):
                                mui.Typography(variant="h6", style={"color": "gray", "minWidth": "200px", "fontSize": "15px", "fontFamily": "Bahnschrift"})("DATE OF BIRTH")
                                mui.Typography(variant="h6", style={"textAlign": "left", "fontSize": "15px","fontFamily": "Bahnschrift"})(f"{birth_date} ({player_age})")
                            
                            # POSITION row
                            with mui.Box(style={"display": "flex", "alignItems": "center"}):
                                mui.Typography(variant="h6", style={"color": "gray", "minWidth": "200px", "fontSize": "15px", "fontFamily": "Bahnschrift"})("POSITION")
                                mui.Typography(variant="h6", style={"textAlign": "left", "fontSize": "15px", "fontFamily": "Bahnschrift"})(" " + template_disp)

                        # New Grid item for additional information (right side)
                        with mui.Grid(item=True, xs=4.4):
                            with mui.Box(style={"display": "flex", "alignItems": "center"}):
                                mui.Typography(variant="h6", style={"color": "gray", "minWidth": "200px", "fontSize": "15px", "fontFamily": "Bahnschrift"})("SEASON")
                                mui.Typography(variant="h6", style={"textAlign": "left", "fontSize": "15px", "fontFamily": "Bahnschrift"})(" " + str(season))

                            with mui.Box(style={"display": "flex", "alignItems": "center"}):
                                mui.Typography(variant="h6", style={"color": "gray", "minWidth": "200px", "fontSize": "15px", "fontFamily": "Bahnschrift"})("MINUTES PLAYED")
                                mui.Typography(variant="h6", style={"textAlign": "left", "fontSize": "15px", "fontFamily": "Bahnschrift"})(" " + str(min_played))

                            with mui.Box(style={"display": "flex", "alignItems": "center"}):
                                mui.Typography(variant="h6", style={"color": "gray", "minWidth": "200px", "fontSize": "15px", "fontFamily": "Bahnschrift"})("MAIN ROLE")
                                mui.Typography(variant="h6", style={"textAlign": "left", "fontSize": "15px", "fontFamily": "Bahnschrift"})(" " + str(main_role))

                # Radar Chart (Display the saved image)
                with mui.Paper(key="radar_chart", elevation=3, style={"padding": "2px", "backgroundColor": "#ffffff"}):
                    html.img(src=f"data:image/png;base64,{encoded_img}", style={"width": "100%", "height": "100%"})

                # Save the figure as an HTML string
                html_str = fig2.to_html(include_plotlyjs='cdn')
                # Create a base64-encoded data URL for the iframe
                html_data_url = f"data:text/html;base64,{base64.b64encode(html_str.encode()).decode()}"

                # Swarm Plot (Display the Plotly chart as an iframe)
                with mui.Paper(key="swarm_plot", elevation=3, style={"padding": "0px", "backgroundColor": "#ffffff"}):
                                html.h4("Role Profile Style", style={"textAlign": "center", "margin": "0px", "color": "#000000", "lineHeight": "18px",
                                                                    "padding": "5px 0px 0px 0px", "fontFamily": "Bahnschrift", "fontSize": "14px",
                                                                    "backgroundColor": "#e6e6e6"})
                                html.iframe(src=html_data_url, style={"width": "100%", "height": "100%", "border": "none",
                                                                    "transform": "scale(1.025)"}
                    )
                                
                # Radar Chart (Display the saved image)
                with mui.Paper(key="passing_profile", elevation=3, style={"padding": "2px", "backgroundColor": "#ffffff"}):
                    html.h4("Passing Style", style={"textAlign": "center", "margin": "0px", "color": "#000000", "lineHeight": "18px",
                                            "padding": "4px 0px 4px 0px", "fontFamily": "Bahnschrift", "fontSize": "14px",
                                            "backgroundColor": "#e6e6e6"})
                    html.img(src=f"data:image/png;base64,{encoded_img3}", style={"width": "99.5%", "height": "91%"})
        
        display_name = db_player['name'].iloc[0]

    #     def generate_pdf(encoded_img, encoded_img3, fig2):
    #         # Create a BytesIO buffer to hold the PDF data
    #         buffer = BytesIO()

    #         # Create a PDF document with landscape orientation (A4 size)
    #         pdf = canvas.Canvas(buffer, pagesize=landscape(A4))

    #         # Set background color to dark
    #         pdf.setFillColorRGB(14 / 255, 17 / 255, 23 / 255)
    #         pdf.rect(0, 0, 842, 595, fill=True)
            
    #         # Draw a line
    #         pdf.setStrokeColorRGB(0.639, 0.094, 0.094)  # Set line color to white
    #         pdf.line(0, 514, 100, 595)

    #         # Add FC Utrecht logo to the top-left corner
    #         pdf.drawImage('logo-fcu.png', 30, 530, width=40, height=40, preserveAspectRatio=True)

    #         # Add title to the PDF
    #         title = f"{display_name}"
    #         pdf.setFillColorRGB(255, 255, 255)
    #         pdf.setFont("Helvetica-Bold", 16)

    #         # Calculate the width of the title text
    #         title_width = pdf.stringWidth(title, "Helvetica-Bold", 16)

    #         # Set the X position to center the title
    #         title_x_position = 120  # Adjust as needed for overall page layout
    #         pdf.drawString(title_x_position, 560, title)

    #         # Add the subtitle centered under the title
    #         subtitle = season
    #         pdf.setFillColorRGB(0.6, 0.6, 0.6)
    #         pdf.setFont("Helvetica-Bold", 12)

    #         # Calculate the width of the subtitle text
    #         subtitle_width = pdf.stringWidth(subtitle, "Helvetica-Bold", 12)
    #         # Calculate the X position to center the subtitle under the title
    #         subtitle_x_position = title_x_position + (title_width - subtitle_width) / 2

    #         # Draw the subtitle
    #         pdf.drawString(subtitle_x_position, 544, subtitle)

    #         # Set font and color for the labels (e.g., "NATIONALITY", "DATE OF BIRTH")
    #         label_font = "Helvetica-Bold"
    #         label_font_size = 10
    #         label_color = (0.6, 0.6, 0.6)  # Gray color

    #         # Set font and color for the values (e.g., the actual nationality, date of birth)
    #         value_font = "Helvetica-Bold"
    #         value_font_size = 10
    #         value_color = (1, 1, 1)  # White color

    #         # Set initial position for the first row of text
    #         x_position = 350
    #         y_position = 570

    #         # NATIONALITY row
    #         pdf.setFont(label_font, label_font_size)
    #         pdf.setFillColorRGB(*label_color)
    #         pdf.drawString(x_position, y_position, "NATIONALITY")

    #         pdf.setFont(value_font, value_font_size)
    #         pdf.setFillColorRGB(*value_color)
    #         pdf.drawString(x_position + 110, y_position, passport)

    #         # Move to the next row
    #         y_position -= 15

    #         # DATE OF BIRTH row
    #         pdf.setFont(label_font, label_font_size)
    #         pdf.setFillColorRGB(*label_color)
    #         pdf.drawString(x_position, y_position, "DATE OF BIRTH")

    #         pdf.setFont(value_font, value_font_size)
    #         pdf.setFillColorRGB(*value_color)
    #         pdf.drawString(x_position + 110, y_position, birth_date)

    #         # Move to the next row
    #         y_position -= 15

    #         # POSITION row
    #         pdf.setFont(label_font, label_font_size)
    #         pdf.setFillColorRGB(*label_color)
    #         pdf.drawString(x_position, y_position, "TEMPLATE")

    #         pdf.setFont(value_font, value_font_size)
    #         pdf.setFillColorRGB(*value_color)
    #         pdf.drawString(x_position + 110, y_position, pos_disp.get(selected_template))

    #         # Draw a line underneath the title
    #         pdf.setStrokeColorRGB(0.639, 0.094, 0.094)
    #         pdf.line(90, 520, 800, 520)

    #         # Set text color to white
    #         pdf.setFillColorRGB(1, 1, 1)  # RGB values are from 0 to 1

    #         # Decode the base64 encoded image
    #         radar_chart_data = base64.b64decode(encoded_img)
    #         radar_chart_io = BytesIO(radar_chart_data)
    #         radar_chart_image = ImageReader(radar_chart_io)

    #         # Draw the radar chart image on the PDF
    #         pdf.drawImage(radar_chart_image, -180, 70, width=800, height=400, preserveAspectRatio=True)

    #         passing_chart_data = base64.b64decode(encoded_img3)
    #         passing_chart_io = BytesIO(passing_chart_data)
    #         passing_chart_image = ImageReader(passing_chart_io)

    #         # Add title to the PDF
    #         passing_title = "Passing Style"
    #         pdf.setFillColorRGB(255, 255, 255)
    #         pdf.setFont("Helvetica-Bold", 14)
    #         pdf.drawString(600, 270, passing_title)

    #         pdf.drawImage(passing_chart_image, 450, 70, width=350, height=200, preserveAspectRatio=True)

    #         # Add title to the PDF
    #         passing_title = "Role Profle"
    #         pdf.setFillColorRGB(255, 255, 255)
    #         pdf.setFont("Helvetica-Bold", 14)
    #         pdf.drawString(608, 493, passing_title)

    #         # Create a BytesIO buffer to save the image
    #         img_buffer = BytesIO()
    #         # Export the figure as a PNG image to the buffer
    #         fig2.write_image(img_buffer, format="png", engine="kaleido", width=500, height=300)
    #         # Ensure the buffer's position is at the beginning
    #         img_buffer.seek(0)

    #         # Convert the BytesIO buffer to an ImageReader object
    #         plotly_image = ImageReader(img_buffer)

    #         # Draw the Plotly figure image on the PDF
    #         pdf.drawImage(plotly_image, 450, 290, width=350, height=200, preserveAspectRatio=False)

    #         # Set the stroke color for the line (e.g., white)
    #         pdf.setStrokeColorRGB(0.5, 0.5, 0.5) 
    #         pdf.setLineWidth(1)
    #         pdf.setDash(3, 2)  # 3 units on, 2 units off
    #         pdf.line(430, 20, 430, 510)
    #         pdf.setDash(1, 0)

    #         # Finalize the PDF
    #         pdf.showPage()
    #         pdf.save()

    #         # Get the PDF data from the buffer
    #         buffer.seek(0)
    #         return buffer.getvalue()

    #     # Placeholder for the base64 encoded radar chart image
    #     encoded_img = encoded_img
    #     encoded_img3 = encoded_img3
    #     fig2 = fig2

    #     selected_player_team = selected_player_data['last_club_name'].iloc[0]
    #     selected_player_team = ' '.join(selected_player_team.split())
    #     today = datetime.today()
    #     today_fm = today.strftime("%d-%m-%Y")
    #     naming_conv = f"{selected_player_name}_{selected_player_team}_{season}_{today_fm}.pdf"

    #     # Button to generate and download the PDF in one step
    #     st.download_button(label="Download PDF Report", 
    #                         data=generate_pdf(encoded_img, encoded_img3, fig2), 
    #                         file_name=naming_conv, 
    #                         mime='application/pdf')
        
    # else:
    #     # Display an error message if the player has not played enough minutes
    #     thresh_min = int(selected_player_data['minutes_on_field'].iloc[0])
    #     st.error(f"{selected_player_data['full_name'].iloc[0]} has not played enough minutes to have a representative data profile according to the selected minute threshold.")
    #     st.error(f"{selected_player_data['full_name'].iloc[0]} has played {thresh_min} minutes "
    #                 f"({pctmin:.0f}% of possible minutes). Please change the minute input.")
        
with tab2:
    # Display a warning or introductory note (optional)
    st.markdown("""
    **Role Profile Scores aim to give a 
    <span style='color:#a31818'>classification</span> of the player; it's 
    <span style='color:#a31818'>not definitive</span> about quality.**
    """, unsafe_allow_html=True)

    # Conditionally display roles based on the value of `template_disp`
    if template_disp == "Centre-Back":
        
        st.markdown("""
        **Progressor**:  
        Central defenders that progress the ball upfield, both by dribbling and passing. They consistently move the ball into the final third and have above-average quality in these efforts. Possession-oriented.  
        **Type**: Olivier Boscagli, Daley Blind, Nico Schlotterbeck.
        """)
        
        st.markdown("""
        **Man-to-Man**:  
        Central defenders that focus on playing man-to-man (ground- and aerial) duels. They have a high volume/number of defensive actions on the pitch & win a lot. They will tend to stay in their defensive line and don't step forward too often.  
        **Type**: Mike van der Hoorn, Éder Militão, Dante.
        """)
        
        st.markdown("""
        **Aggressor**:  
        Central defenders that chase the ball/opponent down. They'll commit into tackles, fouls & blocks more often than their counterparts.  
        **Type**: Ryan Flamingo, Marcos Senesi, Cristian Romero.
        """)

    elif template_disp == "Full-Back":
        
        st.markdown("""
        **Wide Overlap**:  
        Full-backs that will tend to get to the wide assist zone a lot and can drive forward in possession. They are focused on getting the ball into the box in a more direct/crossing way.  
        **Type**: Souffian El Karouani, Alejandro Grimaldo, Pedro Porro.
        """)
        
        st.markdown("""
        **Inside/Build-up**:  
        Full backs that will be involved in first-phase possession (act as a +1) and are able to build-up tidily. They are more focused on transporting the ball upfield than to be offensively active.  
        **Type**: Lutsharel Geertruida, Joško Gvardiol, Ben White.
        """)
        
        st.markdown("""
        **Deep Outlet**:  
        Full backs that receive the ball high and are an outlet in possession for long passes. They'll drive forward mainly without the ball and also get into the opposition box. They don't tend to offer a lot in forward passing. Often wing-backs.  
        **Type**: Yan Couto, Denzel Dumfries, Thomas Meunier.
        """)

    elif template_disp in ["Defensive Midfield", "Centre-Midfield"]:

        # Display Deep-Lying Playmaker and Ball-Winner roles for both Defensive Midfield and Centre-Midfield
        st.markdown("""
        **Deep-Lying Playmaker**:  
        Defensive/central midfielders that sit at the base, receive a lot of (short) passes and help distribute the play forward. Engine in possession. They help in progressing upfield and generally target the final third successfully. Defensively they'll mainly intercept passes.  \n**Type**: Joey Veerman, Yves Bissouma, Rodri.
        """)

        st.markdown("""
        **Ball-Winner**:  
        Defensive/central midfielders whose primary aim is recycling possession. They'll win a lot of possession by dueling and intercepting and then pass it to the more creative players. They generally have a high pass accuracy due to the low-risk nature.  \n**Type**: Alonzo Engwanda, Azor Matusiwa, N'Golo Kanté.        
        """)

        # Display additional roles only if Centre-Midfield is selected
        if template_disp == "Centre-Midfield":
            st.markdown("""
            **Box-to-Box**:  
            Central midfielders that move from box to box and do work in both ends. They'll get into the opposition box/to chances and have the ability to drive/transition with the ball. Generally physically strong. They don't add too much creative passing.  \n**Type**: Quinten Timber, Dominik Szoboszlai, Andy Diouf.        
            """)    

            st.markdown("""
            **Shadow Striker/9,5**:  
            Central midfielders that mainly operate in attacking/higher zones and will try to find space without the ball. Their main threat is in the box - getting to many and good quality chances. They generally don't drop in possession to be available/move the ball to the final third.  \n**Type**: Guus Til, Thomas Müller, Joelinton.        
            """)    

            st.markdown("""
            **Advanced Creator**:  
            Central midfielders that are the higher creative hub in the team. Their aim is to set up the offense, get the ball into the box and create chances. Generally have good individual skill and risk more in their passing.  \n**Type**: Malik Tillman, James Maddison, Jamal Musiala.        
            """)  

    elif template_disp == "Winger":

        st.markdown("""
        **Inside Forward**:  
        Wingers that are involved in play and help the offense move into the final third. They'll be active in possession/short combinations, are skillful 1v1 and combine towards the box, instead of the more direct wing/crossing play.  \n**Type**: Osame Sahraoui, Lamine Yamal, Michael Olise.        
        """)  

        st.markdown("""
        **Deep Threat**:  
        Wingers that mainly run deep and try to receive the ball high/behind the last line. They get into the box and to the end of chances. But offer less in general possession/forward passing. Often correlated with higher pace.  \n**Type**: Hirving Lozano, Bradley Barcola, Serge Gnabry.                    
        """)  

        st.markdown("""
        **Wide Winger**:  
        Wingers that play a more traditional wing role, staying wide and aimed at (early) direct crossing. They'll try to target the box and create chances. Generally have a lower passing accuracy/overall involvement.  \n**Type**: Mohammed Kudus, Kingsley Coman, Sheraldo Becker.       
        """) 

    elif template_disp == "Striker":

        st.markdown("""
        **Target**:  
        Strikers that are used as a target; they receive a lot of long passes, are involved aerially & draw more fouls. In possession they tend to hold up play/pass short, so that the rest of the team can join the offense. Won't see a lot of 1v1 action.  \n**Type**: Tobias Lauritsen, Dominic Calvert-Lewin, Alexander Sørloth.        
        """)  

        st.markdown("""
        **Deep-Lying Striker**:  
        Strikers that are dropping away from the '9' position to be involved in play. They are often more of a facilitator than a pure goalscorer and will combine shortly/into the box, trying to create chances.  \n**Type**: Alexander Isak, Harry Kane, Gerard Moreno.        
        """)  

        st.markdown("""
        **Poacher**:  
        Strikers in the purest sense of the word; being there at the end of moves. They have most of their touches in the box, relatively. And the quality of the chances they get to is high. Does not necessarily mean they don't offer more in possession.  \n**Type**: Erling Haaland, Luis Suárez, Robert Lewandowski.        
        """)   

    else:
        st.error("No roles available for the selected player position.")

with tab3:

    # Initialize variables
    countries = ["Netherlands", "Netherlands", "Netherlands", "Belgium", "Belgium", "France", "Germany", "Germany", "Germany", "Denmark", "Denmark", "Norway", "Sweden"]
    leagues = ["eredivisie", "keuken-kampioen-divisie", "u21-divisie-1-herbst", "jupiler-pro-league", "proximus-league", "ligue-2",
               "2-bundesliga", "3-liga", "u19-dfb-nachwuchsliga-vorrunde-gruppe-3", "superligaen", "nordicbet-liga", "eliteserien", "allsvenskan"]
    league_codes = ["NL1", "NL2", "211F", "BE1", "BE2", "FR2", "L2", "L3", "19D3", "DK1", "DK2", "NO1", "SE1"]
    seasons = [2024] * 11 + [2023] * 2
    
    # Function to scrape data for multiple leagues
    def scrape_games(countries, leagues, league_codes, seasons):
        all_games = []
        
        for country, league, league_code, season in zip(countries, leagues, league_codes, seasons):
            url = f"https://www.transfermarkt.nl/{league}/gesamtspielplan/wettbewerb/{league_code}/saison_id/{season}"
            headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'}
            response = requests.get(url, headers=headers)
            page = BeautifulSoup(response.content, 'html.parser')
            
            dates = [date.get_text().strip() for date in page.select("td.hide-for-small:nth-child(1)")]
            home_teams = [team.get_text().strip() for team in page.select(".no-border-rechts.hauptlink a")]
            away_teams = [team.get_text().strip() for team in page.select(".no-border-links.hauptlink a")]
            times = [time.get_text().strip() for time in page.select("td.zentriert.hide-for-small")]
            
            # Create a DataFrame for the scraped data
            match_data = pd.DataFrame({
                'League': [league] * len(dates),
                'Country': [country] * len(dates),
                'Date': dates,
                'Time': times,
                'HomeTeam': home_teams,
                'AwayTeam': away_teams
            })
            
            all_games.append(match_data)
        
        # Combine all data into a single DataFrame
        combined_data = pd.concat(all_games, ignore_index=True)
        return combined_data
    
    # Function to clean and parse the date, while tracking the last valid date
    def clean_and_parse_date(date_str, last_valid_date=[None]):
        if pd.isna(date_str) or date_str.strip() == '':
            return last_valid_date[0]
        
        # Remove the day-of-the-week prefix and any extra whitespace
        parts = date_str.strip().split(' ', 1)
        if len(parts) > 1:
            cleaned_date_str = parts[1].strip()
            try:
                parsed_date = parser.parse(cleaned_date_str, dayfirst=True).date()
                last_valid_date[0] = parsed_date
                return parsed_date
            except ValueError:
                return last_valid_date[0]
        
        return last_valid_date[0]
    
    # Function to clean data
    def clean_data(match_data):
        match_data['Date'] = match_data['Date'].apply(lambda x: clean_and_parse_date(x))
        match_data['Time'] = match_data['Time'].apply(lambda x: x.strip() if x else None)
        match_data['Time'] = match_data['Time'].replace('', None).ffill()
        match_data['League'] = match_data['League'].str.replace('-', ' ').str.title()
        match_data['Game'] = match_data['HomeTeam'] + ' - ' + match_data['AwayTeam']

        return match_data
    
    # Caching the data loading and processing function
    @st.cache_data
    def load_and_process_data(countries, leagues, league_codes, seasons):
        # Scrape data for multiple leagues
        match_data = scrape_games(countries, leagues, league_codes, seasons)
        match_data = clean_data(match_data)

        # Define the replacements as a dictionary
        league_names = {
            'Keuken Kampioen Divisie': 'KKD',
            'Proximus League': 'Challenger Pro League',
            'Nordicbet Liga': '1. Division',
            'U21 Divisie 1 Herbst': 'O21 Divisie 1',
            '2 Bundesliga': '2. Bundesliga',
            '3 Liga': '3. Liga',
            'U19 Dfb Nachwuchsliga Vorrunde Gruppe 3': 'A-Junioren'
        }

        # Replace multiple values using the dictionary
        match_data['League'] = match_data['League'].replace(league_names)

        # Filter to include only games from today and beyond
        today = datetime.today().date()
        match_data = match_data[match_data['Date'] >= today].reset_index(drop=True)

        return match_data
    
    # Streamlit UI
    if 'match_data' not in st.session_state:
        st.session_state['match_data'] = None

    # Initialize `selected_df` as an empty DataFrame early
    selected_df = pd.DataFrame()

    if st.button("Load Games"):
        # Load and process the data
        st.session_state['match_data'] = load_and_process_data(countries, leagues, league_codes, seasons)

    if st.session_state['match_data'] is not None:
        match_data = st.session_state['match_data']

        # Get unique leagues for the multiselect
        unique_leagues = match_data['League'].unique()

        # Multiselect for filtering
        selected_leagues = st.multiselect('Leagues', unique_leagues, default=unique_leagues)

        # Filter the data based on the selected leagues
        if selected_leagues:
            selected_df = match_data[match_data['League'].isin(selected_leagues)]
        else:
            st.write("Please select a league.")
    else:
        st.write("Please load the games first.")

    # Only proceed if `selected_df` is not empty
    if not selected_df.empty:
        # Function to get flag URL based on country name
        def get_flag_url(country):
            country_flags = {
                'Belgium': 'https://upload.wikimedia.org/wikipedia/commons/1/15/Twemoji12_1f1e7-1f1ea.svg',
                'Netherlands': 'https://upload.wikimedia.org/wikipedia/commons/b/b8/Twemoji12_1f1f3-1f1f1.svg',
                'Denmark': 'https://upload.wikimedia.org/wikipedia/commons/6/6d/Twemoji12_1f1e9-1f1f0.svg',
                'Norway': 'https://upload.wikimedia.org/wikipedia/commons/3/3b/Twemoji12_1f1e7-1f1fb.svg',
                'Sweden': 'https://upload.wikimedia.org/wikipedia/commons/e/e7/Twemoji12_1f1f8-1f1ea.svg',
                'France': 'https://upload.wikimedia.org/wikipedia/commons/0/02/Twemoji_1f1eb-1f1f7.svg',
                'Germany': 'https://upload.wikimedia.org/wikipedia/commons/8/80/Twemoji_1f1e9-1f1ea.svg'
            }
            return country_flags.get(country, '')

        # Group by Date
        match_data_sorted = selected_df.sort_values(by=['Date', 'Time']).reset_index()
        grouped = match_data_sorted.groupby('Date')

        # Define the fixed card size
        card_width = 180
        card_height = 75

        # Set the maximum number of columns (cards) per row
        max_columns = 5

        # Include Font Awesome CDN
        st.markdown(
            '<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0-beta3/css/all.min.css">',
            unsafe_allow_html=True
        )

        # Iterate over each date group
        for date, games in grouped:
            # Format the date as "Day of the week, dd-mm-yyyy"
            formatted_date = f"{date.strftime('%A')}, {date.strftime('%d-%m-%Y')}"
            
            # Display the formatted date as a header with custom font, size, and border
            st.markdown(
                f"""
                <div style="
                    font-family: Bahnschrift, sans-serif; 
                    font-size: 16px; 
                    margin-bottom: 10px;
                    padding-bottom: 5px; 
                    border-bottom: 1px solid lightgray;">
                    {formatted_date}
                </div>
                """, 
                unsafe_allow_html=True
            )
            
            # Create a container for each date's games
            with st.container():
                # Split the games into chunks of max_columns
                chunks = [games.iloc[i:i + max_columns] for i in range(0, len(games), max_columns)]
                
                for chunk in chunks:
                    # Always create max_columns columns, but only fill them as needed
                    cols = st.columns(max_columns)
                    
                    for i, (index, row) in enumerate(chunk.iterrows()):
                        flag_url = get_flag_url(row['Country'])
                        with cols[i]:
                            # HTML and CSS to style the card with fixed size and flag
                            st.markdown(
                                f"""
                                <div style="
                                    background-color: black; 
                                    width: {card_width}px;
                                    height: {card_height}px;
                                    padding: 10px; 
                                    margin: 10px 0; 
                                    border: 2px solid lightgrey;
                                    border-radius: 10px; 
                                    box-shadow: 0px 4px 6px rgba(0, 0, 0, 0.1); 
                                    text-align: center;
                                    display: flex;
                                    flex-direction: column;
                                    align-items: center;
                                    justify-content: center;
                                    position: relative;">
                                    <img src="{flag_url}" style="
                                        position: absolute; 
                                        top: 3px; 
                                        right: 7px; 
                                        width: 12px; 
                                        height: auto;
                                        border-radius: 3px;">
                                    <div style="font-weight: bold; font-size: 11px; font-family: Bahnschrift; margin-top: 5px;">
                                        {row['Game']}
                                    </div>
                                    <div style="font-size: 11px; font-weight: bold; font-family: Bahnschrift; color: gray; margin-top: 3px; text-align: center;">
                                        <i class="fas fa-clock" style="margin-right: 2px;"></i> {row['Time']}
                                    </div>
                                </div>
                                """, 
                                unsafe_allow_html=True
                            )