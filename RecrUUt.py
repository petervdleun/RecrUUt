import sqlite3
import pandas as pd
import streamlit as st
from PIL import Image
import requests
from io import BytesIO
import plotly.graph_objs as go
import streamlit_shadcn_ui as ui
from streamlit_option_menu import option_menu
from unidecode import unidecode
from datetime import datetime
import textwrap
import pycountry
import gzip

st.set_page_config(layout='wide')

def decompress_db_file(db_path):
    decompressed_path = db_path.replace('.gz', '')
    with gzip.open(db_path, 'rb') as f_in:
        with open(decompressed_path, 'wb') as f_out:
            f_out.write(f_in.read())
    return decompressed_path

def connect_db():
    db_path = 'data/wysc.db.gz'
    decompressed_path = decompress_db_file(db_path)
    return sqlite3.connect(decompressed_path)

# Load players data
@st.cache_data
def load_players():
    with connect_db() as conn:
        query = 'SELECT * FROM players'
        return pd.read_sql_query(query, conn)
    
# Load players data
@st.cache_data
def load_logos():
    with connect_db() as conn:
        query = 'SELECT * FROM logos'
        return pd.read_sql_query(query, conn)
    
# Load leagues data
@st.cache_data
def load_leagues():
    with connect_db() as conn:
        query = 'SELECT * FROM leagues'
        return pd.read_sql_query(query, conn)
    
# Load leagues data and create the mapping dictionary
leagues_df = load_leagues()
leagues_df['competition_id'] = leagues_df['competition_id'].astype(str)
leagues_dict = {
    row['competition_id']: {
        'competition_name': row['competition_name'],
        'country_name': row['country_name']
    }
    for _, row in leagues_df.iterrows()
}

# Load player stats
@st.cache_data
def load_player_stats(player_id):
    with connect_db() as conn:
        query = '''
        SELECT season_id, competition_id, last_club_name, primary_position, minutes_on_field, goals, assists, non_penalty_goal, extraction_timestamp
        FROM player_stats 
        WHERE player_id = ?
        '''
        return pd.read_sql_query(query, conn, params=(player_id,))

# Load player percentiles with optional filters
@st.cache_data
def load_player_percentiles(player_id=None, filters=None):
    with connect_db() as conn:
        query = 'SELECT * FROM percentiles'
        params = []
        if player_id:
            query += ' WHERE player_id = ?'
            params.append(player_id)
        elif filters:
            conditions = []
            if 'season_id' in filters:
                conditions.append("season_id = ?")
                params.append(filters['season_id'])
            if 'template' in filters:
                conditions.append("template = ?")
                params.append(filters['template'])
            if 'competition_id' in filters:
                conditions.append("competition_id = ?")
                params.append(filters['competition_id'])
            for key, value in filters.items():
                if key not in ['season_id', 'template', 'competition_id']:
                    conditions.append(f"{key} >= ?")
                    params.append(value)
            query += ' WHERE ' + ' AND '.join(conditions)
        return pd.read_sql_query(query, conn, params=params)

# Function to load role_scores from the wysc.db database
@st.cache_data
def load_role_scores(player_id):
    with connect_db() as conn:
        query = '''
            SELECT *
            FROM role_scores
            WHERE player_id = ?
        '''
        return pd.read_sql_query(query, conn, params=(player_id,))
    
# Function to join players and logos on team name
def logo_join():
    # Load players and logos data
    players_df = load_players()
    logos_df = load_logos()

    # Merge the two DataFrames on the current_team_name and team_name columns
    merged_df = pd.merge(players_df, logos_df, left_on='current_team_name', right_on='team_name', how='left')

    return merged_df
    
def add_to_shadow_list(player_id):
    with connect_db() as conn:
        cursor = conn.cursor()
        # Insert player_id and current date
        cursor.execute(
            '''
            INSERT INTO shadow_list (player_id, date_added)
            VALUES (?, ?)
            ''',
            (player_id, datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        )
        conn.commit()

def get_country_code(country_name):
    """Map country name to ISO Alpha-2 country code using pycountry."""
    try:
        return pycountry.countries.lookup(country_name).alpha_2.lower()
    except LookupError:
        return None

def display_player_profile(player_id, players_df):
    player_data = players_df[players_df['player_id'] == player_id].iloc[0]
    player_image_url = player_data['image_url']
    player_name = player_data['full_name']
    birth_date = player_data['birth_date']
    birth_country = player_data['birth_country_name']

    # Calculate formatted birth date and age
    if birth_date:
        birth_date_obj = datetime.strptime(birth_date, "%Y-%m-%d")
        formatted_birth_date = birth_date_obj.strftime("%d-%m-%Y")
        
        # Calculate age
        today = datetime.today()
        age = today.year - birth_date_obj.year - ((today.month, today.day) < (birth_date_obj.month, birth_date_obj.day))
        formatted_birth_date = f"{formatted_birth_date} ({age})"
    else:
        formatted_birth_date = "N/A"

    # Get the ISO Alpha-2 code for the flag
    country_code = get_country_code(birth_country)
    flag_url = f"https://flagcdn.com/w40/{country_code}.png" if country_code else None

    with st.container():
        st.markdown(
            """
            <style>
            .profile-card {
                display: flex;
                align-items: center;
                background-color: #0e1117;
                padding: 25px;
                margin-top: -15px;
                margin-bottom: 10px;
                color: white;
            }
            .profile-image {
                width: 100px;
                height: 100px;
                border-radius: 50%;
                object-fit: cover;
                margin-right: 20px;
                border: 3px solid #4f4f4f;
            }
            .profile-details h3 {
                margin: 0 0 -12px 0;
                font-size: 24px;
                display: flex;
                align-items: center;
            }
            .profile-details h3 img {
                margin-left: 10px;
                width: 24px;
                height: 16px;
                object-fit: cover;
            }
            .profile-details p {
                margin: -3px 0 0 15px;
                font-size: 16px;
            }
            </style>
            """,
            unsafe_allow_html=True
        )

        # Generate the HTML with the flag
        flag_html = f'<img src="{flag_url}" alt="{birth_country} flag" />' if flag_url else ''

        st.markdown(
            f"""
            <div class="profile-card">
                <img src="{player_image_url}" alt="Player Image" class="profile-image" />
                <div class="profile-details">
                    <h3>{player_name} {flag_html}</h3>
                    <p>{formatted_birth_date}</p>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

        # # Button to add player to shadow list
        # trigger_btn = ui.button(
        #     text="+ add to shadow list",
        #     variant="ghost",
        #     size="sm",
        #     key="trigger_btn"
        # )

        # # Perform the action directly if the button is clicked
        # if trigger_btn:
        #     add_to_shadow_list(player_id)  # Add player to the shadow list
        #     st.toast(f"{player_name} has been added to the Shadow List!", icon="✅")

# Function to display player information with team logo
def display_player_information(player_stats_df, leagues_dict):
    # Function to generate flag HTML based on country name
    def get_flag_html(country_name):
        country_code = get_country_code(country_name)  # Function to get ISO Alpha-2 code
        flag_url = f"https://flagcdn.com/w40/{country_code}.png" if country_code else None
        return f'<img src="{flag_url}" alt="{country_name} flag" style="width: 16px; height: 11px; vertical-align: middle; margin-bottom: 3px; margin-left: 10px;">' if flag_url else ''

    # Map competition_id to "competition_name" with flag image
    player_stats_df['competition_name'] = player_stats_df['competition_id'].astype(str).map(
        lambda x: (
            f"{leagues_dict[x]['competition_name']}"
            f"{get_flag_html(leagues_dict[x]['country_name'])}"
            if x in leagues_dict else "Unknown Competition"
        )
    )

    # Select relevant columns for display, including goals and assists
    player_info_df = player_stats_df[['season_id', 'competition_name', 'last_club_name', 'minutes_on_field', 'goals', 'assists', 'non_penalty_goal']]
    # Rename columns for clarity
    player_info_df.columns = ['Season', 'Competition', 'Club', 'Minutes Played', 'Goals', 'Assists', 'Non-Penalty Goals']

    # Sort by season (descending) and minutes played (descending)
    player_info_df = player_info_df.sort_values(by=['Season', 'Minutes Played'], ascending=[False, False])

    # Merge player data with logos data
    merged_data = logo_join()

    col1, col2 = st.columns([0.65, 0.35])

    with col1:
        # Display the title
        st.markdown(
            """
            <div style="color: #ff7e82; font-size: 18px; font-weight: bold; margin-bottom: 8px;">
                CAREER OVERVIEW
            </div>
            """,
            unsafe_allow_html=True
        )

        # Display title row with only icons for columns
        st.markdown(
            """
            <div style="display: flex; justify-content: space-between; background-color: #0e1117; padding: 10px; border-radius: 5px; font-size: 15px; margin-top: -15px;">
                <div style="flex: 0.5;"></div>
                <div style="flex: 1.6;"></div>
                <div style="flex: 2;"></div>
                <div style="flex: 0.55; display: flex; justify-content: center; align-items: center;">
                    <img src="https://img.icons8.com/ios-filled/50/ffffff/clock--v1.png" alt="Clock Icon" style="width: 16px; height: 16px;">
                </div>
                <div style="flex: 0.2; display: flex; justify-content: center; align-items: center;">
                    <img src="https://img.icons8.com/?size=100&id=9770&format=png&color=FFFFFF" alt="Goals Icon" style="width: 16px; height: 16px;">
                </div>
                <div style="flex: 0.5; display: flex; justify-content: center; align-items: center;">
                    <img src="https://img.icons8.com/ios-filled/50/ffffff/handshake-heart.png" alt="Assists Icon" style="width: 16px; height: 16px;">
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

        # Display each row as a compact row-card, including logos and new columns
        st.markdown("<div style='display: flex; flex-direction: column;'>", unsafe_allow_html=True)
        for _, row in player_info_df.iterrows():
            # Get the logo URL for the team from merged_data
            logo_url = merged_data[merged_data['current_team_name'] == row['Club']]['logo_url'].values[0] if len(merged_data[merged_data['current_team_name'] == row['Club']]) > 0 else None

            st.markdown(
                f"""
                <div style="display: flex; justify-content: space-between; background-color: #1e1e1e; padding: 8px; margin-top: 5px; border-radius: 5px; font-size: 14px">
                    <div style="color: #ff7e82; flex: 0.5;"><strong>{row['Season']}</strong></div>
                    <div style="color: white; flex: 1.6;">{row['Competition']}</div>
                    <div style="color: white; flex: 2; display: flex; align-items: center;">
                        <div style="margin-right: 10px;">{row['Club']}</div>
                        {f'<img src="{logo_url}" alt="Team Logo" style="width: 20px; height: 20px; border-radius: 50%;">' if logo_url else ''}
                    </div>
                    <div style="color: white; flex: 0.45;">{row['Minutes Played']}'</div>
                    <div style="color: white; flex: 0.3;">{row['Non-Penalty Goals']}</div>
                    <div style="color: white; flex: 0.3;">{row['Assists']}</div>
                </div>
                """,
                unsafe_allow_html=True
            )
        st.markdown("</div>", unsafe_allow_html=True)

        # Add the annotation for the icon
        st.markdown(
            f"""
            <div style="margin-top: 10px; text-align: left; font-size: 11px; font-style: italic; color: white;">
                <img src="https://img.icons8.com/?size=100&id=9770&format=png&color=FFFFFF" alt="Non-Penalty Goals Icon" style="width: 15px; height: 15px; margin-right: 5px; vertical-align: middle;">
                Non-Penalty Goals
            </div>
            """,
            unsafe_allow_html=True
        )
        st.markdown("</div>", unsafe_allow_html=True)

# Mapping templates to their specific variables
template_variable_map = {
    'CB': ['defduels_perc', 'defduelswon_perc', 'aerial_avg_perc', 'aerialwon_perc', 'recvpass_perc', 'passacc_perc', 'progpassrate_perc', 'succfwdpass_perc', 'comp_f3_perc', 'possadjintercept_perc', 'shotblock_perc', 'fouls_perc'],
    'FB': ['npxg_perc', 'touchbox_perc', 'recvpass_perc', 'passacc_perc', 'deepcomp_perc', 'comp_box_perc', 'crosses_rp_perc', 'xa_per_rp_perc', 'progrun_perc', 'possadj_defduels_perc', 'defduelswon_perc', 'possadjintercept_perc'],
    'DM': ['possadj_defduels_perc', 'defduelswon_perc', 'aerial_avg_perc', 'aerialwon_perc', 'possadjintercept_perc', 'recvpass_perc', 'passacc_perc', 'progpassrate_perc', 'comp_f3_perc', 'progrun_perc', 'comp_drib_perc'], 
    'CM': ['npxg_perc', 'shots_perc', 'npxg_shot_perc', 'touchbox_perc', 'recvpass_perc', 'passacc_perc', 'comp_f3_perc', 'comp_box_perc', 'xa_per_rp_perc', 'progrun_perc','comp_drib_perc', 'possadjintercept_perc', 'possadj_defduels_perc', 'defduelswon_perc'],
    'W': ['npxg_perc', 'shots_perc', 'npxg_shot_perc', 'touchbox_perc', 'recvpass_perc', 'deepcomp_perc', 'comp_f3_perc', 'comp_box_perc', 'xa_per_rp_perc', 'comp_drib_perc', 'progrun_perc', 'succdefactions_perc'],
    'ST': ['npxg_perc', 'shots_perc', 'npxg_shot_perc', 'xg_conv_perc', 'touchbox_perc', 'recvpass_perc', 'comp_f3_perc', 'comp_box_perc', 'xa_per_rp_perc', 'comp_drib_perc', 'aerialwon_perc', 'succdefactions_perc']
}

# Dictionary mapping percentile column names to descriptive names for further use
percentile_labels = {
    "xg_perc": "xG",
    "xa_perc": "xA",
    "shots_perc": "Shots",
    "sot_perc": "Shots on Target (%)",
    "touchbox_perc": "Box Touches",
    "passes_perc": "Passes",
    "passacc_perc": "Pass Accuracy (%)",
    "fwdpass_perc": "Forward Passes",
    "longpass_perc": "Long Passes",
    "passfinalthird_perc": "Passes to Final Third",
    "passaccfinalthird_perc": "Passes to Final Third Accuracy (%)",
    "progpass_perc": "Progressive Passes",
    "succfwdpass_perc": "Forward Pass Accuracy (%)",
    "succlongpass_perc": "Long Passes Accuracy (%)",
    "shotassist_perc": "Shot Assists",
    "deepcomp_perc": "Deep Completions",
    "deepcross_perc": "Deep Completed Crosses",
    "cross_perc": "Crosses",
    "crossacc_perc": "Cross Accuracy (%)",
    "recvpass_perc": "Received Passes",
    "recvlongpass_perc": "Received Long Passes",
    "drib_perc": "Dribbles",
    "succdrib_perc": "Dribble Completion (%)",
    "progrun_perc": "Progressive Runs",
    "accel_perc": "Accelerations",
    "succdefactions_perc": "Successful Defensive Actions",
    "defduels_perc": "Defensive Duels",
    "possadj_defduels_perc": "PAdj Defensive Duels",
    "defduelswon_perc": "Defensive Duels Won (%)",
    "aerial_avg_perc": "Aerial Duels",
    "aerialwon_perc": "Aerial Duels Won (%)",
    "possadjintercept_perc": "PAdj Interceptions",
    "possadj_tackle_perc": "PAdj Tackles",
    "fouls_perc": "Fouls (Inverted)",
    "shotblock_perc": "Shot Blocks",
    "npxg_perc": "Non-Penalty xG",
    "npxg_shot_perc": "Non-Penalty xG per Shot",
    "crosses_rp_perc": "Crosses per 10 Received Passes",
    "comp_crosses_perc": "Completed Crosses",
    "comp_drib_perc": "Completed Dribbles",
    "recvlongrate_perc": "Received Long Rate (%)",
    "fwdpassrate_perc": "Forward Pass Rate (%)",
    "longpassrate_perc": "Long Pass Rate (%)",
    "progpassrate_perc": "Progressive Pass Rate (%)",
    "xa_avg_perc": "xA per 90",
    "xa_per_rp_perc": "xA per 10 Received Passes",
    "comp_f3_perc": "Pass Entries Final Third",
    "comp_box_perc": "Pass Entries Box",
    "xg_conv_perc": "xG Conversion"
}

def ordinal_suffix(n):
    # Helper function to get ordinal suffix for a given integer
    if 10 <= n % 100 <= 20:
        suffix = 'th'
    else:
        suffix = {1: 'st', 2: 'nd', 3: 'rd'}.get(n % 10, 'th')
    return f"{n}{suffix}"

def display_pizza_chart(player_id, percentiles_df, season_id, competition_id, template, key=None):
    # Define stat categories using original keys, not formatted labels
    stat_categories = {
        'Finishing': {
            'stats': ['npxg_perc', 'shots_perc', 'npxg_shot_perc', 'touchbox_perc', 'xg_conv_perc'],
            'color': '#FFB6A0'
        },
        'Distribution': {
            'stats': ['recvpass_perc', 'passacc_perc', 'comp_f3_perc', 'comp_box_perc', 'deepcomp_perc', 'progpassrate_perc', 'succfwdpass_perc'],
            'color': '#FFF2CC'
        },
        'Creativity': {
            'stats': ['xa_per_rp_perc', 'comp_drib_perc', 'progrun_perc', 'crosses_rp_perc'],
            'color': '#EAD1DC'
        },
        'Defending': {
            'stats': ['succdefactions_perc', 'possadjintercept_perc', 'defduels_perc', 'defduelswon_perc', 'aerial_avg_perc',
                      'aerialwon_perc', 'shotblock_perc', 'fouls_perc', 'possadj_defduels_perc'],
            'color': '#CFE2F3'
        }
    }

    # Helper function to get category and color based on original stat key
    def get_category_color(stat_key):
        for category, details in stat_categories.items():
            if stat_key in details['stats']:
                return details['color']
        return 'salmon'  # Default color if stat not found in predefined categories

    # Helper function to wrap and center-align labels with `textwrap`
    def format_label(label):
        wrapped_label = textwrap.fill(label, width=16)  # Adjust width for line length
        return wrapped_label.replace("\n", "<br>")  # Replace newlines with HTML line breaks for Plotly

    # Corrected filtering line
    season_template_data = percentiles_df[
        (percentiles_df['season_id'] == season_id) & 
        (percentiles_df['competition_id'] == competition_id) & 
        (percentiles_df['template'] == template)
    ]
    
    if not season_template_data.empty:
        selected_template_data = season_template_data.iloc[0]
        player_name = selected_template_data['full_name']

        # Get specific variables for the template
        variables = template_variable_map.get(template, [])
        values = [round(selected_template_data.get(var, 0)) for var in variables]  # Default to 0 if variable is missing

        # Map each variable to its formatted label using `percentile_labels` and apply wrapping
        categories = [format_label(percentile_labels.get(var, var.replace('_perc', ''))) for var in variables]
        
        # Set up hover labels with correct ordinal suffixes
        hover_labels = [f"{ordinal_suffix(value)} percentile for {category}" for category, value in zip(categories, values)]

        # Create the pizza chart
        pizza_chart = go.Figure()
        angle_step = 360 / len(categories) if categories else 0  # Avoid division by zero
        
        for i, (var, value, label) in enumerate(zip(variables, values, hover_labels)):
            color = get_category_color(var)  # Use original variable name to fetch color
            pizza_chart.add_trace(go.Barpolar(
                r=[value],
                theta=[i * angle_step],
                width=[angle_step],
                hoverinfo="text",
                text=label,
                marker=dict(
                    color=color,
                    line=dict(color='black', width=1.5)
                ),
                opacity=0.75,
                showlegend=False  # Exclude these traces from the legend
            ))

        # Add dummy traces for each category to create the legend
        for category_name, details in stat_categories.items():
            pizza_chart.add_trace(go.Barpolar(
                r=[0],  # Zero radius so these won't show up on the chart itself
                theta=[0],  # Arbitrary angle
                marker=dict(color=details['color']),
                showlegend=True,
                name=category_name
            ))

        # Dynamically adjust the radial axis range based on data
        radial_range = [-6, 100]

        # Update chart layout with centered title and labels
        competition_info = leagues_dict.get(str(competition_id), {})
        competition_name = competition_info.get('competition_name', "Unknown Competition")
        title_text = f"<span style='font-size:20px'>{player_name}</span><br><span style='font-size:15px; color:gray'>{season_id} • {competition_name} • {selected_template_data['minutes_on_field']} minutes </span>"
        pizza_chart.update_layout(
            title=dict(
                text=title_text,
                x=0.47,  # Center the title
                y=0.95,  # Position it slightly above the chart
                xanchor='center',
                yanchor='top',
                font=dict(size=15, color='white')  # Customize title font
            ),
            polar=dict(
                angularaxis=dict(
                    tickmode='array',
                    tickvals=[i * angle_step for i in range(len(categories))],
                    ticktext=categories,
                    direction='clockwise',
                    ticks='',
                    tickfont=dict(size=10, color='white', weight='bold')                ),
                radialaxis=dict(
                    visible=True,
                    range=radial_range,
                    gridcolor='#636363',
                    gridwidth=1,
                    griddash='dashdot',
                    showline=False,
                    showticklabels=False,
                    ticks=''
                ),
                bgcolor='#0e1117'
            ),
            paper_bgcolor='#0e1117',
            plot_bgcolor='#0e1117',
            showlegend=True,
            margin=dict(b=30, t=80, l=40)
        )

        # Display the pizza chart
        st.plotly_chart(pizza_chart)

    else:
        st.write("No percentile data available for this player in the selected season and competition.")

@st.cache_data
def get_comparison_options(players_df, leagues_dict):
    """
    Generate options for the comparison selectbox by merging percentiles data with league information
    and returning player names with additional metadata (seasons, templates, and competitions).
    """
    with connect_db() as conn:
        # Query to fetch unique player data from the percentiles table
        query = '''
        SELECT DISTINCT player_id, full_name, template, season_id, competition_id
        FROM percentiles
        '''
        percentiles_data = pd.read_sql_query(query, conn)

    # Convert leagues_dict to a DataFrame for merging
    leagues_df = pd.DataFrame.from_dict(leagues_dict, orient='index').reset_index()
    leagues_df.rename(columns={'index': 'competition_id'}, inplace=True)

    # Merge percentiles data with leagues data
    merged_data = percentiles_data.merge(leagues_df, on='competition_id', how='left')

    # Generate comparison options and metadata
    merged_data['option'] = merged_data.apply(
        lambda row: f"{row['full_name']} • {row['season_id']} • {row['template']} • "
                    f"{row.get('competition_name', 'Unknown Competition')} "
                    f"({row.get('country_name', 'Unknown Country')})",
        axis=1
    )

    comparison_options = merged_data['option'].tolist()
    option_metadata = merged_data[['player_id', 'season_id', 'template', 'competition_id']].to_dict('records')

    return comparison_options, option_metadata

def render_comparison_selectboxes(players_df, leagues_dict, key_suffix=""):
    """
    Render the comparison selectbox and return the selected metadata for the comparison player.
    """
    # Fetch the comparison options and metadata
    comparison_options, option_metadata = get_comparison_options(players_df, leagues_dict)

    # Add a default "Select a Player" option
    comparison_options = ["Select a Player"] + comparison_options

    # Single column layout for the selectbox
    col1 = st.columns(1)[0]

    with col1:
        # Display the combined selectbox with a unique key
        selected_comparison_option = st.selectbox(
            "Select Player and Season-Template-League Combination for Comparison",
            options=comparison_options,
            key=f"comparison_combined_selectbox_{key_suffix}"  # Unique key
        )

    # Handle default selection
    if selected_comparison_option == "Select a Player":
        return None

    # Find metadata for the selected option
    selected_metadata = option_metadata[comparison_options.index(selected_comparison_option) - 1]
    return selected_metadata

# Function to dynamically add a filter with default values
def add_filter():
    # Ensure there’s a default column to add
    all_columns = list(percentile_labels.keys())
    next_column = all_columns[len(st.session_state.percentile_filters) % len(all_columns)]
    st.session_state.percentile_filters.append({'column': next_column, 'value': 50})
    st.session_state['filter_added'] = True  # Set a flag to indicate a filter was added

# Function to remove a filter by index
def remove_filter(index):
    if len(st.session_state.percentile_filters) > 1:  # Ensure at least one filter remains
        st.session_state.percentile_filters.pop(index)

def player_search():
    st.header("Player Search")
    
    # Initialize session state for percentile filters
    if 'percentile_filters' not in st.session_state:
        st.session_state.percentile_filters = [{'column': list(percentile_labels.keys())[0], 'value': 50}]
    
    # Load percentile, player, and logo data
    percentiles_df = load_player_percentiles()
    players_df = load_players()
    logos_df = load_logos()
    
    # Merge players with logos for later use
    players_with_logos = players_df.merge(
        logos_df, 
        left_on='current_team_name', 
        right_on='team_name', 
        how='left',
        suffixes=('', '_logo')
    )

    # Filters for season, template, and competition
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        available_seasons = percentiles_df['season_id'].unique()
        selected_season = st.selectbox("Select Season", available_seasons)
    with col2:
        # Manual order for templates
        template_options = ['CB', 'FB', 'DM', 'CM', 'W', 'ST']
        selected_template = st.selectbox("Select Template", template_options, index=template_options.index('CM'))
    with col3:
        # Prepare competition options as "country - competition", sorted alphabetically
        available_competitions = percentiles_df['competition_id'].unique()
        competition_options = sorted([
            f"{leagues_dict[comp_id]['country_name']} - {leagues_dict[comp_id]['competition_name']}"
            for comp_id in available_competitions
        ])
        selected_competition_display = st.selectbox(
            "Select Competition", 
            competition_options, 
            index=competition_options.index("Netherlands - Eredivisie")
        )
        
        # Reverse-map selected competition display to competition_id
        selected_competition_id = next(
            comp_id for comp_id, data in leagues_dict.items()
            if f"{data['country_name']} - {data['competition_name']}" == selected_competition_display
        )
    with col4:
        # Get unique age values and sort them
        players_df['age'] = pd.to_datetime('today').year - pd.to_datetime(players_df['birth_date']).dt.year
        unique_ages = sorted(players_df['age'].dropna().astype(int).unique())
        age_options = ["No Selection"] + [str(age) for age in unique_ages]
        selected_age = st.selectbox("Select Maximum Age", age_options)
    
    # Dynamic percentile filter section
    st.write("---")

    # Button to add new filters
    if st.button("Add Filter", key="add_filter_btn"):
        st.session_state.percentile_filters.append({'column': list(percentile_labels.keys())[0], 'value': 50})

    # Render existing filters
    for i, filter in enumerate(st.session_state.percentile_filters):
        cols = st.columns([1.2, 0.7, 0.4])
        
        # Dropdown for column selection with labels
        with cols[0]:
            selected_label = percentile_labels[filter['column']]
            selected_column = st.selectbox(
                "Filter Column",
                list(percentile_labels.values()),
                index=list(percentile_labels.values()).index(selected_label),
                key=f"percentile_column_{i}"
            )
            actual_column = next(key for key, value in percentile_labels.items() if value == selected_column)
            st.session_state.percentile_filters[i]['column'] = actual_column
            
        # Number input for setting percentile value
        with cols[1]:
            st.session_state.percentile_filters[i]['value'] = st.number_input(
                "Minimum Percentile",
                min_value=0, max_value=100,
                value=filter['value'],
                step=5,
                key=f"percentile_value_{i}"
            )
        
        # Remove filter button
        with cols[2]:
            st.markdown('<div class="align-center">', unsafe_allow_html=True)
            st.button("−", on_click=remove_filter, args=(i,), key=f"remove_filter_{i}", help="Remove this filter")
            st.markdown('</div>', unsafe_allow_html=True)

    # Map filters for actual use
    filters = {
        'season_id': selected_season,
        'template': selected_template,
        'competition_id': selected_competition_id
    }
    for filter in st.session_state.percentile_filters:
        filters[filter['column']] = filter['value']

    # Apply filters to load relevant players
    filtered_players = load_player_percentiles(filters=filters)

    # Merge filtered players with `players_df` to get age and other details
    filtered_players = filtered_players.merge(
        players_df[['player_id', 'birth_date', 'full_name', 'current_team_name']], 
        on='player_id', 
        how='left',
        suffixes=('', '_player')  # Avoid _x and _y; add '_player' suffix if there's overlap
    )

    # Handle duplicate `current_team_name` columns
    if 'current_team_name_player' in filtered_players.columns:
        filtered_players.rename(columns={'current_team_name_player': 'current_team_name'}, inplace=True)

    # Apply age filter
    if selected_age != "No Selection":
        max_age = int(selected_age)
        filtered_players['age'] = pd.to_datetime('today').year - pd.to_datetime(filtered_players['birth_date']).dt.year
        filtered_players = filtered_players[filtered_players['age'] <= max_age]

    # Merge filtered players with logos
    filtered_players = filtered_players.merge(
        players_with_logos[['player_id', 'logo_url', 'current_team_name']], 
        on='player_id', 
        how='left',
        suffixes=('', '_logo')
    )

    # Handle duplicate `current_team_name` columns from logos merge
    if 'current_team_name_logo' in filtered_players.columns:
        filtered_players.drop(columns=['current_team_name_logo'], inplace=True)

    # TEMPORARY FIX: Remove players without a club
    filtered_players = filtered_players[filtered_players['current_team_name'].notna()]

    # Calculate index score (average of selected percentile filters)
    selected_columns = [f['column'] for f in st.session_state.percentile_filters]
    filtered_players['index_score'] = filtered_players[selected_columns].mean(axis=1)

    # Sort by index score in descending order
    filtered_players = filtered_players.sort_values(by='index_score', ascending=False)

    # Display the filtered players using row-cards
    st.markdown("<div style='display: flex; flex-direction: column;'>", unsafe_allow_html=True)  # Open container

    # Header (if missing or needs alignment consistency)
    headers_html = f"""
    <div style="display: flex; justify-content: space-between; align-items: center; background-color: #333333; padding: 8px; margin-top: 5px; border-radius: 5px;">
        <div style="color: white; flex: 1.7; font-weight: bold;">Player</div>
        <div style="color: white; flex: 1.5; font-weight: bold;">Club</div>
        <div style="flex: 1; text-align: center; color: white; font-weight: bold;">
            <img src="https://img.icons8.com/ios-filled/50/ffffff/clock--v1.png" alt="Clock Icon" style="width: 16px; height: 16px; vertical-align: middle;">
        </div>
        {''.join([f"<div style='color: white; flex: 1; font-weight: bold; text-align: center;'>{col}</div>" for col in selected_columns])}
        <div style="color: #ff7e82; flex: 1; font-weight: bold; text-align: center;">INDEX</div>
    </div>
    """
    st.markdown(headers_html, unsafe_allow_html=True)

    # Loop through and display each player's data
    for _, row in filtered_players.iterrows():
        # Prepare club and logo details dynamically for each player
        club_display = row['current_team_name'] if pd.notna(row['current_team_name']) else "No Club"
        logo_html = (
            f'<img src="{row["logo_url"]}" alt="Team Logo" '
            f'style="width: 20px; height: 20px; border-radius: 50%; filter: brightness(1.3)">'
            if pd.notna(row["logo_url"]) else ''
        )

        # Format percentile and index score for display
        percentiles_html = "".join([f"<div style='color: white; flex: 1; text-align: center;'>{round(row[col], 1)}</div>" for col in selected_columns])
        if pd.notna(row['birth_date']):
            birth_year = datetime.strptime(row['birth_date'], "%Y-%m-%d").strftime("'%y")
        else:
            birth_year = "Unknown"
        name_display = f"{row['full_name']} ({birth_year})"

        # Round and center-align the index score in bold and red
        index_score_html = f"<div style='color: #ff7e82; flex: 1; text-align: center; font-weight: bold;'>{round(row['index_score'], 1)}</div>"

        # Minutes Played column
        minutes_played_html = f"<div style='color: white; flex: 1; text-align: center; display: flex; align-items: center; justify-content: center;'>{int(row['minutes_on_field'])}</div>"

        # Render player row
        st.markdown(
            f"""
            <div style="display: flex; justify-content: space-between; align-items: center; background-color: #1e1e1e; padding: 8px; margin-top: 5px; border-radius: 5px;">
                <div style="color: white; flex: 1.7;"><strong>{name_display}</strong></div>
                <div style="color: white; flex: 1.5; display: flex; align-items: center;">
                    <div style="margin-right: 10px;">{club_display}</div>
                    {logo_html}
                </div>
                {minutes_played_html}
                {percentiles_html}
                {index_score_html}
            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown("</div>", unsafe_allow_html=True)  # Close container


# Player Percentiles tab content
def player_percentiles():

    # Load players and leagues
    players_df = load_players()

    # Function to safely handle None values and apply unidecode
    def safe_unidecode(value):
        if value is None:
            return "N/A" 
        return unidecode(value)  # Apply unidecode for non-None values

    # Create a dictionary of player_id and a formatted string of full_name - current_team_name
    player_options = {
        row['player_id']: f"{safe_unidecode(row['full_name'])} • {safe_unidecode(row['current_team_name'])}"
        for _, row in players_df.iterrows()
    }

    # Separate Utrecht players and others
    utrecht_players = {
        player_id: player_options[player_id]
        for player_id in player_options
        if player_options[player_id].endswith(" • Utrecht")
    }
    other_players = {
        player_id: player_options[player_id]
        for player_id in player_options
        if not player_options[player_id].endswith(" • Utrecht")
    }

    # Sort Utrecht and other players by player_id in descending order
    sorted_utrecht_players = dict(sorted(utrecht_players.items(), key=lambda x: str(x[0]), reverse=True))
    sorted_other_players = dict(sorted(other_players.items(), key=lambda x: str(x[0]), reverse=True))

    # Combine Utrecht players at the top
    sorted_player_options = {**sorted_utrecht_players, **sorted_other_players}

    col1, col2 = st.columns(2)

    with col1:
        # Use the selectbox to display the cleaned player name and current team, with Utrecht players on top
        selected_player_id = st.selectbox(
            "Choose a player",
            options=sorted_player_options.keys(),
            format_func=lambda player_id: sorted_player_options[player_id],
            key="player_selectbox"
        )

    # Display player profile
    player_stats_df = load_player_stats(selected_player_id)
    available_seasons = player_stats_df['season_id'].unique() if not player_stats_df.empty else ['2024-25']
    default_season = '2024-25' if '2024-25' in available_seasons else available_seasons[0]

    if st.session_state['selected_season'] is None:
        st.session_state['selected_season'] = default_season

    # Profile card
    display_player_profile(selected_player_id, players_df)

    # Set up the styled tabs using streamlit-option-menu
    selected_tab = option_menu(
        menu_title=None,  # No menu title to keep it looking like tabs
        options=['Information', 'Performance', 'Roles'],  # Tab names
        icons=['info-circle', 'clipboard-data', 'layers'],  # Icons for each tab (optional)
        default_index=0,  # Default tab selected
        orientation="horizontal",  # Horizontal layout to mimic tabs
        styles={
            "container": {"background-color": "#1e1e1e", "padding": "0px", "border-radius": "5px"},
            "nav-link": {
                "font-size": "14px",
                "color": "white",
                "padding": "10px 20px",
                "margin": "0px 5px",
                "text-align": "center",
                "border-radius": "5px",
                "transition": "all 0.3s ease",
            },
            "nav-link-selected": {
                "font-weight": "bold",
                "color": "black",
                "background-color": "#ff7e82",
                "border-radius": "5px",
            },
        }
    )

    # Red divider line
    st.markdown(
        """
        <hr style="border: none; height: 1px; background-color: #ff7e82; margin: 5px 0 30px;">
        """,
        unsafe_allow_html=True
    )

    # Load percentiles data for performance tab
    percentiles_df = load_player_percentiles(player_id=selected_player_id)
    # Tab Content Logic
    if selected_tab == 'Information':
        # Display all available season and club information for the selected player
        if not player_stats_df.empty:
            display_player_information(player_stats_df, leagues_dict)
        else:
            st.write("No player information available for this player.")

    # Code for Performance Tab Content
    elif selected_tab == 'Performance':
        with st.container():
            col1, col2 = st.columns([0.5, 0.5])
            
            # Filter player_stats_df for options where minutes_on_field > 199
            player_stats_filtered = player_stats_df[player_stats_df['minutes_on_field'] > 199]

            # Sort by season (descending) and minutes_on_field (descending)
            player_stats_filtered = player_stats_filtered.sort_values(
                by=['season_id', 'minutes_on_field'], 
                ascending=[False, False]
            )
            
            # Create combined filter options for Season-Competition-Club with country
            with col1:
                if not player_stats_filtered.empty:
                    combined_options = [
                        f"{row['season_id']} • {row['last_club_name']} • "
                        f"{leagues_dict.get(row['competition_id'], {}).get('competition_name', 'Unknown Competition')} "
                        f"({leagues_dict.get(row['competition_id'], {}).get('country_name', 'Unknown Country')})"
                        for _, row in player_stats_filtered.iterrows()
                    ]

                    # Display combined filter using option_menu
                    selected_combined_option = option_menu(
                        menu_title=None,
                        options=combined_options,
                        icons=['circle-fill'] * len(combined_options), 
                        default_index=0,
                        orientation="vertical",
                        styles={
                            "container": {"background-color": "#1e1e1e", "padding": "0px", "border-radius": "5px"},
                            "nav-link": {
                                "font-size": "13px",
                                "color": "white",
                                "padding": "5px 15px",
                                "margin": "0px 5px",
                                "text-align": "left",
                                "border-radius": "5px",
                            },
                            "nav-link-selected": {
                                "font-weight": "bold",
                                "color": "black",
                                "background-color": "#ff7e82",
                                "border-radius": "5px",
                            },
                        }
                    )

                    # Parse the selected option to retrieve season, club, competition, and country
                    selected_season, selected_club, selected_competition_with_country = selected_combined_option.split(" • ")
                    selected_competition_name, selected_country_name = selected_competition_with_country.rsplit("(", 1)
                    selected_competition_name = selected_competition_name.strip()
                    selected_country_name = selected_country_name.strip(")")

                    # Find competition ID based on competition name and country
                    selected_competition_id = next(
                        (comp_id for comp_id, details in leagues_dict.items() 
                        if details['competition_name'] == selected_competition_name 
                        and details['country_name'] == selected_country_name), 
                        None
                    )

                    # Store selected season and competition ID in session state
                    st.session_state['selected_season'] = selected_season
                    st.session_state['selected_competition'] = selected_competition_id

            # Display Performance Sub-tabs (Radar Chart, Passing Profile, Compare) within col2
            with col2:
                performance_tab = option_menu(
                    menu_title=None,
                    options=['Radar Chart', 'Passing Profile', 'Compare Player'],
                    icons=['pie-chart-fill', 'arrow-left-right', 'people-fill'],  # Different icons for clarity
                    default_index=0,
                    orientation="horizontal",
                    styles={
                        "container": {"background-color": "#1e1e1e", "padding": "0px", "border-radius": "5px"},
                        "nav-link": {
                            "font-size": "13px",
                            "color": "white",
                            "padding": "5px 15px",
                            "margin": "10px 5px",
                            "text-align": "center",
                            "border-radius": "5px",
                        },
                        "nav-link-selected": {
                            "font-weight": "bold",
                            "color": "black",
                            "background-color": "#ff7e82",
                            "border-radius": "5px",
                        },
                    },
                )

            # Radar Chart Logic
            if performance_tab == 'Radar Chart':
                season_templates = percentiles_df[
                    (percentiles_df['season_id'] == st.session_state['selected_season']) &
                    (percentiles_df['competition_id'] == st.session_state['selected_competition'])
                ]['template'].unique()

                if season_templates.any():
                    selected_template = st.radio("Select Template", season_templates, index=0, key="template_radio")
                    st.session_state['selected_template'] = selected_template
                    display_pizza_chart(selected_player_id, percentiles_df, st.session_state['selected_season'], st.session_state['selected_competition'], selected_template)
                else:
                    st.write("No templates available for this player in the selected competition.")
            
            # Passing Profile Logic
            elif performance_tab == 'Passing Profile':
                st.write("Passing Profile stats will be displayed here.")
            
            # Compare Logic
            elif performance_tab == 'Compare Player':
                with st.container():
                    col1, col2, col3 = st.columns([0.25, 0.5, 0.25])
                    # Render the selectbox on top
                    with col2:
                        selected_metadata = render_comparison_selectboxes(players_df, leagues_dict)
                    st.write('')
                    # Two columns for charts
                    col1, col2 = st.columns(2)

                    # Left: Current Player
                    with col1:
                        # Radar Chart for the Current Player
                        display_pizza_chart(
                            selected_player_id,
                            percentiles_df,
                            st.session_state['selected_season'],
                            st.session_state['selected_competition'],
                            st.session_state.get('selected_template', None),
                            key="current_player_chart"
                        )

                    # Right: Comparison Player
                    with col2:
                        if selected_metadata:
                            # Load percentiles for the selected comparison player
                            comparison_percentiles_df = load_player_percentiles(
                                player_id=selected_metadata['player_id'],
                                filters={
                                    'season_id': selected_metadata['season_id'],
                                    'competition_id': selected_metadata['competition_id']
                                }
                            )

                            # Radar Chart for Comparison Player
                            display_pizza_chart(
                                selected_metadata['player_id'],
                                comparison_percentiles_df,
                                selected_metadata['season_id'],
                                selected_metadata['competition_id'],
                                selected_metadata['template'],
                                key="comparison_player_chart"
                            )
                        else:
                            # Display a message prompting the user to select a player
                            st.write("Please select a player-league-season combination to compare to.")


    # Roles Tab
    elif selected_tab == 'Roles':

        # Map templates to full display names
        TEMPLATE_DISPLAY_NAMES = {
            "CB": "Centre-Back",
            "FB": "Full-Back",
            "DM": "Defensive-Midfield",
            "CM": "Centre-Midfield",
            "W": "Winger",
            "ST": "Striker"
        }

        # Map roles based on template
        ROLE_MAPPINGS = {
            "CB": {"bpd": "Ball-Playing", "phd": "Physical", "ffd": "Front-Foot"},
            "FB": {"bdt": "Build-up/Distributor", "wol": "Wide Overlap", "ivr": "Inverted Runner"},
            "DM": {"bwr": "Ball-Winner", "pmk": "Playmaker", "anc": "Anchor"},
            "CM": {"sws": "Shadow Striker", "cre": "Advanced Creator", "b2b": "Box-to-Box"},
            "W": {"wwr": "Wide Winger", "dth": "Deep Threat", "inf": "Inside Forward"},
            "ST": {"dls": "Deep-Lying Striker", "pch": "Poacher", "trg": "Target"}
        }

        # Role descriptions for Centre-Back
        CENTRE_BACK_ROLE_DESCRIPTIONS = {
            "bpd": "<b>Ball-Playing</b>: Progress the ball upfield, both by dribbling and passing. "
                "They consistently move the ball into the final third and have above-average quality in these efforts. "
                "Possession-oriented. Type Olivier Boscagli, Daley Blind, Nico Schlotterbeck.",
            "phd": "<b>Physical</b>: Focus on playing man-to-man (ground- and aerial) duels. "
                "They have a high volume/number of defensive actions on the pitch & win a lot. "
                "They will tend to stay in their defensive line and don't step forward too often. "
                "Type Mike van der Hoorn, Strahinja Pavlovic, Dante.",
            "ffd": "<b>Front-Foot</b>: Chase the ball/opponent down. They'll commit into tackles, fouls & blocks more often than their counterparts. "
                "Type Ryan Flamingo, Marcos Senesi, Cristian Romero."
        }

        # Role descriptions for Full-Back
        FULL_BACK_ROLE_DESCRIPTIONS = {
            "wol": "<b>Wide Overlap</b>: Tend to get to the wide assist zone a lot and can drive forward in possession. "
                "They are focused on getting the ball into the box in a more direct/crossing way. "
                "Type: Souffian El Karouani, Milos Kerkez, Pedro Porro.",
            "bdt": "<b>Build-up/Distributor</b>: Will be involved in first-phase possession (act as a +1) and are able to build-up tidily. "
                "They are more focused on transporting the ball upfield than to be offensively active. "
                "Type: Lutsharel Geertruida, Joško Gvardiol, Ben White.",
            "ivr": "<b>Inverted Runner</b>: Receive the ball high and are an outlet in possession for long passes. "
                "They'll drive forward mainly without the ball and also get into the opposition box. "
                "They don't tend to offer a lot in forward passing. Often wing-backs. "
                "Type: Denzel Dumfries, Jeremie Frimpong, Dani Carvajal. "
        }

        # Role descriptions for Defensive Midfield
        DEFENSIVE_MIDFIELD_ROLE_DESCRIPTIONS = {
            "pmk": "<b>Playmaker</b>: Sit at the base, receive a lot of (short) passes and help distribute play. "
                "Engine in possession. They help in progressing upfield. "
                "Type: Joey Veerman, Rodri, Youri Tielemans",
            "bwr": "<b>Ball-Winner</b>: Primary aim is recycling possession. They'll win a lot of possession by dueling and intercepting and then pass it to the more creative players. "
                "They generally have a high pass accuracy due to the low-risk nature. Type: Alonzo Engwanda, Azor Matusiwa, N'Golo Kanté.",
            "anc": "<b>Anchor</b>: Protecting defense, positionally disciplined. Their involvement in long passes helps transition play effectively when needed. Quarter-back like. "
                "Type: Sergio Busquets, Jordan Henderson, Emre Can."
        }

        # Role descriptions for Central Midfield
        CENTRAL_MIDFIELD_ROLE_DESCRIPTIONS = {
            "b2b": "<b>Box-to-Box</b>: Move from box to box and do work in both ends. They'll get into the opposition box/to chances "
                "and have the ability to drive/transition with the ball. Generally physically strong. They don't add too much creative passing. "
                "Type Quinten Timber, Dominik Szoboszlai, Andy Diouf.",
            "sws": "<b>Shadow Striker</b>: Mainly operate in attacking/higher zones and will try to find space without the ball. "
                "Their main threat is in the box, getting to many and good quality chances. They generally don't drop in possession to be available/move the ball to the final third. "
                "Type Guus Til, Thomas Müller, Davide Frattesi.",
            "cre": "<b>Advanced Creator</b>: Higher creative hub in the team. Their aim is to set up the offense, "
                "get the ball into the box and create chances. Generally have good individual skill and risk more in their passing. "
                "Type Calvin Stengs, James Maddison, Kevin De Bruyne."
        }

        # Role descriptions for Winger
        WINGER_ROLE_DESCRIPTIONS = {
            "inf": "<b>Inside Forward</b>: Involved in play and help the offense move into the final third. "
                "They'll be active in possession/short combinations, are skillful 1v1 and combine towards the box, "
                "instead of the more direct wing/crossing play. Type Osame Sahraoui, Ángel Di María & Michael Olise.",
            "dth": "<b>Deep Threat</b>: Mainly run deep and try to receive the ball high/behind the last line. "
                "They get into the box and to the end of chances. But offer less in general possession/forward passing. "
                "Often correlated with higher pace. Type Hirving Lozano, Bradley Barcola & Serge Gnabry.",
            "wwr": "<b>Wide Winger</b>: Play a more traditional wing role, staying wide and aimed at (early) direct crossing. "
                "They'll try to target the box and create chances. Generally have a lower passing accuracy/overall involvement. "
                "Type Bukayo Saka, Kingsley Coman & Francisco Conceição."
        }

        # Role descriptions for Striker
        STRIKER_ROLE_DESCRIPTIONS = {
            "pch": "<b>Poacher</b>: Striker in the purest sense of the word, being there at the end of moves. "
                "Most of their touches in the box, and the quality of the chances they get to is high. "
                "Type Erling Haaland, Robert Lewandowski & Luis Suárez.",
            "dls": "<b>Deep-Lying Striker</b>: Dropping away from the #9 position to be involved in-play. "
                "Often more of a facilitator than a pure goalscorer. Will combine short/into the box, trying to set up chances. "
                "Type Julián Álvarez, Harry Kane & Gerard Moreno.",
            "trg": "<b>Target</b>: Used as an outlet; they receive a lot of long passes, are involved aerially & draw fouls. "
                "In possession they tend to hold up play/pass short so that the rest of the team can join the offense. "
                "Won't see a lot of 1v1 action. Type Tobias Lauritsen, Dominic Calvert-Lewin & Alexander Sørloth."
        }

        def plot_styled_role_scores(role_scores_df, template, power=2):
            # Select role mapping based on template
            if template == "ST":
                role_keys = ["dls", "pch", "trg"]
                role_labels = ROLE_MAPPINGS["ST"]
            elif template == "W":
                role_keys = ["wwr", "dth", "inf"]
                role_labels = ROLE_MAPPINGS["W"]
            elif template == "CM":
                role_keys = ["sws", "cre", "b2b"]
                role_labels = ROLE_MAPPINGS["CM"]
            elif template == "DM":
                role_keys = ["bwr", "pmk", "anc"]
                role_labels = ROLE_MAPPINGS["DM"]
            elif template == "FB":
                role_keys = ["bdt", "wol", "ivr"]
                role_labels = ROLE_MAPPINGS["FB"]
            elif template == "CB":
                role_keys = ["phd", "bpd", "ffd"]
                role_labels = ROLE_MAPPINGS["CB"]

            # Normalize and apply power transformation to exaggerate scores
            score_sums = role_scores_df[role_keys].sum(axis=1)
            normalized_scores = role_scores_df[role_keys].div(score_sums, axis=0) * 100
            transformed_scores = normalized_scores ** power

            # Re-normalize after transformation so they still sum to 100
            transformed_sums = transformed_scores.sum(axis=1)
            final_scores = transformed_scores.div(transformed_sums, axis=0) * 100
            role_scores_df = role_scores_df.reset_index(drop=True)
            # Generate tooltips with season, competition name, and roles with %-scores
            tooltips = [
                f"<b>{row['season_id']} - {leagues_dict[row['competition_id']]['competition_name']}</b><br>"
                f"{role_labels[role_keys[0]]}: {final_scores.iloc[i, 0]:.2f}%<br>"
                f"{role_labels[role_keys[1]]}: {final_scores.iloc[i, 1]:.2f}%<br>"
                f"{role_labels[role_keys[2]]}: {final_scores.iloc[i, 2]:.2f}%"
                for i, row in role_scores_df.iterrows()
            ]

            # Initialize a Plotly figure for the ternary plot with a dark theme
            fig = go.Figure()

            # Add player's normalized role scores as markers on the ternary plot
            fig.add_trace(go.Scatterternary({
                'mode': 'markers+text',
                'a': final_scores[role_keys[1]],   # Top corner
                'b': final_scores[role_keys[0]],   # Left-bottom corner
                'c': final_scores[role_keys[2]],   # Right-bottom corner
                'marker': {
                    'symbol': 'star',  # Star shape
                    'size': 12,
                    'color': '#ff7e82',  # Red color for markers
                    'opacity': 0.8  # Increased opacity
                },
                'hovertext': tooltips,  # Tooltip text with season, competition, and scores
                'hoverinfo': 'text',  # Show tooltip on hover
                'name': "Role Occurrences"
            }))

            # Configure the layout for the ternary plot
            fig.update_layout({
                'paper_bgcolor': '#0e1117',  # Dark background for the plot
                'plot_bgcolor': '#0e1117',  # Dark background for the plot area
                'ternary': {
                    'sum': 100,  # Ensure all axes sum to 100 for normalized percentages
                    'aaxis': {
                        'title': '',
                        'min': 0, 'linewidth': 3, 'showgrid': True, 'gridcolor': 'grey',
                        'griddash': 'dashdot', 'ticks': '', 'showticklabels': False, 'color': 'grey',
                        'nticks': 4  # Only 4 divisions to create 4 inside triangles
                    },
                    'baxis': {
                        'title': '',
                        'min': 0, 'linewidth': 3, 'showgrid': True, 'gridcolor': 'grey',
                        'griddash': 'dashdot', 'ticks': '', 'showticklabels': False, 'color': 'grey',
                        'nticks': 4  # Only 4 divisions to create 4 inside triangles
                    },
                    'caxis': {
                        'title': '',
                        'min': 0, 'linewidth': 3, 'showgrid': True, 'gridcolor': 'grey',
                        'griddash': 'dashdot', 'ticks': '', 'showticklabels': False, 'color': 'grey',
                        'nticks': 4  # Only 4 divisions to create 4 inside triangles
                    }
                },
                'showlegend': False
            })

            # Add thick border around the triangle and remove inner grid
            fig.update_traces(marker=dict(line=dict(width=2, color='#ff7e82')))

            # Add main annotations for the three corners
            fig.add_annotation(x=0.5, y=1.15, text=f"<b>{role_labels[role_keys[1]]}</b>", showarrow=False, font=dict(size=16, color="white"))
            fig.add_annotation(x=0.05, y=0, text=f"<b>{role_labels[role_keys[0]]}</b>", showarrow=False, font=dict(size=16, color="white"), xanchor="left")
            fig.add_annotation(x=0.9, y=0, text=f"<b>{role_labels[role_keys[2]]}</b>", showarrow=False, font=dict(size=16, color="white"), xanchor="right")

            # Add "Balanced" label in the center of the triangle
            fig.add_annotation(x=0.5, y=0.45, text="<b>Balanced</b>", showarrow=False, font=dict(size=14, color="grey"))

            return fig

        # Usage in Streamlit
        role_scores_df = load_role_scores(selected_player_id).reset_index(drop=True)
        # If data is available, create an option menu for template selection
        if not role_scores_df.empty:
            # Get the template with the most occurrences for the default
            most_common_template = role_scores_df['template'].mode()[0]

            col1, col2 = st.columns([0.25, 0.75])

            with col1:
                # Create a dropdown menu for template selection
                selected_template = st.selectbox(
                    "Select Template",
                    options=role_scores_df['template'].unique(),
                    index=role_scores_df['template'].unique().tolist().index(most_common_template),
                    format_func=lambda x: TEMPLATE_DISPLAY_NAMES.get(x, x)
                )

            # Filter data for the selected template
            template_role_scores_df = role_scores_df[role_scores_df['template'] == selected_template]

            with col2:
                # Plot the ternary plot for the selected template
                fig = plot_styled_role_scores(template_role_scores_df, selected_template)
                st.plotly_chart(fig, use_container_width=True)

                TEMPLATE_ROLE_DESCRIPTIONS = {
                    "ST": STRIKER_ROLE_DESCRIPTIONS,
                    "W": WINGER_ROLE_DESCRIPTIONS,
                    "CM": CENTRAL_MIDFIELD_ROLE_DESCRIPTIONS,
                    "DM": DEFENSIVE_MIDFIELD_ROLE_DESCRIPTIONS,
                    "FB": FULL_BACK_ROLE_DESCRIPTIONS,
                    "CB": CENTRE_BACK_ROLE_DESCRIPTIONS
                }

                # Dynamically display role descriptions
                if selected_template in TEMPLATE_ROLE_DESCRIPTIONS:
                    with st.expander("Role Descriptions"):
                        role_descriptions = TEMPLATE_ROLE_DESCRIPTIONS[selected_template]
                        for role_key in ROLE_MAPPINGS[selected_template]:
                            st.markdown(
                                f'<div style="font-size:12px;">{role_descriptions[role_key]}</div><hr>',
                                unsafe_allow_html=True
                            )

        else:
            st.write("No role score data available for this player.")

# Function to load and join data
def load_games_from_db(db_name="wysc.db"):
    """Load games and join with scheduling_locations on home_team."""
    conn = sqlite3.connect(db_name)

    # Load scheduling_games and scheduling_locations, joining on home_team
    query = """
        SELECT 
            g.*,
            l.dist_km
        FROM 
            scheduling_games g
        LEFT JOIN 
            scheduling_locations l
        ON 
            g.home_team = l.home_team
    """
    match_data = pd.read_sql_query(query, conn)
    conn.close()

    # Convert 'date' column to datetime
    match_data['date'] = pd.to_datetime(match_data['date'], errors='coerce').dt.date
    return match_data

def scheduling_games():
    # Load data from the database
    match_data = load_games_from_db()

    # Filter games to include only those from today onwards
    today = datetime.today().date()
    match_data = match_data[match_data['date'] >= today]

    # If no data is found, display a message
    if match_data.empty:
        st.write("No games scheduled from today onwards.")
        return

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

    # Add a distance filter for maximum kilometers
    with st.container():
        col1, col2, col3 = st.columns([3, 1.5, 3])
        with col1:
            st.markdown(
                '<div style="line-height:0; margin-top:50px;">Games within the next month are displayed, within a reach of</div>',
                unsafe_allow_html=True
            )
        with col2:
            # Text input for maximum distance
            max_km_input = st.text_input(
                "",
                value="",  # Default is empty for no filter
                placeholder="Enter km"
            )
        with col3:
            st.markdown(
                '<div style="line-height:0; margin-top:50px;">kilometers from Utrecht.</div>',
                unsafe_allow_html=True
            )

    # Convert the text input to an integer if valid, otherwise set to 0 for no filter
    try:
        max_km = int(max_km_input) if max_km_input.strip().isdigit() else 0
    except ValueError:
        max_km = 0

    # Apply distance filter if a value greater than 0 is provided
    if max_km > 0:
        match_data = match_data[
            (match_data['dist_km'].notna()) & (match_data['dist_km'] <= max_km)
        ]

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

    # Add flags to league names
    match_data['league_with_flags'] = match_data.apply(
        lambda row: f"{row['league']} ({row['country']})", axis=1
    )

    # Get unique leagues
    unique_leagues = sorted(match_data['league'].unique())

    # Set default_selected to include all leagues
    default_selected = unique_leagues

    # Multiselect for leagues
    selected_leagues = st.multiselect(
        'Select Leagues',
        options=unique_leagues,
        default=default_selected
    )

    # Filter games based on the selected leagues
    if selected_leagues:
        selected_df = match_data[match_data['league'].isin(selected_leagues)]
    else:
        st.write("Please select at least one league.")
        return

    # Only proceed if `selected_df` is not empty
    if not selected_df.empty:
        # Group games by date
        match_data_sorted = selected_df.sort_values(by=['date', 'time']).reset_index()
        grouped = match_data_sorted.groupby('date')

        # Define the fixed card size
        card_width = 180
        card_height = 75
        max_columns = 5

        # Include Font Awesome CDN
        st.markdown(
            '<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0-beta3/css/all.min.css">',
            unsafe_allow_html=True
        )

        # Iterate over each date group
        for date, games in grouped:
            st.markdown(
                f"""
                <div style="
                    font-family: Bahnschrift, sans-serif; 
                    font-size: 16px; 
                    margin-bottom: 10px;
                    padding-bottom: 5px; 
                    border-bottom: 1px solid lightgray;">
                    {date.strftime('%A')}, {date.strftime('%d-%m-%Y')}
                </div>
                """,
                unsafe_allow_html=True
            )

            with st.container():
                chunks = [games.iloc[i:i + max_columns] for i in range(0, len(games), max_columns)]
                for chunk in chunks:
                    cols = st.columns(max_columns)
                    for i, (index, row) in enumerate(chunk.iterrows()):
                        flag_url = get_flag_url(row['country'])
                        distance_km = int(row['dist_km']) if not pd.isna(row['dist_km']) else "N/A"
                        with cols[i]:
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
                                    position: relative;">
                                    <img src="{flag_url}" style="
                                        position: absolute; 
                                        top: 3px; 
                                        right: 7px; 
                                        width: 14px;">
                                    <div style="font-weight: bold; font-size: 11px; font-family: Bahnschrift; margin-top: 5px;">
                                        {row['game']}
                                    </div>
                                    <div style="font-size: 11px; font-weight: bold; font-family: Bahnschrift; color: gray; margin-top: 3px; text-align: center;">
                                        <i class="fas fa-clock" style="margin-right: 5px;"></i> {row['time']} 
                                        <i class="fas fa-car" style="margin-left: 10px; margin-right: 5px;"></i> {distance_km} km
                                    </div>
                                </div>
                                """,
                                unsafe_allow_html=True
                            )
    else:
        st.write("No games found for the selected leagues.")

def main():
    if 'active_tab' not in st.session_state:
        st.session_state['active_tab'] = 'Player Profile'
    if 'selected_season' not in st.session_state:
        st.session_state['selected_season'] = None
    if 'selected_competition' not in st.session_state:
        st.session_state['selected_competition'] = None
    if 'selected_template' not in st.session_state:
        st.session_state['selected_template'] = None

    # Sidebar menu for tab selection with custom menu title styling
    with st.sidebar:
        # Custom title with styled "UU" in RECRUUT
        st.markdown(
            """
            <div style="font-size: 26px; font-weight: bold; margin-bottom: 20px; text-align: center;">
                RECR<span style="color: #ff7e82;">UU</span>T
            </div>
            """,
            unsafe_allow_html=True
        )

        # Add a small, centered image
        st.markdown(
            """
            <div style="text-align: center; margin-bottom: 20px;">
                <img src="fcu.png" alt="FCU Logo" style="width: 100px; height: auto;">
            </div>
            """,
            unsafe_allow_html=True
        )

        # Sidebar menu for tab selection
        st.session_state['active_tab'] = option_menu(
            menu_title=None,  # Hide default menu title
            menu_icon="list",
            options=["Player Profile", "Player Search", "Scheduling Games"],
            icons=["bar-chart-line", "search", "calendar"],
            default_index=0,
            key="menu_option",
            styles={
                "nav-link": {"font-size": "14px", "text-align": "left", "margin": "0px"},
                "nav-link-selected": {"font-weight": "bold", "color": "black", "background-color": "#ff7e82"}
            }
        )

        # Last data update information
        update = load_player_stats(-133882)
        update['extraction_timestamp'] = pd.to_datetime(update['extraction_timestamp'])
        last_update = update['extraction_timestamp'].max().strftime('%d-%m-%Y')
        st.markdown(f"<div style='font-size: 12px; text-align: center;'>Last data refresh: {last_update}</div>", unsafe_allow_html=True)
        st.write("")
        st.markdown(f"<div style='font-size: 12px; text-align: center;'>Performance radars available when >200 minutes played", unsafe_allow_html=True)

    if st.session_state['active_tab'] == "Player Profile":
        player_percentiles()
    elif st.session_state['active_tab'] == "Player Search":
        player_search()
    elif st.session_state['active_tab'] == 'Scheduling Games':
        scheduling_games()

if __name__ == "__main__":
    main()