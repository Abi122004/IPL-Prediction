import pandas as pd
import numpy as np

# Read the original dataset
df = pd.read_csv('ipl.csv')

# Define team mappings (old names to new names)
team_mappings = {
    'Royal Challengers Bangalore': 'Royal Challengers Bengaluru',
    'Kings XI Punjab': 'Punjab Kings',
    'Delhi Daredevils': 'Delhi Capitals'
}

# Update team names
df['team1'] = df['team1'].replace(team_mappings)
df['team2'] = df['team2'].replace(team_mappings)

# Filter out old teams that are no longer in IPL
current_teams = [
    'Chennai Super Kings',
    'Delhi Capitals',
    'Gujarat Titans',
    'Kolkata Knight Riders',
    'Lucknow Super Giants',
    'Mumbai Indians',
    'Punjab Kings',
    'Rajasthan Royals',
    'Royal Challengers Bengaluru',
    'Sunrisers Hyderabad'
]

# Keep only matches between current teams
df = df[df['team1'].isin(current_teams) & df['team2'].isin(current_teams)]

# Update winner column to match new team names
df['winner'] = df['winner'].replace(team_mappings)

# Update toss_winner column to match new team names
df['toss_winner'] = df['toss_winner'].replace(team_mappings)

# Save the updated dataset
df.to_csv('ipl_2025.csv', index=False)

print("Dataset has been updated with current teams!")
print("\nCurrent teams in the dataset:")
print(sorted(df['team1'].unique())) 