import pandas as pd

# Create a sample dataset for demonstration purposes
sample_data = {
    'id': list(range(335982, 336092)),
    'city': ['Bangalore', 'Mumbai', 'Chennai', 'Delhi', 'Hyderabad'] * 22,
    'date': ['4/18/2008', '4/19/2008', '4/20/2008', '4/21/2008', '4/22/2008'] * 22,
    'player_of_match': ['Player1', 'Player2', 'Player3', 'Player4', 'Player5'] * 22,
    'venue': ['Stadium1', 'Stadium2', 'Stadium3', 'Stadium4', 'Stadium5'] * 22,
    'neutral_venue': [0, 0, 0, 0, 1] * 22,
    'team1': ['Royal Challengers Bangalore', 'Mumbai Indians', 'Chennai Super Kings', 'Delhi Capitals', 'Sunrisers Hyderabad'] * 22,
    'team2': ['Kolkata Knight Riders', 'Chennai Super Kings', 'Rajasthan Royals', 'Kolkata Knight Riders', 'Mumbai Indians'] * 22,
    'toss_winner': ['Royal Challengers Bangalore', 'Mumbai Indians', 'Chennai Super Kings', 'Delhi Capitals', 'Sunrisers Hyderabad'] * 22,
    'toss_decision': ['field', 'bat', 'field', 'bat', 'field'] * 22,
    'winner': ['Kolkata Knight Riders', 'Mumbai Indians', 'Chennai Super Kings', 'Delhi Capitals', 'Mumbai Indians'] * 22,
    'result': ['runs', 'wickets', 'runs', 'wickets', 'runs'] * 22,
    'result_margin': [10, 5, 15, 7, 20] * 22,
    'eliminator': ['N', 'N', 'N', 'N', 'Y'] * 22,
    'method': [None, None, 'D/L', None, None] * 22,
    'umpire1': ['Umpire1', 'Umpire2', 'Umpire3', 'Umpire4', 'Umpire5'] * 22,
    'umpire2': ['Umpire6', 'Umpire7', 'Umpire8', 'Umpire9', 'Umpire10'] * 22
}

# Create dataframe and save to CSV
sample_df = pd.DataFrame(sample_data)
sample_df.to_csv('ipl.csv', index=False)
print("Sample IPL dataset created.") 