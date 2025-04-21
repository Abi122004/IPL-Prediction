"""
IPL Match Winner Prediction Model
This script generates a sample IPL dataset and trains a LightGBM model to predict match winners.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import lightgbm as lgb
import os

def create_sample_data(filename='ipl.csv'):
    """Create a sample IPL dataset for demonstration purposes"""
    sample_data = {
        'id': list(range(335982, 336292)),
        'city': ['Bangalore', 'Mumbai', 'Chennai', 'Delhi', 'Hyderabad'] * 62,
        'date': ['4/18/2008', '4/19/2008', '4/20/2008', '4/21/2008', '4/22/2008'] * 62,
        'player_of_match': ['Player1', 'Player2', 'Player3', 'Player4', 'Player5'] * 62,
        'venue': ['Stadium1', 'Stadium2', 'Stadium3', 'Stadium4', 'Stadium5'] * 62,
        'neutral_venue': [0, 0, 0, 0, 1] * 62,
        'team1': ['Royal Challengers Bangalore', 'Mumbai Indians', 'Chennai Super Kings', 'Delhi Capitals', 'Sunrisers Hyderabad'] * 62,
        'team2': ['Kolkata Knight Riders', 'Chennai Super Kings', 'Rajasthan Royals', 'Kolkata Knight Riders', 'Mumbai Indians'] * 62,
        'toss_winner': ['Royal Challengers Bangalore', 'Mumbai Indians', 'Chennai Super Kings', 'Delhi Capitals', 'Sunrisers Hyderabad'] * 62,
        'toss_decision': ['field', 'bat', 'field', 'bat', 'field'] * 62,
        'winner': ['Kolkata Knight Riders', 'Mumbai Indians', 'Chennai Super Kings', 'Delhi Capitals', 'Mumbai Indians'] * 62,
        'result': ['runs', 'wickets', 'runs', 'wickets', 'runs'] * 62,
        'result_margin': [10, 5, 15, 7, 20] * 62,
        'eliminator': ['N', 'N', 'N', 'N', 'Y'] * 62,
        'method': [None, None, 'D/L', None, None] * 62,
        'umpire1': ['Umpire1', 'Umpire2', 'Umpire3', 'Umpire4', 'Umpire5'] * 62,
        'umpire2': ['Umpire6', 'Umpire7', 'Umpire8', 'Umpire9', 'Umpire10'] * 62
    }
    
    # Create dataframe and save to CSV
    sample_df = pd.DataFrame(sample_data)
    sample_df.to_csv(filename, index=False)
    print(f"Sample IPL dataset created and saved to {filename}")
    return sample_df

def preprocess_data(data):
    """Preprocess the IPL dataset"""
    # Replace team names to standardize the dataset
    data.team1.replace({'Rising Pune Supergiants': 'Rising Pune Supergiant'}, regex=True, inplace=True)
    data.team2.replace({'Rising Pune Supergiants': 'Rising Pune Supergiant'}, regex=True, inplace=True)
    data.winner.replace({'Rising Pune Supergiants': 'Rising Pune Supergiant'}, regex=True, inplace=True)
    data.toss_winner.replace({'Rising Pune Supergiants': 'Rising Pune Supergiant'}, regex=True, inplace=True)

    data.team1.replace({'Deccan Chargers': 'Sunrisers Hyderabad'}, regex=True, inplace=True)
    data.team2.replace({'Deccan Chargers': 'Sunrisers Hyderabad'}, regex=True, inplace=True)
    data.winner.replace({'Deccan Chargers': 'Sunrisers Hyderabad'}, regex=True, inplace=True)
    data.toss_winner.replace({'Deccan Chargers': 'Sunrisers Hyderabad'}, regex=True, inplace=True)

    data.team1.replace({'Delhi Daredevils': 'Delhi Capitals'}, regex=True, inplace=True)
    data.team2.replace({'Delhi Daredevils': 'Delhi Capitals'}, regex=True, inplace=True)
    data.winner.replace({'Delhi Daredevils': 'Delhi Capitals'}, regex=True, inplace=True)
    data.toss_winner.replace({'Delhi Daredevils': 'Delhi Capitals'}, regex=True, inplace=True)

    data.team1.replace({'Pune Warriors': 'Rising Pune Supergiant'}, regex=True, inplace=True)
    data.team2.replace({'Pune Warriors': 'Rising Pune Supergiant'}, regex=True, inplace=True)
    data.winner.replace({'Pune Warriors': 'Rising Pune Supergiant'}, regex=True, inplace=True)
    data.toss_winner.replace({'Pune Warriors': 'Rising Pune Supergiant'}, regex=True, inplace=True)

    # Fill missing values
    data['city'].fillna('Unknown', inplace=True)
    cols_to_fill = ['player_of_match', 'result', 'eliminator']
    data[cols_to_fill] = data[cols_to_fill].fillna('Not Available')
    mean_result_margin = data['result_margin'].mean()
    data['result_margin'].fillna(mean_result_margin, inplace=True)

    # Drop unwanted columns
    data.drop(['id', 'method'], axis=1, inplace=True)
    
    # Drop rows with missing values in the 'winner' column
    data.dropna(subset=['winner'], inplace=True)
    
    # Convert date and extract features
    try:
        data['date'] = pd.to_datetime(data['date'])
        data['season'] = pd.DatetimeIndex(data['date']).year
        data['day'] = data['date'].dt.day
        data['month'] = data['date'].dt.month
        data['year'] = data['date'].dt.year
    except:
        print("Warning: Date conversion failed. Using default values.")
        data['season'] = 2022
        data['day'] = 1
        data['month'] = 5
        data['year'] = 2022

    # Encode categorical features
    # Teams
    team_mapping = {
        'Kolkata Knight Riders': 1,
        'Chennai Super Kings': 2,
        'Delhi Capitals': 3,
        'Royal Challengers Bangalore': 4,
        'Rajasthan Royals': 5,
        'Kings XI Punjab': 6,
        'Sunrisers Hyderabad': 7,
        'Mumbai Indians': 8,
        'Rising Pune Supergiant': 9,
        'Kochi Tuskers Kerala': 10,
        'Gujarat Lions': 11
    }
    
    data['team1'] = data['team1'].map(team_mapping)
    data['team2'] = data['team2'].map(team_mapping)
    data['winner'] = data['winner'].map(team_mapping)
    data['toss_winner'] = data['toss_winner'].map(team_mapping)
    
    # Venue
    venue_mapping = {venue: i for i, venue in enumerate(data['venue'].unique())}
    data['venue'] = data['venue'].map(venue_mapping)
    
    # Toss decision
    temp = {'field': 0, 'bat': 1}
    data['toss_decision'] = data['toss_decision'].map(temp)
    
    # Umpires
    umpires_set = set(data['umpire1'].unique()).union(set(data['umpire2'].unique()))
    umpire_dict = {umpire: i for i, umpire in enumerate(umpires_set, 1)}
    data['umpire1'] = data['umpire1'].map(umpire_dict)
    data['umpire2'] = data['umpire2'].map(umpire_dict)
    
    # Player of match
    player_of_match_mapping = {player: i for i, player in enumerate(data['player_of_match'].unique())}
    data['player_of_match'] = data['player_of_match'].map(player_of_match_mapping)
    
    # City
    city_mapping = {city: i for i, city in enumerate(data['city'].unique())}
    data['city'] = data['city'].map(city_mapping)
    
    # Drop unwanted columns
    unwanted_columns = ['date', 'result', 'eliminator', 'season', 'day', 'month', 'year']
    data.drop(columns=unwanted_columns, inplace=True)
    
    return data

def train_model(data):
    """Train a LightGBM model on the preprocessed data"""
    # Split the data into features (X) and the target variable (y)
    X = data.drop(['winner'], axis=1)
    y = data['winner']

    # Split the data into training and testing sets (70% training, 30% testing)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Create an instance of the LGBMClassifier model with updated parameters
    model = lgb.LGBMClassifier(
        boosting_type='gbdt',
        num_leaves=31,
        max_depth=-1,
        learning_rate=0.1,
        n_estimators=100,
        objective='multiclass',
        n_jobs=-1,
        verbose=-1
    )

    # Fit the model on the training data
    model.fit(X_train, y_train)

    # Make predictions on the test set
    y_pred = model.predict(X_test)

    # Calculate accuracy score
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Model Accuracy: {accuracy:.4f}")
    
    return model

def main():
    """Main function to run the IPL match winner prediction analysis"""
    # Check if the dataset exists, otherwise create it
    if os.path.isfile('ipl.csv'):
        print("Loading existing IPL dataset")
        data = pd.read_csv('ipl.csv')
    else:
        print("Creating sample IPL dataset")
        data = create_sample_data()
    
    # Preprocess the data
    print("Preprocessing data...")
    processed_data = preprocess_data(data)
    
    # Train the model
    print("Training model...")
    model = train_model(processed_data)
    
    print("Done! The model is ready for making predictions.")

if __name__ == "__main__":
    main() 