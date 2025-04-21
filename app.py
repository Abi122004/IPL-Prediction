# Importing necessary modules from the Flask framework
from flask import Flask, render_template, request, jsonify, url_for
import os
import time
import random

import numpy as np  # Library for numerical operations
import pandas as pd  # Library for data manipulation and analysis
from sklearn.model_selection import train_test_split  # Function to split data into training and testing sets
from sklearn.metrics import accuracy_score  # Function to calculate accuracy score
import lightgbm as lgb  # Importing LightGBM library, Light Gradient Boosting Machine library

# Create an instance of the Flask class and name it 'app'
app = Flask(__name__)

# Add a cache busting function to ensure fresh CSS/JS loads
@app.context_processor
def utility_processor():
    def cache_bust_url():
        return str(int(time.time()))
    return dict(cache_bust_url=cache_bust_url)

# Define current IPL teams
CURRENT_TEAMS = [
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

# Create sample data if ipl_2025.csv doesn't exist
def create_sample_data():
    # Create a sample dataset for demonstration purposes
    sample_data = {
        'id': list(range(335982, 336092)),
        'city': ['Bangalore', 'Mumbai', 'Chennai', 'Delhi', 'Hyderabad'] * 22,
        'date': ['4/18/2008', '4/19/2008', '4/20/2008', '4/21/2008', '4/22/2008'] * 22,
        'player_of_match': ['Player1', 'Player2', 'Player3', 'Player4', 'Player5'] * 22,
        'venue': ['Stadium1', 'Stadium2', 'Stadium3', 'Stadium4', 'Stadium5'] * 22,
        'neutral_venue': [0, 0, 0, 0, 1] * 22,
        'team1': CURRENT_TEAMS * 11,
        'team2': CURRENT_TEAMS[::-1] * 11,
        'toss_winner': CURRENT_TEAMS * 11,
        'toss_decision': ['field', 'bat', 'field', 'bat', 'field'] * 22,
        'winner': CURRENT_TEAMS * 11,
        'result': ['runs', 'wickets', 'runs', 'wickets', 'runs'] * 22,
        'result_margin': [10, 5, 15, 7, 20] * 22,
        'eliminator': ['N', 'N', 'N', 'N', 'Y'] * 22,
        'method': [None, None, 'D/L', None, None] * 22,
        'umpire1': ['Umpire1', 'Umpire2', 'Umpire3', 'Umpire4', 'Umpire5'] * 22,
        'umpire2': ['Umpire6', 'Umpire7', 'Umpire8', 'Umpire9', 'Umpire10'] * 22
    }
    
    # Create dataframe and save to CSV
    sample_df = pd.DataFrame(sample_data)
    sample_df.to_csv('ipl_2025.csv', index=False)
    print("Sample IPL 2025 dataset created.")
    return sample_df

@app.route('/')
def home():
    return render_template('match_predict.html', teams=CURRENT_TEAMS)

@app.route('/predict', methods=['POST'])
def predict():
    # Get data from JSON if Content-Type is application/json, otherwise get from form
    if request.is_json:
        data = request.get_json()
        Team1 = data.get('team1')
        Team2 = data.get('team2')
        Venue = data.get('venue')
        Toss_winner = data.get('toss_winner')
        Toss_decision = data.get('toss_decision')
        
        # Extract city from venue or use a default
        if Venue == "M Chinnaswamy Stadium":
            City = "Bangalore"
        elif Venue == "Wankhede Stadium":
            City = "Mumbai"
        elif Venue == "MA Chidambaram Stadium":
            City = "Chennai"
        elif Venue == "Arun Jaitley Stadium":
            City = "Delhi"
        elif Venue == "Narendra Modi Stadium":
            City = "Ahmedabad"
        elif Venue == "Eden Gardens":
            City = "Kolkata"
        elif Venue == "Ekana Cricket Stadium":
            City = "Lucknow"
        elif Venue == "Punjab Cricket Association Stadium":
            City = "Mohali"
        elif Venue == "Sawai Mansingh Stadium":
            City = "Jaipur"
        elif Venue == "Rajiv Gandhi International Stadium":
            City = "Hyderabad"
        else:
            City = "Unknown"
            
        # Default values for other parameters
        Neutral_venue = 0
        Umpire1 = "Default Umpire 1"
        Umpire2 = "Default Umpire 2"
    else:
        # Form data
        City = request.form['city']
        Venue = request.form['venue']
        Neutral_venue = int(request.form['neutral_venue'])
        Team1 = request.form['team1']
        Team2 = request.form['team2']
        Toss_winner = request.form['toss_winner']
        Toss_decision = request.form['toss_decision']
        Umpire1 = request.form['umpire1']
        Umpire2 = request.form['umpire2']

    # Check if the ipl.csv file exists, if not, create sample data
    if os.path.isfile('ipl_2025.csv'):
        # Reading the CSV file 'ipl.csv' and storing the data in a DataFrame called 'data'
        data = pd.read_csv('ipl_2025.csv')
    else:
        # Create sample data
        data = create_sample_data()
    
    # Print columns for debugging
    print("Available columns:", data.columns.tolist())
    print("Data types before processing:", data.dtypes)
    
    # Print the venue and derived city for debugging
    print(f"Selected venue: {Venue}")
    print(f"Derived city: {City}")

    # Identifying information about composition and potential data quality
    # data.info()
    
    # Print columns for debugging
    print("Available columns:", data.columns.tolist())
    print("Data types before processing:", data.dtypes)

    # Replacing 'Rising Pune Supergiants' with 'Rising Pune Supergiant' in the 'team1', 'team2', 'winner', and 'toss_winner' columns.
    if 'team1' in data.columns:
        data.team1.replace({'Rising Pune Supergiants' : 'Rising Pune Supergiant'},regex=True,inplace=True)
    if 'team2' in data.columns:
        data.team2.replace({'Rising Pune Supergiants' : 'Rising Pune Supergiant'},regex=True,inplace=True)
    if 'winner' in data.columns:
        data.winner.replace({'Rising Pune Supergiants' : 'Rising Pune Supergiant'},regex=True,inplace=True)
    if 'toss_winner' in data.columns:
        data.toss_winner.replace({'Rising Pune Supergiants' : 'Rising Pune Supergiant'},regex=True,inplace=True)

    # Replacing 'Deccan Chargers' with 'Sunrisers Hyderabad' in the 'team1', 'team2', 'winner', and 'toss_winner' columns.
    if 'team1' in data.columns:
        data.team1.replace({'Deccan Chargers' : 'Sunrisers Hyderabad'},regex=True,inplace=True)
    if 'team2' in data.columns:
        data.team2.replace({'Deccan Chargers' : 'Sunrisers Hyderabad'},regex=True,inplace=True)
    if 'winner' in data.columns:
        data.winner.replace({'Deccan Chargers' : 'Sunrisers Hyderabad'},regex=True,inplace=True)
    if 'toss_winner' in data.columns:
        data.toss_winner.replace({'Deccan Chargers' : 'Sunrisers Hyderabad'},regex=True,inplace=True)

    # Replacing 'Delhi Daredevils' with 'Delhi Capitals' in the 'team1', 'team2', 'winner', and 'toss_winner' columns.
    if 'team1' in data.columns:
        data.team1.replace({'Delhi Daredevils' : 'Delhi Capitals'},regex=True,inplace=True)
    if 'team2' in data.columns:
        data.team2.replace({'Delhi Daredevils' : 'Delhi Capitals'},regex=True,inplace=True)
    if 'winner' in data.columns:
        data.winner.replace({'Delhi Daredevils' : 'Delhi Capitals'},regex=True,inplace=True)
    if 'toss_winner' in data.columns:
        data.toss_winner.replace({'Delhi Daredevils' : 'Delhi Capitals'},regex=True,inplace=True)

    # Replacing 'Pune Warriors' with 'Rising Pune Supergiant' in the 'team1', 'team2', 'winner', and 'toss_winner' columns.
    if 'team1' in data.columns:
        data.team1.replace({'Pune Warriors' : 'Rising Pune Supergiant'},regex=True,inplace=True)
    if 'team2' in data.columns:
        data.team2.replace({'Pune Warriors' : 'Rising Pune Supergiant'},regex=True,inplace=True)
    if 'winner' in data.columns:
        data.winner.replace({'Pune Warriors' : 'Rising Pune Supergiant'},regex=True,inplace=True)
    if 'toss_winner' in data.columns:
        data.toss_winner.replace({'Pune Warriors' : 'Rising Pune Supergiant'},regex=True,inplace=True)

    # checking for the null values in the dataset
    # data.isnull().sum()
    # there exists null values

    # Fill missing values in 'city' column with 'Unknown'
    if 'city' in data.columns:
        data['city'].fillna('Unknown', inplace=True)

    # Fill missing values in 'player_of_match', 'result', and 'eliminator' columns with 'Not Available'
    # First ensure all required columns exist, add them if they don't
    required_columns = ['player_of_match', 'result', 'eliminator']
    for col in required_columns:
        if col not in data.columns:
            data[col] = 'Not Available'
    
    # Now fill NA values
    cols_to_fill = [col for col in required_columns if col in data.columns]
    if cols_to_fill:
        data[cols_to_fill] = data[cols_to_fill].fillna('Not Available')

    # Calculate the mean of the 'result_margin' column
    if 'result_margin' in data.columns:
        mean_result_margin = data['result_margin'].mean()
        # Fill missing values in 'result_margin' column with the mean
        data['result_margin'].fillna(mean_result_margin, inplace=True)
    else:
        # If result_margin doesn't exist, create it with a default value
        data['result_margin'] = 10  # Use a reasonable default value

    # Drop unwanted columns if they exist
    for col in ['id', 'method']:
        if col in data.columns:
            data.drop(col, axis=1, inplace=True)

    # Drop rows with missing values in the 'winner' column
    if 'winner' in data.columns:
        data.dropna(subset=['winner'], inplace=True)
    else:
        # If winner column doesn't exist (highly unlikely), create it with default values
        print("Warning: 'winner' column not found in dataset. Creating default column.")
        data['winner'] = data['team1']  # Default winner is team1

    # Process date if it exists
    if 'date' in data.columns:
        try:
            data['date'] = pd.to_datetime(data['date'])
            data['season'] = pd.DatetimeIndex(data['date']).year
            # Extracting day, month, and year from the 'date' column
            data['day'] = data['date'].dt.day
            data['month'] = data['date'].dt.month
            data['year'] = data['date'].dt.year
        except Exception as e:
            print(f"Date conversion error: {e}")
            # Set default values if date conversion fails
            data['season'] = 2023
            data['day'] = 1
            data['month'] = 5
            data['year'] = 2023
    else:
        # If date doesn't exist, create these columns with default values
        data['season'] = 2023
        data['day'] = 1
        data['month'] = 5
        data['year'] = 2023

    # Handle problematic object columns from the error message
    # Convert object columns to numeric types for LightGBM
    object_columns = data.select_dtypes(include=['object']).columns.tolist()
    print(f"Object columns that need conversion: {object_columns}")
    
    # Specifically handle match_type and super_over columns if they exist
    if 'match_type' in data.columns:
        # Create a mapping for match_type
        match_types = data['match_type'].unique()
        match_type_mapping = {match_type: i for i, match_type in enumerate(match_types)}
        data['match_type'] = data['match_type'].map(match_type_mapping).astype(int)
        print(f"Converted match_type to integers: {match_type_mapping}")
    
    if 'super_over' in data.columns:
        # Handle super_over - typically a boolean (Yes/No) or similar
        if data['super_over'].dtype == 'object':
            # Try to convert to boolean if possible (True/False, Yes/No, Y/N, etc.)
            unique_values = data['super_over'].unique()
            print(f"Unique values in super_over: {unique_values}")
            
            # Create a simple mapping
            if len(unique_values) <= 2:
                super_over_mapping = {val: i for i, val in enumerate(unique_values)}
                data['super_over'] = data['super_over'].map(super_over_mapping).astype(int)
                print(f"Converted super_over to integers: {super_over_mapping}")
            else:
                # If too many unique values, use label encoding
                from sklearn.preprocessing import LabelEncoder
                le = LabelEncoder()
                data['super_over'] = le.fit_transform(data['super_over'])
                print(f"Used LabelEncoder for super_over with {len(unique_values)} unique values")

    # Create a dictionary to map team names to unique numbers
    team_mapping = {
        'Kolkata Knight Riders': 1,
        'Chennai Super Kings': 2,
        'Delhi Capitals': 3,
        'Royal Challengers Bangalore': 4,
        'Royal Challengers Bengaluru': 4,  # Map to the same number as RCB
        'Rajasthan Royals': 5,
        'Kings XI Punjab': 6,
        'Punjab Kings': 6,  # Map to the same number as Kings XI Punjab
        'Sunrisers Hyderabad': 7,
        'Mumbai Indians': 8,
        'Rising Pune Supergiant': 9,
        'Kochi Tuskers Kerala': 10,
        'Gujarat Lions': 11,
        'Gujarat Titans': 12,
        'Lucknow Super Giants': 13
    }

    # Create a reverse mapping for prediction output
    team_mapping_reverse = {v: k for k, v in team_mapping.items()}

    # Ensure required team columns exist
    for col in ['team1', 'team2', 'winner', 'toss_winner']:
        if col not in data.columns:
            # Default to one of the input teams
            if col == 'team1':
                data[col] = Team1
            elif col == 'team2':
                data[col] = Team2
            elif col == 'winner':
                data[col] = Team1  # Default winner
            elif col == 'toss_winner':
                data[col] = Toss_winner

    # Replace team names in 'team1' and 'team2' columns with unique numbers
    data['team1'] = data['team1'].map(team_mapping)
    data['team2'] = data['team2'].map(team_mapping)

    # Replace winner names in 'winner' column with unique numbers
    data['winner'] = data['winner'].map(team_mapping)
    data['toss_winner'] = data['toss_winner'].map(team_mapping)

    # Handle missing values after mapping
    for col in ['team1', 'team2', 'winner', 'toss_winner']:
        data[col] = data[col].fillna(-1)  # Use -1 for unknown teams

    # Create a dictionary to map each unique venue name to a unique number
    if 'venue' not in data.columns:
        data['venue'] = Venue  # Use input venue if column doesn't exist
        
    venue_mapping = {venue: i for i, venue in enumerate(data['venue'].unique())}
    
    # Replace the venue names in the 'venue' column with the corresponding unique numbers
    data['venue'] = data['venue'].map(venue_mapping)

    # Create a dictionary to map 'toss_decision' values to numerical values
    temp = {'field': 0, 'bat': 1}
    
    # Ensure toss_decision exists
    if 'toss_decision' not in data.columns:
        data['toss_decision'] = Toss_decision

    # Use the map() function to replace 'toss_decision' values with numerical values
    data['toss_decision'] = data['toss_decision'].map(temp)

    # Ensure umpire columns exist
    for col in ['umpire1', 'umpire2']:
        if col not in data.columns:
            if col == 'umpire1':
                data[col] = Umpire1
            else:
                data[col] = Umpire2

    # Create a set of unique umpires
    umpires_set = set(data['umpire1'].unique()).union(set(data['umpire2'].unique()))

    # Create a dictionary to map umpire names to unique numbers
    umpire_dict = {umpire: i for i, umpire in enumerate(umpires_set, 1)}

    # Apply the dictionary to create new encoded columns for 'umpire1' and 'umpire2'
    data['umpire1'] = data['umpire1'].map(umpire_dict)
    data['umpire2'] = data['umpire2'].map(umpire_dict)

    # Ensure player_of_match exists and use a default value
    if 'player_of_match' not in data.columns:
        data['player_of_match'] = 'Not Available'

    # Create a dictionary to map each unique player_of_match name to a unique number
    player_of_match_mapping = {player: i for i, player in enumerate(data['player_of_match'].unique())}
    default_player_value = player_of_match_mapping.get('Not Available', 0)

    # Replace the player_of_match names with the corresponding unique numbers
    data['player_of_match'] = data['player_of_match'].map(player_of_match_mapping)

    # Ensure city exists
    if 'city' not in data.columns:
        data['city'] = City

    # Create a dictionary to map each unique city name to a unique number
    city_mapping = {city: i for i, city in enumerate(data['city'].unique())}

    # Replace the city names with the corresponding unique numbers
    data['city'] = data['city'].map(city_mapping)

    # Ensure neutral_venue exists
    if 'neutral_venue' not in data.columns:
        data['neutral_venue'] = Neutral_venue

    # List of unwanted columns
    unwanted_columns = ['date', 'result', 'eliminator', 'season', 'day', 'month', 'year']

    # Drop the unwanted columns from the DataFrame if they exist
    existing_unwanted = [col for col in unwanted_columns if col in data.columns]
    if existing_unwanted:
        data.drop(columns=existing_unwanted, inplace=True)

    # Convert any remaining object columns to numeric
    # Find remaining object columns
    remaining_objects = data.select_dtypes(include=['object']).columns.tolist()
    print(f"Remaining object columns after initial processing: {remaining_objects}")
    
    for col in remaining_objects:
        print(f"Converting column {col} from object to numeric")
        # Use label encoding for categorical variables
        from sklearn.preprocessing import LabelEncoder
        le = LabelEncoder()
        data[col] = le.fit_transform(data[col].astype(str))

    # Handle NaN values in all columns
    for col in data.columns:
        if data[col].dtype == 'float64' or data[col].dtype == 'int64':
            data[col] = data[col].fillna(0)
        else:
            data[col] = data[col].fillna('Unknown')
            # Convert any remaining object columns to numeric
            from sklearn.preprocessing import LabelEncoder
            le = LabelEncoder()
            data[col] = le.fit_transform(data[col].astype(str))

    # Print data types for verification
    print("Data types after processing:", data.dtypes)

    # Final check - ensure all columns are numeric
    for col in data.columns:
        if data[col].dtype == 'object':
            print(f"Warning: Column {col} is still an object type. Converting to numeric.")
            # Force convert to numeric
            data[col] = pd.to_numeric(data[col], errors='coerce').fillna(0).astype(int)

    # Split the data into features (X) and the target variable (y)
    X = data.drop(['winner'], axis=1)
    y = data['winner']

    # Add feature engineering for better team performance metrics
    print("Adding feature engineering to improve model accuracy...")
    
    # 1. Create historical win rates for teams
    team_stats = {}
    for team in data['team1'].unique():
        # Calculate team's overall win rate
        team_matches = data[(data['team1'] == team) | (data['team2'] == team)]
        if len(team_matches) > 0:
            win_rate = len(team_matches[team_matches['winner'] == team]) / len(team_matches)
            team_stats[team] = win_rate
        else:
            team_stats[team] = 0.5  # Default if no matches
    
    # Add team win rates as features
    X['team1_win_rate'] = X['team1'].map(team_stats).fillna(0.5)
    X['team2_win_rate'] = X['team2'].map(team_stats).fillna(0.5)
    X['win_rate_diff'] = X['team1_win_rate'] - X['team2_win_rate']
    
    # 2. Create head-to-head matchup features
    head_to_head = {}
    for team1 in data['team1'].unique():
        head_to_head[team1] = {}
        for team2 in data['team2'].unique():
            if team1 == team2:
                continue
                
            # Get matches between these teams
            matchups = data[((data['team1'] == team1) & (data['team2'] == team2)) | 
                           ((data['team1'] == team2) & (data['team2'] == team1))]
            
            if len(matchups) > 0:
                # Calculate win rate of team1 against team2
                team1_wins = len(matchups[matchups['winner'] == team1])
                head_to_head[team1][team2] = team1_wins / len(matchups)
            else:
                head_to_head[team1][team2] = 0.5  # Default if no matchups
    
    # Add head-to-head feature
    def get_head_to_head(row):
        team1 = row['team1']
        team2 = row['team2']
        try:
            return head_to_head.get(team1, {}).get(team2, 0.5)
        except:
            return 0.5
        
    X['head_to_head'] = X.apply(get_head_to_head, axis=1)
    
    # 3. Add toss win advantage
    toss_advantage = {}
    for decision in [0, 1]:  # field (0) or bat (1)
        matches = data[data['toss_decision'] == decision]
        if len(matches) > 0:
            # How often does toss winner win the match for this decision?
            toss_winner_wins = len(matches[matches['toss_winner'] == matches['winner']])
            toss_advantage[decision] = toss_winner_wins / len(matches)
        else:
            toss_advantage[decision] = 0.5
    
    X['toss_decision_advantage'] = X['toss_decision'].map(toss_advantage).fillna(0.5)
    X['is_toss_winner_team1'] = (X['toss_winner'] == X['team1']).astype(int)
    X['toss_win_value'] = X['is_toss_winner_team1'] * X['toss_decision_advantage']
    
    # 4. Venue advantage
    venue_team_advantage = {}
    for venue in X['venue'].unique():
        venue_matches = data[data['venue'] == venue]
        venue_team_advantage[venue] = {}
        
        for team in data['team1'].unique():
            team_venue_matches = venue_matches[
                (venue_matches['team1'] == team) | (venue_matches['team2'] == team)
            ]
            if len(team_venue_matches) > 0:
                team_venue_wins = len(team_venue_matches[team_venue_matches['winner'] == team])
                venue_team_advantage[venue][team] = team_venue_wins / len(team_venue_matches)
            else:
                venue_team_advantage[venue][team] = 0.5
    
    # Add venue advantage features
    def get_venue_advantage(row, team_col):
        venue = row['venue']
        team = row[team_col]
        try:
            return venue_team_advantage.get(venue, {}).get(team, 0.5)
        except:
            return 0.5
    
    X['team1_venue_advantage'] = X.apply(lambda row: get_venue_advantage(row, 'team1'), axis=1)
    X['team2_venue_advantage'] = X.apply(lambda row: get_venue_advantage(row, 'team2'), axis=1)
    X['venue_advantage_diff'] = X['team1_venue_advantage'] - X['team2_venue_advantage']
    
    # Print feature importance after model training to see if these help
    print("Added engineered features:", [col for col in X.columns if col not in data.columns])

    # Split the data into training and testing sets (70% training, 30% testing)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Display the shapes of the training and testing sets
    print("X_train shape:", X_train.shape)
    print("y_train shape:", y_train.shape)

    # Create an instance of the LGBMClassifier model with improved parameters
    model = lgb.LGBMClassifier(
        boosting_type='gbdt', 
        num_leaves=31, 
        max_depth=5,  # Limiting depth to prevent overfitting
        learning_rate=0.05,  # Lower learning rate for better generalization
        n_estimators=200,  # More trees for better performance
        subsample=0.8,  # Use 80% of data for each tree to reduce overfitting
        colsample_bytree=0.8,  # Use 80% of features for each tree
        reg_alpha=0.1,  # L1 regularization
        reg_lambda=0.1,  # L2 regularization
        min_child_samples=20,  # Minimum samples in each leaf
        objective='multiclass',
        class_weight='balanced',  # Handle class imbalance
        verbosity=-1  # Set verbosity in constructor, not in fit method
    )

    # Fit the model on the training data without early stopping or verbose parameter
    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        eval_metric='multi_logloss'
    )

    # Calculate feature importance
    feature_importance = pd.DataFrame({
        'feature': X_train.columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("\nTop 10 most important features:")
    print(feature_importance.head(10))

    # Evaluate the model
    from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, brier_score_loss, classification_report, confusion_matrix
    from sklearn.model_selection import cross_val_score
    import numpy as np
    
    # Perform cross-validation to get a better estimate of model performance
    print("\nPerforming 5-fold cross-validation...")
    cv_scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')
    print(f"Cross-validation accuracy scores: {cv_scores}")
    print(f"Mean CV accuracy: {cv_scores.mean():.4f} (±{cv_scores.std():.4f})")
    
    # Make predictions on the test set
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)
    
    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    
    # Calculate F1 score (macro average for multiclass)
    f1 = f1_score(y_test, y_pred, average='macro')
    
    # Calculate ROC AUC (for multiclass using one-vs-rest)
    # First prepare one-hot encoded target
    from sklearn.preprocessing import label_binarize
    classes = np.unique(y)
    y_test_bin = label_binarize(y_test, classes=classes)
    
    # Calculate micro-average ROC AUC 
    try:
        roc_auc = roc_auc_score(y_test_bin, y_pred_proba, multi_class='ovr', average='macro')
    except:
        roc_auc = "Not applicable (insufficient samples)"
    
    # Calculate Brier score (scaled for multiclass)
    # For multiclass, we use the average of Brier scores for each class
    try:
        # Convert predictions to one-hot encoding
        y_pred_bin = label_binarize(y_pred, classes=classes)
        brier = np.mean([brier_score_loss(y_test_bin[:, i], y_pred_proba[:, i]) for i in range(len(classes))])
    except:
        brier = "Not applicable (insufficient samples)"
    
    # Calculate Gini coefficient
    try:
        gini = 2 * roc_auc - 1 if isinstance(roc_auc, float) else "Not applicable"
    except:
        gini = "Not applicable"
    
    # Generate confusion matrix to better understand classification errors
    conf_matrix = confusion_matrix(y_test, y_pred)
    
    # Print model evaluation metrics
    print("\nModel Evaluation Metrics (on Test Data):")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"F1 Score (macro): {f1:.4f}")
    print(f"ROC AUC Score: {roc_auc}")
    print(f"Brier Score: {brier}")
    print(f"Gini Coefficient: {gini}")
    
    # Print classification report
    print("\nClassification Report:")
    target_names = [team_mapping_reverse.get(c, f"Class {c}") for c in classes]
    report = classification_report(y_test, y_pred, target_names=target_names)
    print(report)
    
    # Print confusion matrix
    print("\nConfusion Matrix:")
    print(conf_matrix)
    
    # Note on accuracy for unlabeled data
    print("\nNote: Accuracy for unlabeled data is estimated using test set performance.")
    print("True performance on new data can only be assessed after matches are played.")

    # Map user input to numerical forms based on the mappings
    city_numeric = city_mapping.get(City, -1)
    venue_numeric = venue_mapping.get(Venue, -1)
    team1_numeric = team_mapping.get(Team1,-1)
    team2_numeric = team_mapping.get(Team2,-1)
    toss_winner_numeric = team_mapping.get(Toss_winner,-1)
    toss_decision_numeric = temp.get(Toss_decision,-1)
    umpire1_numeric = umpire_dict.get(Umpire1,-1)
    umpire2_numeric = umpire_dict.get(Umpire2,-1)

    # Prepare user data for prediction
    user_data_dict = {
        'city': [city_numeric],
        'player_of_match': [default_player_value],  # Use a default value
        'venue': [venue_numeric],
        'neutral_venue': [Neutral_venue],
        'team1': [team1_numeric],
        'team2': [team2_numeric],
        'toss_winner': [toss_winner_numeric],
        'toss_decision': [toss_decision_numeric],
        'result_margin': [mean_result_margin if 'result_margin' in data.columns else 10],  # Use mean or default
        'umpire1': [umpire1_numeric],
        'umpire2': [umpire2_numeric]
    }
    
    # Add any additional columns that are in X_train but not in user_data_dict
    for col in X_train.columns:
        if col not in user_data_dict:
            print(f"Adding missing column to user data: {col}")
            user_data_dict[col] = [0]  # Default value
    
    # Create user DataFrame with same columns as X_train
    user_data = pd.DataFrame(user_data_dict)
    
    # Add the same engineered features for user prediction
    # 1. Team win rates
    user_data['team1_win_rate'] = team_stats.get(team1_numeric, 0.5)
    user_data['team2_win_rate'] = team_stats.get(team2_numeric, 0.5)
    user_data['win_rate_diff'] = user_data['team1_win_rate'] - user_data['team2_win_rate']
    
    # 2. Head-to-head matchup
    try:
        user_data['head_to_head'] = head_to_head.get(team1_numeric, {}).get(team2_numeric, 0.5)
    except:
        user_data['head_to_head'] = 0.5
    
    # 3. Toss advantage
    user_data['toss_decision_advantage'] = toss_advantage.get(toss_decision_numeric, 0.5)
    user_data['is_toss_winner_team1'] = 1 if toss_winner_numeric == team1_numeric else 0
    user_data['toss_win_value'] = user_data['is_toss_winner_team1'] * user_data['toss_decision_advantage']
    
    # 4. Venue advantage
    try:
        user_data['team1_venue_advantage'] = venue_team_advantage.get(venue_numeric, {}).get(team1_numeric, 0.5)
        user_data['team2_venue_advantage'] = venue_team_advantage.get(venue_numeric, {}).get(team2_numeric, 0.5)
    except:
        user_data['team1_venue_advantage'] = 0.5
        user_data['team2_venue_advantage'] = 0.5
        
    user_data['venue_advantage_diff'] = user_data['team1_venue_advantage'] - user_data['team2_venue_advantage']
    
    # Ensure user_data has the same columns in the same order as X_train
    user_data = user_data.reindex(columns=X_train.columns, fill_value=0)
    
    print("User data columns:", user_data.columns.tolist())
    print("User data types:", user_data.dtypes)

    # Make predictions on the user input data
    predicted_class = model.predict(user_data)[0]
    
    # Get probabilities for all classes
    proba = model.predict_proba(user_data)[0]
    
    # Find indices of team1 and team2 in the class labels
    team1_idx = None
    team2_idx = None
    
    for i, label in enumerate(model.classes_):
        if label == team1_numeric:
            team1_idx = i
        if label == team2_numeric:
            team2_idx = i
    
    # Get winning probability
    win_probability = {}
    
    if team1_idx is not None:
        win_probability[Team1] = proba[team1_idx] * 100
    else:
        win_probability[Team1] = 0
        
    if team2_idx is not None:
        win_probability[Team2] = proba[team2_idx] * 100
    else:
        win_probability[Team2] = 0
    
    # Calculate draw probability (100% minus the sum of both team win probabilities)
    # Cap it at a minimum of 0%
    draw_probability = max(0, 100 - (win_probability[Team1] + win_probability[Team2]))
    
    # Determine the predicted winner
    predicted_winner = team_mapping_reverse.get(predicted_class, "Unknown team")
    
    # Create an explanation for why this team is predicted to win
    explanation = ""
    if predicted_winner == Team1:
        winning_team = Team1
        losing_team = Team2
        venue_advantage = user_data['team1_venue_advantage'].values[0] > user_data['team2_venue_advantage'].values[0]
        head_to_head_advantage = user_data['head_to_head'].values[0] > 0.5
        toss_advantage = toss_winner_numeric == team1_numeric
    else:
        winning_team = Team2
        losing_team = Team1
        venue_advantage = user_data['team2_venue_advantage'].values[0] > user_data['team1_venue_advantage'].values[0]
        head_to_head_advantage = user_data['head_to_head'].values[0] < 0.5
        toss_advantage = toss_winner_numeric == team2_numeric
    
    reasons = []
    
    # Check historical performance
    if winning_team == Team1 and user_data['team1_win_rate'].values[0] > user_data['team2_win_rate'].values[0]:
        reasons.append(f"{winning_team} has a better overall winning record in the tournament")
    elif winning_team == Team2 and user_data['team2_win_rate'].values[0] > user_data['team1_win_rate'].values[0]:
        reasons.append(f"{winning_team} has a better overall winning record in the tournament")
    
    # Check head to head
    if head_to_head_advantage:
        reasons.append(f"{winning_team} has a stronger head-to-head record against {losing_team}")
    
    # Check venue advantage
    if venue_advantage:
        reasons.append(f"{winning_team} has historically performed better at {Venue}")
    
    # Check toss advantage
    if toss_advantage:
        reasons.append(f"Winning the toss and choosing to {Toss_decision} gives {winning_team} an advantage")
    
    # If no specific reasons found, add a general one
    if not reasons:
        reasons.append(f"Based on the overall statistical analysis of past performance")
    
    explanation = "This prediction is based on several factors: " + ", ".join(reasons) + "."
    
    # Format the prediction string with enhanced HTML for better visual display
    prediction_html = f"""
    <div class="prediction-container">
        <div class="team-prediction-row">
            <div class="team-box team1" data-team="{Team1}">
                <span class="team-name">{Team1}</span>
                <div class="probability-bar" style="width: {min(win_probability[Team1], 100)}%"></div>
                <span class="win-percentage">{win_probability[Team1]:.1f}%</span>
            </div>
            
            <div class="vs-container">VS</div>
            
            <div class="team-box team2" data-team="{Team2}">
                <span class="team-name">{Team2}</span>
                <div class="probability-bar" style="width: {min(win_probability[Team2], 100)}%"></div>
                <span class="win-percentage">{win_probability[Team2]:.1f}%</span>
            </div>
        </div>
        
        <div class="draw-box">
            <span class="draw-label">Draw Probability:</span>
            <span class="draw-percentage">{draw_probability:.1f}%</span>
        </div>
        
        <div class="prediction-winner">
            <span class="winner-label">Predicted Winner:</span>
            <span class="winner-name">{predicted_winner}</span>
        </div>
        
        <div class="prediction-explanation">
            <p>{explanation}</p>
        </div>
        
        <div class="prediction-metrics">
            <h3 class="metrics-title">Model Performance Metrics</h3>
            <div class="metrics-grid">
                <div class="metric-box">
                    <span class="metric-label">Cross-validation Accuracy:</span>
                    <span class="metric-value">{cv_scores.mean():.2f} (±{cv_scores.std():.2f})</span>
                </div>
                
                <div class="metric-box">
                    <span class="metric-label">Test Set Accuracy:</span>
                    <span class="metric-value">{accuracy:.2f}</span>
                </div>
                
                <div class="metric-box">
                    <span class="metric-label">F1 Score:</span>
                    <span class="metric-value">{f1:.2f}</span>
                </div>
    """
    
    # Only add other metrics if they are numeric
    if isinstance(roc_auc, float):
        prediction_html += f"""
                <div class="metric-box">
                    <span class="metric-label">ROC AUC:</span>
                    <span class="metric-value">{roc_auc:.2f}</span>
                </div>
        """
    if isinstance(brier, float):
        prediction_html += f"""
                <div class="metric-box">
                    <span class="metric-label">Brier Score:</span>
                    <span class="metric-value">{brier:.4f} <span class="note">(lower is better)</span></span>
                </div>
        """
    if isinstance(gini, float):
        prediction_html += f"""
                <div class="metric-box">
                    <span class="metric-label">Gini Coefficient:</span>
                    <span class="metric-value">{gini:.2f}</span>
                </div>
        """
        
    prediction_html += """
            </div>
        </div>
    """
    
    # Add key deciding factors section
    prediction_html += """
        <div class="deciding-factors">
            <h3 class="factors-title">Key Deciding Factors</h3>
            <ul class="factors-list">
                <li class="factor">
                    <i class="fas fa-handshake"></i>
                    <span>Historical matchups between these teams</span>
                </li>
                <li class="factor">
                    <i class="fas fa-chart-line"></i>
                    <span>Overall team performance record</span>
                </li>
                <li class="factor">
                    <i class="fas fa-map-marker-alt"></i>
                    <span>Team performance at this venue</span>
                </li>
                <li class="factor">
                    <i class="fas fa-coins"></i>
                    <span>Toss winner and decision</span>
                </li>
            </ul>
        </div>
    """
    
    prediction_html += """
    </div>
    """

    # At the end, before return statement, check if it's a JSON request
    if request.is_json:
        # For JSON requests, return JSON response instead of HTML
        if team1_idx is not None and team2_idx is not None:
            team1_prob = proba[team1_idx] * 100
            team2_prob = proba[team2_idx] * 100
        else:
            team1_prob = 0
            team2_prob = 0
            
        return jsonify({
            'team1': Team1,
            'team2': Team2,
            'prediction': predicted_winner,
            'team1_probability': f"{team1_prob:.1f}%",
            'team2_probability': f"{team2_prob:.1f}%",
            'explanation': explanation
        })
    else:
        # For form submissions, return HTML as before
        return render_template('match_predict.html', prediction=prediction_html)

if __name__ == '__main__':
    # This block of code runs the Flask application when the script is executed directly.

    # The app.run() function starts the Flask development server.
    # - 'debug=True' enables the debug mode, which provides helpful error messages during development.
    # - 'host='0.0.0.0'' makes the app accessible from all network interfaces, allowing external access.
    app.run(debug=True, host='0.0.0.0')








