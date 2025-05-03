import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.calibration import CalibratedClassifierCV
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from sklearn.metrics import brier_score_loss, accuracy_score, confusion_matrix
from flask import Flask, request, render_template
import pickle
import os
import matplotlib.pyplot as plt

# Define home venues globally
home_venues = {
    'Chennai Super Kings': ['MA Chidambaram Stadium, Chepauk'],
    'Royal Challengers Bangalore': ['M Chinnaswamy Stadium'],
    'Mumbai Indians': ['Wankhede Stadium'],
    'Kolkata Knight Riders': ['Eden Gardens'],
    'Delhi Capitals': ['Arun Jaitley Stadium'],
    'Sunrisers Hyderabad': ['Rajiv Gandhi International Stadium, Uppal'],
    'Rajasthan Royals': ['Sawai Mansingh Stadium'],
    'Punjab Kings': ['Punjab Cricket Association IS Bindra Stadium, Mohali']
}

# Define a color mapping for all teams
team_colors = {
    'Chennai Super Kings': '#FFC107',  # Yellow
    'Royal Challengers Bangalore': '#D32F2F',  # Red
    'Mumbai Indians': '#0288D1',  # Blue
    'Kolkata Knight Riders': '#7B1FA2',  # Purple
    'Delhi Capitals': '#D81B60',  # Pink
    'Sunrisers Hyderabad': '#F57C00',  # Orange
    'Rajasthan Royals': '#C2185B',  # Deep Pink
    'Punjab Kings': '#0288D1'  # Light Blue
}

app = Flask(__name__)

# Load datasets
try:
    matches = pd.read_csv('matches.csv')
    deliveries = pd.read_csv('deliveries.csv')
    print("Datasets loaded successfully.")
except FileNotFoundError as e:
    print(f"Error: {e}. Ensure 'matches.csv' and 'deliveries.csv' are in the project directory.")
    exit(1)

# Debug: Check column names and missing values
print("Matches columns:", matches.columns.tolist())
print("Deliveries columns:", deliveries.columns.tolist())
print("Missing values in matches:", matches.isnull().sum())
print("Missing values in deliveries:", deliveries.isnull().sum())
if 'venue' not in matches.columns:
    print("Error: 'venue' column not found in matches.csv.")
    exit(1)
if 'match_id' not in deliveries.columns or 'id' not in matches.columns:
    print("Error: 'match_id' or 'id' column missing for merging.")
    exit(1)

# Clean data: Drop rows with missing critical columns
matches = matches.dropna(subset=['winner', 'team1', 'team2', 'venue'])

# Define recent win ratio function globally
def calculate_recent_win_ratio(team, match_id, matches, n_matches=5):
    """Calculate win ratio for a team based on its last n_matches."""
    if match_id is None:  # For predictions, use latest matches
        team_matches = matches[
            (matches['team1'] == team) | (matches['team2'] == team)
        ].tail(n_matches)
    else:
        team_matches = matches[
            ((matches['team1'] == team) | (matches['team2'] == team)) & (matches['id'] < match_id)
        ].tail(n_matches)
    wins = team_matches[team_matches['winner'] == team].shape[0]
    return wins / max(1, team_matches.shape[0])

# Updated to use dynamic team color mapping
def create_win_probability_plot(batting_team, bowling_team, batting_prob, bowling_prob):
    """Generate a bar plot of win probabilities for the two teams."""
    teams = [batting_team, bowling_team]
    probabilities = [batting_prob, bowling_prob]
    colors = [team_colors.get(team, '#2196F3') for team in teams]  # Use team_colors, fallback to blue
    
    plt.figure(figsize=(8, 6))
    bars = plt.bar(teams, probabilities, color=colors, edgecolor='black')
    plt.title('Win Probability for Each Team', fontsize=14)
    plt.ylabel('Probability (%)', fontsize=12)
    plt.ylim(0, 100)
    
    # Add percentage labels on top of bars
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + 2, f'{yval:.1f}%', ha='center', fontsize=10)
    
    plt.tight_layout()
    plot_path = 'static/plot.png'
    plt.savefig(plot_path)
    plt.close()
    return plot_path

# Data Preprocessing and Feature Engineering
def preprocess_data(matches, deliveries):
    """Preprocess matches and deliveries data for training the prediction model."""
    deliveries = deliveries.merge(matches[['id', 'venue']], left_on='match_id', right_on='id', how='left')
    print("Merged DataFrame columns:", deliveries.columns.tolist())
    if 'venue' not in deliveries.columns:
        print("Error: 'venue' column not found in merged DataFrame.")
        exit(1)
    print("Merged venue null count:", deliveries['venue'].isnull().sum())
    
    deliveries_2nd = deliveries[deliveries['inning'] == 2].copy()
    if deliveries_2nd['venue'].isnull().any():
        print("Warning: Missing venue data in some rows. Dropping rows with null venues.")
        deliveries_2nd = deliveries_2nd.dropna(subset=['venue'])

    deliveries_2nd['over_ball'] = deliveries_2nd['over'] + deliveries_2nd['ball'] / 6
    deliveries_2nd['cum_runs'] = deliveries_2nd.groupby('match_id')['total_runs'].cumsum()
    deliveries_2nd['wickets'] = deliveries_2nd['is_wicket'].astype(int)
    deliveries_2nd['cum_wickets'] = deliveries_2nd.groupby('match_id')['wickets'].cumsum()

    over_points = [1, 5, 10, 15, 20]
    training_data = []
    for match_id in deliveries_2nd['match_id'].unique():
        match_data = deliveries_2nd[deliveries_2nd['match_id'] == match_id]
        batting_team = match_data['batting_team'].iloc[0]
        bowling_team = match_data['bowling_team'].iloc[0]
        venue = match_data['venue'].iloc[0]
        toss_decision_row = matches[matches['id'] == match_id]['toss_decision']
        toss_decision = toss_decision_row.iloc[0] if not toss_decision_row.empty else 'field'
        final_runs = match_data['total_runs'].sum()
        for over in over_points:
            subset = match_data[match_data['over_ball'] <= over]
            if not subset.empty:
                runs_at_over = subset['total_runs'].sum()
                wickets_at_over = subset['wickets'].sum()
                remaining_wickets = 10 - wickets_at_over
                training_data.append({
                    'match_id': match_id,
                    'batting_team': batting_team,
                    'bowling_team': bowling_team,
                    'overs': over,
                    'runs_at_over': runs_at_over,
                    'wickets_at_over': wickets_at_over,
                    'remaining_wickets': remaining_wickets,
                    'venue': venue,
                    'toss_decision': toss_decision,
                    'final_runs': final_runs
                })

    training_df = pd.DataFrame(training_data)
    matches_subset = matches[['id', 'team1', 'team2', 'winner', 'venue', 'toss_winner', 'toss_decision']]
    data = training_df.merge(matches_subset, left_on='match_id', right_on='id', how='left', suffixes=('', '_matches'))

    if data['venue'].isnull().any() or data['winner'].isnull().any():
        print("Warning: Dropping rows with null venue or winner after merge.")
        data = data.dropna(subset=['venue', 'winner'])

    data['target'] = (data['batting_team'] == data['winner']).astype(int)
    data['current_rr'] = data['runs_at_over'] / data['overs']
    data['remaining_overs'] = 20 - data['overs']
    data['required_rr'] = np.where(
        data['remaining_overs'] > 0,
        (data['final_runs'] - data['runs_at_over']) / data['remaining_overs'],
        data['current_rr']
    )
    team_wins = matches['winner'].value_counts().to_dict()
    team_matches = pd.concat([matches['team1'], matches['team2']]).value_counts().to_dict()
    data['batting_team_win_ratio'] = data['batting_team'].map(
        lambda x: team_wins.get(x, 0) / team_matches.get(x, 1) if x in team_matches else 0.5
    )
    data['bowling_team_win_ratio'] = data['bowling_team'].map(
        lambda x: team_wins.get(x, 0) / team_matches.get(x, 1) if x in team_matches else 0.5
    )

    data['batting_team_recent_win_ratio'] = data.apply(
        lambda x: calculate_recent_win_ratio(x['batting_team'], x['match_id'], matches), axis=1
    )
    data['bowling_team_recent_win_ratio'] = data.apply(
        lambda x: calculate_recent_win_ratio(x['bowling_team'], x['match_id'], matches), axis=1
    )

    data['is_home_match'] = data.apply(
        lambda x: 1 if x['venue'] in home_venues.get(x['batting_team'], []) else 0, axis=1
    )
    home_wins = matches[matches.apply(lambda x: x['venue'] in home_venues.get(x['team1'], []) and x['winner'] == x['team1'], axis=1)]['winner'].value_counts()
    home_matches = matches[matches.apply(lambda x: x['venue'] in home_venues.get(x['team1'], []) or x['venue'] in home_venues.get(x['team2'], []), axis=1)]['team1'].value_counts()
    data['home_win_rate'] = data['batting_team'].map(
        lambda x: home_wins.get(x, 0) / home_matches.get(x, 1) if x in home_matches else 0
    )
    venue_runs = deliveries[deliveries['inning'] == 1].groupby(['match_id', 'venue'])['total_runs'].sum().groupby('venue').mean().to_dict()
    data['venue_avg_runs'] = data['venue'].map(venue_runs).fillna(np.mean(list(venue_runs.values())))

    rr_ratio = data['required_rr'] / data['current_rr'].replace(0, np.nan)
    rr_ratio = rr_ratio.fillna(1)
    data['target_difficulty'] = np.clip(rr_ratio, 0, 2)

    le_team = LabelEncoder()
    le_venue = LabelEncoder()
    le_toss = LabelEncoder()
    data['batting_team'] = le_team.fit_transform(data['batting_team'])
    data['bowling_team'] = le_team.transform(data['bowling_team'])
    data['venue'] = le_venue.fit_transform(data['venue'])
    data['toss_decision'] = le_toss.fit_transform(data['toss_decision'])

    try:
        with open('le_team.pkl', 'wb') as f:
            pickle.dump(le_team, f)
        with open('le_venue.pkl', 'wb') as f:
            pickle.dump(le_venue, f)
        with open('le_toss.pkl', 'wb') as f:
            pickle.dump(le_toss, f)
        print("Encoders saved successfully.")
    except PermissionError as e:
        print(f"Error saving encoders: {e}. Check directory permissions.")
        exit(1)

    features = [
        'batting_team', 'bowling_team', 'runs_at_over', 'wickets_at_over', 'remaining_wickets',
        'current_rr', 'required_rr', 'batting_team_win_ratio', 'bowling_team_win_ratio',
        'batting_team_recent_win_ratio', 'bowling_team_recent_win_ratio',
        'is_home_match', 'home_win_rate', 'venue', 'venue_avg_runs', 'toss_decision', 'target_difficulty'
    ]
    X = data[features]
    y = data['target']

    print("Training data size:", len(data))
    print("Training features:", features)
    print("Class distribution:", data['target'].value_counts(normalize=True))
    if data['target'].value_counts(normalize=True).max() > 0.7:
        print("Warning: Severe class imbalance detected. Consider undersampling or adjusting SMOTE.")

    return deliveries_2nd, data, le_team.classes_, le_venue.classes_, le_toss.classes_

def train_model(X, y):
    """Train the RandomForestClassifier with calibration and save the model."""
    X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.25, random_state=42)
    undersampler = RandomUnderSampler(random_state=42)
    X_train_under, y_train_under = undersampler.fit_resample(X_train, y_train)
    smote = SMOTE(random_state=42, k_neighbors=3)
    X_train_balanced, y_train_balanced = smote.fit_resample(X_train_under, y_train_under)
    base_model = RandomForestClassifier(
        n_estimators=200,
        max_depth=6,
        min_samples_leaf=10,
        class_weight='balanced',
        random_state=42
    )
    model = CalibratedClassifierCV(base_model, method='sigmoid', cv=15)
    model.fit(X_train_balanced, y_train_balanced)
    
    y_val_pred = model.predict(X_val)
    y_val_prob = model.predict_proba(X_val)[:, 1]
    print("Validation Accuracy:", accuracy_score(y_val, y_val_pred))
    print("Validation Confusion Matrix:\n", confusion_matrix(y_val, y_val_pred))
    print("Validation Brier score:", brier_score_loss(y_val, y_val_prob))
    
    y_test_pred = model.predict(X_test)
    y_test_prob = model.predict_proba(X_test)[:, 1]
    print("Test Accuracy:", accuracy_score(y_test, y_test_pred))
    print("Test Confusion Matrix:\n", confusion_matrix(y_test, y_test_pred))
    print("Test Brier score:", brier_score_loss(y_test, y_test_prob))
    print("Test Probability distribution:", np.histogram(y_test_prob, bins=10)[0])
    print("Feature importances:", dict(zip(X.columns, model.calibrated_classifiers_[0].estimator.feature_importances_)))
    
    try:
        with open('model.pkl', 'wb') as f:
            pickle.dump(model, f)
        print("Model saved successfully.")
    except PermissionError as e:
        print(f"Error saving model: {e}. Check directory permissions.")
        exit(1)
    return model

def test_predictions(model, data, le_team, le_venue, le_toss, n_samples=100):
    """Test model predictions on a sample of historical data."""
    sample_data = data.sample(n=min(n_samples, len(data)), random_state=42)
    X_test = sample_data[[
        'batting_team', 'bowling_team', 'runs_at_over', 'wickets_at_over', 'remaining_wickets',
        'current_rr', 'required_rr', 'batting_team_win_ratio', 'bowling_team_win_ratio',
        'batting_team_recent_win_ratio', 'bowling_team_recent_win_ratio',
        'is_home_match', 'home_win_rate', 'venue', 'venue_avg_runs', 'toss_decision', 'target_difficulty'
    ]]
    y_test = sample_data['target']
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    print("Test Sample Accuracy:", accuracy_score(y_test, y_pred))
    print("Test Sample Brier score:", brier_score_loss(y_test, y_prob))
    print("Sample Probability Range:", y_prob.min(), y_prob.max())

def load_artifacts():
    """Load the trained model and encoders."""
    try:
        with open('model.pkl', 'rb') as f:
            model = pickle.load(f)
        with open('le_team.pkl', 'rb') as f:
            le_team = pickle.load(f)
        with open('le_venue.pkl', 'rb') as f:
            le_venue = pickle.load(f)
        with open('le_toss.pkl', 'rb') as f:
            le_toss = pickle.load(f)
        print("Loaded model feature count:", model.calibrated_classifiers_[0].estimator.n_features_in_)
        return model, le_team, le_venue, le_toss
    except FileNotFoundError as e:
        print(f"Error: {e}. Run the app to generate model and encoder files.")
        exit(1)

@app.route('/')
def home():
    model, le_team, le_venue, le_toss = load_artifacts()
    print("Teams:", le_team.classes_.tolist())
    print("Venues:", le_venue.classes_.tolist())
    print("Toss decisions:", le_toss.classes_.tolist())
    return render_template('index.html',
                           teams=le_team.classes_,
                           venues=le_venue.classes_,
                           toss_decisions=le_toss.classes_,
                           form_data={})

@app.route('/predict', methods=['POST'])
def predict():
    model, le_team, le_venue, le_toss = load_artifacts()
    form_data = request.form.to_dict()
    
    required_fields = [
        'batting_team', 'bowling_team', 'venue', 'toss_decision',
        'overs', 'runs_at_over', 'wickets_at_over', 'target_runs'
    ]
    
    missing_fields = [field for field in required_fields if field not in request.form or not request.form[field]]
    if missing_fields:
        error_message = f"Missing required fields: {', '.join(missing_fields)}"
        print(error_message)
        return render_template('index.html',
                               prediction=error_message,
                               teams=le_team.classes_,
                               venues=le_venue.classes_,
                               toss_decisions=le_toss.classes_,
                               form_data=form_data)

    try:
        batting_team = request.form['batting_team']
        bowling_team = request.form['bowling_team']
        venue = request.form['venue']
        toss_decision = request.form['toss_decision']
        overs = float(request.form['overs'])
        runs_at_over = float(request.form['runs_at_over'])
        wickets_at_over = int(request.form['wickets_at_over'])
        target_runs = float(request.form['target_runs'])

        if batting_team not in le_team.classes_:
            raise ValueError(f"Invalid batting team: {batting_team}")
        if bowling_team not in le_team.classes_:
            raise ValueError(f"Invalid bowling team: {bowling_team}")
        if venue not in le_venue.classes_:
            raise ValueError(f"Invalid venue: {venue}")
        if toss_decision not in le_toss.classes_:
            raise ValueError(f"Invalid toss decision: {toss_decision}")
        if batting_team == bowling_team:
            raise ValueError("Batting and bowling teams cannot be the same.")
        if overs < 0 or overs > 20:
            raise ValueError("Overs must be between 0 and 20.")
        if wickets_at_over < 0 or wickets_at_over > 10:
            raise ValueError("Wickets must be between 0 and 10.")
        if runs_at_over < 0 or target_runs <= runs_at_over:
            raise ValueError("Invalid runs or target runs (target must be greater than current runs).")

        dataset_teams = set(matches['team1']).union(set(matches['team2']))
        dataset_venues = set(matches['venue'])
        if batting_team not in dataset_teams or bowling_team not in dataset_teams:
            print(f"Warning: Team mismatch. Batting: {batting_team}, Bowling: {bowling_team}, Dataset teams: {dataset_teams}")
        if venue not in dataset_venues:
            print(f"Warning: Venue mismatch. Venue: {venue}, Dataset venues: {dataset_venues}")

        batting_team_enc = le_team.transform([batting_team])[0]
        bowling_team_enc = le_team.transform([bowling_team])[0]
        venue_enc = le_venue.transform([venue])[0]
        toss_decision_enc = le_toss.transform([toss_decision])[0]

        remaining_wickets = 10 - wickets_at_over
        current_rr = runs_at_over / overs if overs > 0 else 0
        remaining_overs = 20 - overs
        required_rr = (target_runs - runs_at_over) / remaining_overs if remaining_overs > 0 else current_rr
        team_wins = matches['winner'].value_counts().to_dict()
        team_matches = pd.concat([matches['team1'], matches['team2']]).value_counts().to_dict()
        batting_team_win_ratio = team_wins.get(batting_team, 0) / team_matches.get(batting_team, 1) if batting_team in team_matches else 0.5
        bowling_team_win_ratio = team_wins.get(bowling_team, 0) / team_matches.get(bowling_team, 1) if bowling_team in team_matches else 0.5
        batting_team_recent_win_ratio = calculate_recent_win_ratio(batting_team, None, matches)
        bowling_team_recent_win_ratio = calculate_recent_win_ratio(bowling_team, None, matches)
        is_home_match = 1 if venue in home_venues.get(batting_team, []) else 0
        home_win_rate = data[data['batting_team'] == batting_team_enc]['home_win_rate'].mean()
        venue_avg_runs = data[data['venue'] == venue_enc]['venue_avg_runs'].mean()
        rr_ratio = required_rr / current_rr if current_rr != 0 else 1
        target_difficulty = np.clip(rr_ratio, 0, 2)

        batting_team_win_ratio = batting_team_win_ratio if pd.notna(batting_team_win_ratio) else 0.5
        bowling_team_win_ratio = bowling_team_win_ratio if pd.notna(bowling_team_win_ratio) else 0.5
        home_win_rate = home_win_rate if pd.notna(home_win_rate) else 0.0
        venue_avg_runs = venue_avg_runs if pd.notna(venue_avg_runs) else np.mean(list(data['venue_avg_runs']))

        input_data = np.array([[
            batting_team_enc, bowling_team_enc, runs_at_over, wickets_at_over, remaining_wickets,
            current_rr, required_rr, batting_team_win_ratio, bowling_team_win_ratio,
            batting_team_recent_win_ratio, bowling_team_recent_win_ratio,
            is_home_match, home_win_rate, venue_enc, venue_avg_runs, toss_decision_enc, target_difficulty
        ]])

        print("Received form data:", request.form.to_dict())
        print("Input data:", input_data.tolist())
        print(f"Target difficulty: {target_difficulty:.4f}")

        prediction = model.predict(input_data)[0]
        prob = model.predict_proba(input_data)[0]
        winner = batting_team if prediction == 1 else bowling_team
        win_prob = min(prob[prediction] * 100, 90.0)  # Cap at 90%
        batting_prob = prob[1] * 100  # Probability batting team wins
        bowling_prob = prob[0] * 100  # Probability bowling team wins

        plot_path = create_win_probability_plot(batting_team, bowling_team, batting_prob, bowling_prob)

        print(f"Prediction: {winner} with probability {win_prob:.2f}%")

        return render_template('index.html',
                               prediction=f"Predicted Winner: {winner} ({win_prob:.2f}% probability)",
                               teams=le_team.classes_,
                               venues=le_venue.classes_,
                               toss_decisions=le_toss.classes_,
                               form_data=form_data,
                               plot_path=plot_path)

    except ValueError as e:
        error_message = f"Invalid input: {str(e)}. Please check your inputs (e.g., teams, venue, toss decision)."
        print(error_message)
        return render_template('index.html',
                               prediction=error_message,
                               teams=le_team.classes_,
                               venues=le_venue.classes_,
                               toss_decisions=le_toss.classes_,
                               form_data=form_data)
    except Exception as e:
        error_message = f"Prediction error: {str(e)}"
        print(error_message)
        return render_template('index.html',
                               prediction=error_message,
                               teams=le_team.classes_,
                               venues=le_venue.classes_,
                               toss_decisions=le_toss.classes_,
                               form_data=form_data)

# NEW: Preprocess data, train model, and test predictions at startup
deliveries_2nd, data, team_classes, venue_classes, toss_classes = preprocess_data(matches, deliveries)
X = data[[
    'batting_team', 'bowling_team', 'runs_at_over', 'wickets_at_over', 'remaining_wickets',
    'current_rr', 'required_rr', 'batting_team_win_ratio', 'bowling_team_win_ratio',
    'batting_team_recent_win_ratio', 'bowling_team_recent_win_ratio',
    'is_home_match', 'home_win_rate', 'venue', 'venue_avg_runs', 'toss_decision', 'target_difficulty'
]]
y = data['target']
model = train_model(X, y)
model, le_team, le_venue, le_toss = load_artifacts()
test_predictions(model, data, le_team, le_venue, le_toss)

# MODIFIED: Configure app to run with Render's host and port
if __name__ == '__main__':
    port = int(os.getenv('PORT', 5000))  # Use PORT from environment, default to 5000
    app.run(host='0.0.0.0', port=port)
