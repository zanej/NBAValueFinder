import pandas as pd
import joblib

def predict_probabilities(odds_data_path='data/todays_odds.csv', model_path='models/logistic_model.pkl'):
    model = joblib.load(model_path)
    odds_data = pd.read_csv(odds_data_path)
    X = odds_data[['team_rating_diff', 'home_advantage']]
    probs = model.predict_proba(X)[:, 1]
    odds_data['win_prob'] = probs
    return odds_data
