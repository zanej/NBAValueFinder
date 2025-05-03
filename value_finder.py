import pandas as pd
from .predict import predict_probabilities

def implied_prob(odds):
    return 1 / odds

def find_value_bets(odds_data_path='data/todays_odds.csv', model_path='models/logistic_model.pkl'):
    data = predict_probabilities(odds_data_path, model_path)
    data['implied_prob'] = data['moneyline_odds'].apply(implied_prob)
    data['value'] = data['win_prob'] - data['implied_prob']
    value_bets = data[data['value'] > 0.05]  # Threshold for +EV
    return value_bets
