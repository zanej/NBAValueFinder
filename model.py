import pandas as pd
from sklearn.linear_model import LogisticRegression
import joblib

def train_model(data_path='data/historical_nba_data.csv', model_path='models/logistic_model.pkl'):
    data = pd.read_csv(data_path)
    X = data[['team_rating_diff', 'home_advantage']]
    y = data['win']
    
    model = LogisticRegression()
    model.fit(X, y)
    
    joblib.dump(model, model_path)
    print("Model trained and saved to", model_path)
