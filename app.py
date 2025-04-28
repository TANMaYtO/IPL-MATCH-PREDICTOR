from flask import Flask, render_template, request
import joblib
import pandas as pd
import numpy as np
import random

app = Flask(__name__)

# Load model and column structure
model = joblib.load('ipl_model_new.lb')
column_structure = joblib.load(r'D:\PROJECTs\NEW IPL\final_data_new.pkl')

@app.route('/', methods=['GET'])
def home():
    return render_template('home.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        venue = request.form['venue']
        team1 = request.form['team1']
        team2 = request.form['team2']
        toss_winner = request.form['toss_winner']
        toss_decision = request.form['toss_decision']

        if not all([venue, team1, team2, toss_winner, toss_decision]):
            return render_template('home.html', prediction_text="⚠️ Please fill all fields before submitting!")

        # Random shuffle teams like training
        teams = [team1, team2]
        random.shuffle(teams)
        teamA, teamB = teams[0], teams[1]

        # Create input DataFrame
        input_data = pd.DataFrame([np.zeros(len(column_structure))], columns=column_structure)

        for value in [f'teamA_{teamA}', f'teamB_{teamB}', f'venue_{venue}', f'teamA_toss_decision_{toss_decision}']:
            if value in input_data.columns:
                input_data.at[0, value] = 1

        input_data.at[0, 'teamA_toss_win'] = 1 if toss_winner == teamA else 0

        # Predict with probabilities
        probabilities = model.predict_proba(input_data)[0]
        prediction = model.predict(input_data)[0]

        # Map prediction
        if prediction == "teamA":
            predicted_winner = teamA
            confidence = probabilities[model.classes_.tolist().index('teamA')] * 100
        else:
            predicted_winner = teamB
            confidence = probabilities[model.classes_.tolist().index('teamB')] * 100

        return render_template('home.html', prediction_text=f'🏆 Predicted Winner: {predicted_winner} ({confidence:.1f}% chance)')

    except Exception as e:
        return render_template('home.html', prediction_text=f"❌ Error: {str(e)}")

if __name__ == '__main__':
    app.run(debug=True)
