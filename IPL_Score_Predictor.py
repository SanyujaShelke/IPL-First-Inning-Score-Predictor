from flask import Flask, render_template, request, jsonify
import pickle

app = Flask(__name__)

# Load the trained model and label encodings
with open('ipl_model.pkl', 'rb') as model_file, open('label_encodings.pkl', 'rb') as label_encodings_file:
    model = pickle.load(model_file)
    label_encodings = pickle.load(label_encodings_file)

@app.route('/')
def home():
    return render_template('IPL_Score_Predictor.html')  # Your HTML file path here

@app.route('/predict', methods=['POST'])
def predict():
    # Get user inputs from the form
    venue = request.form['venue']
    bat_team = request.form['bat_team']
    bowl_team = request.form['bowl_team']
    batsman = request.form['batsman']
    bowler = request.form['bowler']
    runs = int(request.form['runs'])
    wickets = int(request.form['wickets'])
    overs = float(request.form['overs'])
    runs_last_5 = int(request.form['runs_last_5'])
    wickets_last_5 = int(request.form['wickets_last_5'])
    striker = int(request.form['striker'])
    non_striker = int(request.form['non-striker'])

    # Check if the user input exists in label encodings
    # if venue not in label_encodings['venue'] or bat_team not in label_encodings['bat_team'] \
    #         or bowl_team not in label_encodings['bowl_team'] or batsman not in label_encodings['batsman'] \
    #         or bowler not in label_encodings['bowler']:
    #     return "Invalid input. Please check your input values."


    # Convert user inputs to encoded values using label encodings
    venue_encoded = label_encodings['venue'][venue]
    bat_team_encoded = label_encodings['bat_team'][bat_team]
    bowl_team_encoded = label_encodings['bowl_team'][bowl_team]
    batsman_encoded = label_encodings['batsman'][batsman]
    bowler_encoded = label_encodings['bowler'][bowler]

    # Create a new input data point with these encoded values
    new_input = [[venue_encoded, bat_team_encoded, bowl_team_encoded, batsman_encoded, bowler_encoded, runs, wickets, overs, runs_last_5, wickets_last_5, striker, non_striker]]

    # Predict using your trained model
    prediction = model.predict(new_input)
    prediction = int(prediction[0])

    # Return the predicted score to a separate result page or a section on the same page
    return render_template('IPL_Score_Predictor.html', prediction_result=prediction)

if __name__ == '__main__':
    app.run(debug=True)
