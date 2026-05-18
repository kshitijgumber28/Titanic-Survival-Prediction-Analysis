from flask import Flask, request
import joblib
import pandas as pd

app = Flask(__name__)

# Load trained pipeline
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

model_path = os.path.join(BASE_DIR, 'titanic_pipeline.pkl')

model = joblib.load(model_path)

# Prediction mapping
survival = {
    0: "Not Survived ❌",
    1: "Survived ✅"
}

@app.route('/')
def home():
    return '''
    <h1>Titanic Survival Predictor 🚢</h1>
    <form action="/predict" method="post">

        Passenger Class (1,2,3): <input name="pclass" required><br>
        Sex (male/female): <input name="sex" required><br>
        Age: <input name="age" required><br>
        Fare: <input name="fare" required><br>
        Embarked (S/C/Q): <input name="embarked" required><br>

        <button type="submit">Predict</button>
    </form>
    '''

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Read inputs
        pclass = int(request.form['pclass'])
        sex = request.form['sex'].lower()
        age = float(request.form['age'])
        fare = float(request.form['fare'])
        embarked = request.form['embarked'].upper()

        # Create DataFrame EXACTLY like training data
        input_df = pd.DataFrame({
            'Pclass': [pclass],
            'Sex': [sex],
            'Age': [age],
            'Fare': [fare],
            'Embarked': [embarked]
        })

        prediction = model.predict(input_df)
        result = survival[prediction[0]]

        return f'''
        <h2>Prediction: {result}</h2>
        <a href="/">Try Again</a>
        '''

    except Exception as e:
        return f'''
        <h3>Error: Invalid input</h3>
        <p>{e}</p>
        <a href="/">Go Back</a>
        '''

if __name__ == '__main__':
    app.run(debug=True)

