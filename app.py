from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import locale

app = Flask(__name__)
CORS(app)  # This will enable CORS for all routes

# Set the locale for Indian numbering system
locale.setlocale(locale.LC_ALL, 'en_IN.UTF-8')

# Load the model
model = joblib.load('salary_prediction_model.pkl')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    age = int(data['Age'])
    gender = int(data['Gender'])  # Assuming gender is encoded as 0 or 1
    degree = data['degree']
    job_title = int(data['Job_Title'])
    experience = int(data['Experience'])

    # Convert categorical data to numeric if needed
    # You may need to use a pre-trained label encoder or map the values manually
    # Example:
    degree_mapping = {'Bachelor': 0, 'Master': 1, 'PhD': 2}
  
    degree = degree_mapping.get(degree, -1)
    # job_title = job_title_mapping.get(job_title, -1)

   

    # Predict the salary based on input features
    prediction = model.predict([[age, gender, degree, job_title, experience]])[0]

    # Format the prediction to two decimal places and add commas
    formatted_prediction = locale.format_string("%0.2f", prediction, grouping=True)
    
    return jsonify({'salary': formatted_prediction})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')
