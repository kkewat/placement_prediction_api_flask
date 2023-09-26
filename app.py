from flask import Flask, request, jsonify
import pickle
import numpy as np

model = pickle.load(open('ml_model.pkl','rb'))

app = Flask(__name__)

@app.route('/')

def home():
    return "Hello World"

@app.route('/predict', methods=['POST'])

def predict():
    cgpa = request.form.get('cgpa')
    speaking_skills = request.form.get('speaking_skills')
    ml_knowledge = request.form.get('ml_knowledge')
    input_query = np.array([[cgpa, speaking_skills, ml_knowledge]])
    result = model.predict(input_query)[0]
    return jsonify({'Placement' : str(result)})

if __name__ == '__main__':
    app.run(debug = True)
