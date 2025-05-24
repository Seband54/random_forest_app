from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)
model = joblib.load('model.pkl')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        try:
            features = [float(request.form[f]) for f in ['embarazos', 'glucosa', 'presion', 'pliegue', 'insulina', 'imc', 'funcion', 'edad']]
            prediction = model.predict([features])[0]
            resultado = "Alto Riesgo de Diabetes" if prediction == 1 else "Bajo Riesgo de Diabetes"

        except:
            resultado = "Error en los datos ingresados"
        return render_template('index.html', resultado=resultado)

if __name__ == '__main__':
    app.run(debug=True)
