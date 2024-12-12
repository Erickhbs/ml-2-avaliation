from flask import Flask, request, jsonify, render_template, redirect, url_for
import joblib
import numpy as np
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")


app = Flask(__name__)

# Carregar o modelo salvo
model = joblib.load('modelo.pkl')

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/result')
def result():
    predicted_value = request.args.get('predicted_value', None)
    if predicted_value is None:
        return "Erro: Nenhum valor previsto foi enviado.", 400
    return render_template('result.html', predicted_value=predicted_value)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.form

    # Verificar se os dados necessários foram enviados
    required_fields = ['latitude', 'longitude', 'housing_median_age', 'total_rooms', 
                       'total_bedrooms', 'population', 'households', 'median_income', 
                       'rooms_per_household', 'bedrooms_per_room', 'ocean_proximity']
    if not all(field in data for field in required_fields):
        return "Erro: Faltam campos obrigatórios.", 400

    # Mapeamento de valores categóricos para números
    ocean_proximity_mapping = {
        "NEAR OCEAN": 0,
        "INLAND": 1,
        "NEAR BAY": 2,
        "ISLAND": 3
    }

    # Preparar os dados para o modelo
    features = [
        float(data['latitude']),
        float(data['longitude']),
        float(data['housing_median_age']),
        float(data['total_rooms']),
        float(data['total_bedrooms']),
        float(data['population']),
        float(data['households']),
        float(data['median_income']),
        float(data['rooms_per_household']),
        float(data['bedrooms_per_room']),
        ocean_proximity_mapping[data['ocean_proximity']]
    ]

    # Fazer a predição
    prediction = model.predict([features])
    predicted_value = prediction[0]

    # Redirecionar para a página de resultado
    return redirect(url_for('result', predicted_value=predicted_value))

if __name__ == '__main__':
    app.run(debug=True)
