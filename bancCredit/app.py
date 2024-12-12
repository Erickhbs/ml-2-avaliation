from flask import Flask, request, render_template
import joblib
import numpy as np
import pandas as pd

# Inicialização do Flask
app = Flask(__name__)

# Carregar o modelo treinado e o scaler
try:
    model = joblib.load("modelo_final.pkl")  # O modelo treinado
    scaler = joblib.load("scaler.pkl")  # O scaler usado durante o treinamento
except FileNotFoundError:
    raise FileNotFoundError("faltando algum ou um dos .pkl")

# Rota inicial - Página principal
@app.route('/')
def home():
    return render_template('home.html')

# Rota para predição
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Capturar os dados enviados pelo formulário
        gender = int(request.form['gender'])
        dependents = float(request.form['dependents'])
        married = int(request.form['married'])
        self_employed = int(request.form['self_employed'])
        education = int(request.form['education'])
        income = float(request.form['income'])
        loan_amount = float(request.form['loan_amount'])

        # Organizar os dados das variáveis que precisam ser escaladas
        data_num = np.array([[dependents, income, loan_amount]])

        # Aplicar o scaler
        scaled_data = scaler.transform(data_num)

        # Organizar todos os dados, incluindo os valores escalados
        #data = np.array([[gender, married, scaled_data[0, 0], education, self_employed, scaled_data[0, 1], scaled_data[0, 2]]])
        d = {'Gender': [gender], 'Married': [married], 'Dependents': [scaled_data[0, 0]],
             'Education': [education], 'Self_Employed': [self_employed], 'ApplicantIncome': [scaled_data[0, 1]], 'LoanAmount': [scaled_data[0, 2]]}
    
        data = pd.DataFrame(data=d)

        # Realizar a predição
        prediction = model.predict(data)  # 0 ou 1
        probability = model.predict_proba(data)[0][1]  # Probabilidade de 1

        # Traduzir o resultado para exibição
        result = "Sim" if prediction[0] == 1 else "Não"

        # Retornar a página de resultado com os valores calculados
        return render_template('credit.html', result=result, probability=round(probability * 100, 2))

    except Exception as e:
        return f"Erro ao realizar a predição: {str(e)}"

if __name__ == '__main__':
    app.run(debug=True)
