import pickle
import os
import numpy as np
from sklearn import preprocessing
from flask import Flask, request, render_template, jsonify
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)

model_x = os.path.abspath('model-flask-lr_x.pkl')
model = pickle.load(open(model_x, 'rb'))

@app.route('/')
def index():
  return render_template('index.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
  if request.method == 'POST':
    SUHU_TUBUH = float(request.form['SUHU_TUBUH'])
    RIWAYAT_KESEHATAN = int(request.form['RIWAYAT_KESEHATAN'])
    MENGGIGIL = int(request.form['MENGGIGIL'])
    DEMAM = int(request.form['DEMAM'])
    TROMBOSIT = int(request.form['TROMBOSIT'])
    MUAL = int(request.form['MUAL'])
    RUAM = int(request.form['RUAM'])
    NYERI_SENDI = int(request.form['NYERI_SENDI'])
    
    input_data_array = np.array([SUHU_TUBUH,RIWAYAT_KESEHATAN,MENGGIGIL,DEMAM,TROMBOSIT,MUAL,RUAM,NYERI_SENDI])
    input_data_reshaped =  input_data_array.reshape(1,-1)
    std_data = preprocessing.scale(input_data_reshaped)
    std_data_reshaped =  std_data.reshape(1,-1)
    
    result = model.predict(input_data_reshaped)
    print(result)
    
    return render_template('index.html', result=result)
  
if __name__== '__main__':
    app.run(debug=True)
