from codecs import getencoder
from unicodedata import category
from flask import Flask, render_template, request
import pickle
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

model = pickle.load(open('model.pkl', 'rb'))
app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

column_names = ['state', 'party', 'gender', 'criminal cases', 'age',
       'category', 'education', 'assets', 'liabilities', 'total electors',
       'balance assets']

party_dict = {'BJP': 16, 'INC': 15, 'Other': 14, 'IND': 13, 'BSP': 12, 'CPI(M)': 11, 'AITC': 10, 'VBA': 9, 'SP': 8, 
              'NTK': 7, 'MNM': 6, 'SHS': 5, 'AAP': 4, 'YSRCP': 3, 'TDP': 2, 'DMK': 1}

state_dict = {'Uttar Pradesh': 35, 'Bihar': 34, 'Tamil Nadu': 33, 'Maharashtra': 32, 'West Bengal': 31,
       'Andhra Pradesh': 30, 'Madhya Pradesh': 29, 'Rajasthan': 28, 'Telangana': 27, 'Odisha': 26,
       'Karnataka': 25, 'Gujarat': 24, 'Jharkhand': 23, 'Kerala': 22, 'Punjab': 21, 'Assam': 20,
       'Haryana': 19, 'Chhattisgarh': 18, 'Jammu & Kashmir': 17, 'NCT OF Delhi': 16,
       'Uttarakhand': 15, 'Arunachal Pradesh': 14, 'Manipur': 13, 'Himachal Pradesh': 12,
       'Tripura': 11, 'Dadra & Nagar Haveli': 10, 'Goa': 9, 'Meghalaya': 8, 'Mizoram': 7,
       'Andaman and Nicobar Islands': 6, 'Puducherry': 5, 'Sikkim': 4, 'Chandigarh': 3,
       'Nagaland': 2, 'Daman & Diu': 1, 'Lakshadweep': 0}

cat_dict = {'GEN': 0.0, 'SC': 0.5, 'ST': 1.0}
gend_dict = {'male': 1, 'female': 0}
edu_dict = {'Illiterate': 0, 'Literate': 1, '5th Pass': 2, '8th Pass': 3, '10th Pass': 4, 
 'Others': 5, '12th Pass': 6, 'Graduate': 7, 'Post Graduate': 8, 'Doctorate': 9}
df = pd.DataFrame(columns = column_names)
list_arr = []

@app.route('/predict', methods = ['POST'])
def predict():
    state = request.values['state']
    list_arr.append(state_dict[state])

    party = request.values['party']
    list_arr.append(party_dict[party])

    gender = request.values['gender']
    list_arr.append(gend_dict[gender])

    cases = request.values['criminal cases']
    list_arr.append(cases)

    age = float(request.values['age'])
    list_arr.append(age)

    category = request.values['category']
    list_arr.append(cat_dict[category])

    education = request.values['education']
    list_arr.append(edu_dict[education])

    assets = float(request.values['assets'])
    list_arr.append(assets)

    liabilities = float(request.values['liabilities'])
    list_arr.append(liabilities)

    electors = float(request.values['total electors'])
    list_arr.append(electors)

    balance_assets = assets - liabilities
    list_arr.append(balance_assets)
    scale = MinMaxScaler()
    scale.fit_transform([list_arr])

    in_values = np.reshape(list_arr,(1, -1))
    #print(in_values)

    answer = model.predict(in_values)
    output = answer.item()
    if output >= 0.7:
        prediction = 'win'
    else:
        prediction = 'lose'
    
    return render_template('result.html', prediction_text = f'The candidate can expect to {prediction}')

if __name__ == '__main__':
    app.run(port=8000)