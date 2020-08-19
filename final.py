import json
import pickle
import numpy as np
from flask import Flask, request, render_template


app=Flask(__name__)
_locations=None
_data_columns=None
_model=None

@app.route('/')
def home():
    return render_template('banglore.html')
        
@app.route('/predict',methods=['POST'])       
def predict():
    
    global _data_columns
    global _locations
	
    with open("./extra/columns.json",'r') as f:
        _data_columns=json.load(f)['data_columns']
        _locations=_data_columns[4:]
    global _model	
    with open("./extra/rcb_home_prices.pickle",'rb') as f:
        _model=pickle.load(f)
    features=[i for i in request.form.values()]
    print(request.form.values())
    print(features)
    location=features[3]
    try:
        loc_index=_data_column.index(location.lower())
    except:
        loc_index=-1
    x=np.zeros(len(_data_columns))
    x[0]=int(features[0])
    x[1]=int(features[1])
    x[2]=int(features[2])
    x[3]=int(features[3])
    if loc_index >= 0:
        x[loc_index]=1
       
    output=round(_model.predict([x])[0],2)
    return render_template('banglore.html',prediction_text="Predicted Price in Lakh is {}".format(output))





if __name__=="__main__":
    app.run(debug=True, use_reloader=False)
	