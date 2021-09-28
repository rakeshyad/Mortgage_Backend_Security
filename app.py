import pandas as pd
import numpy as np
import pickle
from flask import Flask, render_template, request



app = Flask(__name__)

model_knn  = pickle.load(open("model_pickle_knn","rb"))

# Home Page is good 

@app.route('/',methods=['GET'])
def home():
    return render_template("payment.html")

@app.route('/predict',methods=['GET','POST'])
def predict():
    
    
        CreditScore = request.form.get("CreditScore")
        IsFirstTime = request.form.get("IsFirstTime")
        DTI =request.form.get("DTI")
        LTV =request.form.get("LTV")
        MIP =request.form.get("MIP")
        OCLTV =request.form.get("OCLTV")
        OrigInterestRate =request.form.get("OrigInterestRate")
        OriginalLoanTerm =request.form.get("OriginalLoanTerm")
        MonthsRepayment =request.form.get("MonthsRepayment")
        PPM =request.form.get("PPM")
        NumBorrowers =request.form.get("NumBorrowers")

        

        data=pd.DataFrame({"CreditScore":CreditScore,"IsFirstTime":IsFirstTime,"MIP":MIP, 
        "OCLTV":OCLTV,"DTI":DTI,"LTV":LTV,"OrigInterestRate":OrigInterestRate, "PPM":PPM, "OriginalLoanTerm":OriginalLoanTerm,"NumBorrowers":NumBorrowers, "MonthsRepayment":MonthsRepayment},index=[0])
        data=np.array(data)
        prediction=model_knn.predict_proba(data)
        
        print(prediction[0][0])
        if prediction[0][0] <0.5:
            return render_template('payment.html', prediction_text='This user is Risky to give loan')
        else:
            return render_template('payment.html',prediction_text='This user is good for Loan')




if __name__=="__main__":
    app.run(debug=True)