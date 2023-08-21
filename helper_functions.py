# NAME @: Sahil Ghawali
# TOPIC @: Risk Analytics Model Deployment
# DATE @: 08/10/2020 


# IMPORT THE DEPENDENCIES

import pickle
import joblib
from wtforms.validators import DataRequired
from flask_wtf import FlaskForm
import numpy as np
from wtforms import (StringField, BooleanField, DateTimeField,
                     RadioField, SelectField, TextField,
                     TextAreaField, SubmitField)

# REMEMBER TO LOAD THE MODEL AND THE SCALER!

# 1. LOAD THE SVC MODEL
pkl_filename = "Models/XGBmodel.pkl"
with open(pkl_filename, 'rb') as file2:
    XGBmodel = pickle.load(file2)

# 2. LOAD THE SCALER OBJECT
svc_scaler = joblib.load("Models/svm_scaler.pkl")


# 3. CREATE A PREICTION FUNCTION
def return_prediction(model, scaler, sample_json):
    # For larger data features, you should probably write a for loop
    # That builds out this array for you

    gen = sample_json['Partner']
    ob = sample_json['OnlineBackup']
    dep = sample_json['Dependents']
    os = sample_json['OnlineSecurity']
    tenure = sample_json['tenure']
    mc = sample_json['MonthlyCharges']
    tc = sample_json['TotalCharges']
    dp = sample_json['DeviceProtection']
    ts = sample_json['TechSupport']
    pb = sample_json['PaperlessBilling']
    ct = sample_json['Contract']
    pm = sample_json['PaymentMethod']
    sc = sample_json['SeniorCitizen']

    person = [[gen, ob, dep, os, tenure, mc, tc, dp, ts, pb, ct, pm, sc]]

    person = scaler.transform(person)

    classes = np.array(['Not-Churn', 'Churn'])

    class_ind = model.predict(person)

    return classes[class_ind[0]]



# 4 . CREATE A FLASK FORM

class InfoForm(FlaskForm):
    '''
    This general class to accept form data
    '''

    Name = StringField('Enter Your Full Name', validators=[DataRequired()])

    Partner = RadioField('Please choose your marital status:', choices=[('1', 'Yes'), ('0', 'No')])

    OnlineBackup = SelectField('OnlineBackup Details :',
                               choices=[('2', 'Yes'), ('1', 'No internet service'), ('0', 'No')])

    Dependents = RadioField('Please choose dependent or not :', choices=[('1', 'Yes'), ('0', 'No')])

    OnlineSecurity = SelectField('OnlineSecurity Details :',
                                 choices=[('2', 'Yes'), ('1', 'No internet service'), ('0', 'No')])

    tenure = StringField('Tenure In Months ', validators=[DataRequired()])

    MonthlyCharges = StringField('MonthlyCharges Amount ', validators=[DataRequired()])

    TotalCharges = StringField('TotalCharges Amount ', validators=[DataRequired()])

    DeviceProtection = SelectField('DeviceProtection Details :',
                                   choices=[('2', 'Yes'), ('1', 'No internet service'), ('0', 'No')])

    TechSupport = SelectField('TechSupport Details :',
                              choices=[('2', 'Yes'), ('1', 'No internet service'), ('0', 'No')])

    PaperlessBilling = RadioField('PaperlessBilling :', choices=[('1', 'Yes'), ('0', 'No')])

    Contract = SelectField('Contract Details :',
                           choices=[('2', 'Two year'), ('1', 'One year'), ('0', 'Month-to-month')])

    PaymentMethod = SelectField('PaymentMethod Details :',
                                choices=[('3', 'Mailed check'),('2', 'Electronic check'),
                                         ('1', 'Credit card (automatic)'),
                                         ('0', 'Bank transfer (automatic)')])

    SeniorCitizen = RadioField('SeniorCitizen :', choices=[('1', 'Yes'), ('0', 'No')])

    feedback = TextAreaField()

    submit = SubmitField('Submit')