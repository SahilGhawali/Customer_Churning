from flask import Flask, render_template, session, redirect, url_for, session
from flask_wtf import FlaskForm
import pickle
from sklearn.externals import joblib
import numpy as np
from wtforms import (StringField, BooleanField, DateTimeField,
                     RadioField, SelectField, TextField,
                     TextAreaField, SubmitField)
from wtforms.validators import DataRequired
from xgboost import XGBClassifier

app = Flask(__name__)
# Configure a secret SECRET_KEY
# We will later learn much better ways to do this!!
app.config['SECRET_KEY'] = 'mysecretkey'


# Now create a WTForm Class
# Lots of fields available:
# http://wtforms.readthedocs.io/en/stable/fields.html
class InfoForm(FlaskForm):
    '''
    This general class to accept form data
    '''

    Name = StringField('Enter Your Full Name', validators=[DataRequired()])

    Partner = RadioField('Please choose your marital status:', choices=[('1', 'Yes'), ('0', 'No')])

    Dependents = RadioField('Please choose dependent or not :', choices=[('1', 'Yes'), ('0', 'No')])

    OnlineBackup = SelectField('OnlineBackup Details :',
                               choices=[('2', 'Yes'), ('1', 'No internet service'), ('0', 'No')])

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

    feedback = TextAreaField()

    submit = SubmitField('Submit')


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


# REMEMBER TO LOAD THE MODEL AND THE SCALER!

# LOAD THE SVC MODEL
pkl_filename = "Models/XGBmodel.pkl"
with open(pkl_filename, 'rb') as file2:
    svc_pickle_model = pickle.load(file2)

# LOAD THE SCaleR OBJECT
svc_scaler = joblib.load("Models/svm_scaler.pkl")


@app.route('/', methods=['GET', 'POST'])
def index():
    # Create instance of the form.
    form = InfoForm()
    # If the form is valid on submission 
    if form.validate_on_submit():
        # Grab the data from  form.

        session['Name'] = form.Name.data
        session['Partner'] = form.Partner.data
        session['OnlineBackup'] = form.OnlineBackup.data
        session['Dependents'] = form.Dependents.data
        session['OnlineSecurity'] = form.OnlineSecurity.data
        session['Contract'] = form.Contract.data
        session['tenure'] = form.tenure.data
        session['MonthlyCharges'] = form.MonthlyCharges.data
        session['TotalCharges'] = form.TotalCharges.data
        session['DeviceProtection'] = form.DeviceProtection.data
        session['TechSupport'] = form.TechSupport.data
        session['PaperlessBilling'] = form.PaperlessBilling.data
        session['PaymentMethod'] = form.PaymentMethod.data
        session['SeniorCitizen'] = form.SeniorCitizen.data

        return redirect(url_for("prediction"))

    return render_template('home.html', form=form)


@app.route('/prediction')
def prediction():
    content = {}

    content['Partner'] = float(session['Partner'])
    content['OnlineBackup'] = float(session['OnlineBackup'])
    content['Dependents'] = float(session['Dependents'])
    content['OnlineSecurity'] = float(session['OnlineSecurity'])
    content['Contract'] = float(session['Contract'])
    content['tenure'] = float(session['tenure'])
    content['MonthlyCharges'] = float(session['MonthlyCharges'])
    content['TotalCharges'] = float(session['TotalCharges'])
    content['DeviceProtection'] = float(session['DeviceProtection'])
    content['TechSupport'] = float(session['TechSupport'])
    content['PaperlessBilling'] = float(session['PaperlessBilling'])
    content['PaymentMethod'] = float(session['PaymentMethod'])
    content['SeniorCitizen'] = float(session['SeniorCitizen'])

    results = return_prediction(model=svc_pickle_model,
                                scaler=svc_scaler,
                                sample_json=content)

    return render_template('thankyou.html', results=results)


# if __name__ == '__main__':
#     app.run('0.0.0.0',port = 8080 , debug=True)


if __name__ == '__main__':
    app.run(debug=True)
