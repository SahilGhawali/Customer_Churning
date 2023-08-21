# NAME @: Devdatta Supnekar
# TOPIC @: Risk Analytics Model Deployment
# DATE @: 08/10/2020 


# IMPORT THE DEPENDENCIES 

from flask import (Flask,
                   render_template,
                   session, redirect,
                   url_for, request,
                   jsonify)

from helper_functions import (XGBmodel,
                              svc_scaler,
                              return_prediction,
                              InfoForm)


app = Flask(__name__)
# Configure a secret SECRET_KEY
app.config['SECRET_KEY'] = 'mysecretkey'


# 1. View point to show the form and collect the data from user
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


# 2. View point to show the result 
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

    # PRINT THE DATA PRESENT IN THE REQUEST
    print("[INFO] WEB Request  - ", content)

    # Actual prediction done by this function
    results = return_prediction(model=XGBmodel, scaler=svc_scaler, sample_json=content)

    # PRINT THE RESULT
    print("[INFO] WEB Responce - ", results)

    return render_template('thankyou.html', results=results)


# 3. View point to handle the restfull api for prediciton
@app.route('/api/prediction', methods=['POST'])
def predict_flower():
    # RECIEVE THE REQUEST
    content = request.json

    # PRINT THE DATA PRESENT IN THE REQUEST 
    print("[INFO] API Request - ", content)

    # PREDICT THE CLASS USING HELPER FUNCTION 
    results = return_prediction(model=XGBmodel, scaler=svc_scaler, sample_json=content)

    # PRINT THE RESULT 
    print("[INFO] API Responce - ", results)

    # SEND THE RESULT AS JSON OBJECT 
    return jsonify(results)


# 4. View Point To handle the 404 Not found Error
@app.errorhandler(404)
def page_not_found(e):
    return render_template('notfound.html'), 404


# if __name__ == '__main__':
#     app.run('0.0.0.0', 8080,debug=False)


if __name__ == '__main__':
    app.run()
