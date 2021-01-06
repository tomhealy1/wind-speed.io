import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__)
model = pickle.load(open('lin_model2.pkl', 'rb'))
model1 = pickle.load(open('rf_model.pkl', 'rb'))

#flask home route
@app.route('/')
def home():
    return render_template('home.html')
#flask route for predictions using post method
@app.route('/predict',methods=['POST'])
def predict():
    #takes a float from our input form and pass it to numpy array pass it to our model.
    int_feature = [float(x) for x in request.form.values()]
    final_feature = [np.array(int_feature)]
    prediction = model.predict(final_feature)
    # round our predict to two places
    output = round(prediction[0], 2)
    # Renders our home.html and our prediction text
    return render_template('home.html', prediction_text="The predicted power output using Linear Regression is {} KWh".format(output))

@app.route('/predictrf',methods=['POST'])
def predictrf():
    int_feature1 = [float(x) for x in request.form.values()]
    final_feature1 = [np.array(int_feature1)]
    prediction1 = model1.predict(final_feature1)

    output1 = round(prediction1[0], 2)

    return render_template('home.html', prediction_textrf="The predicted power output using Random Forest is {} KWh".format(output1))


if __name__ == "__main__":
    app.run(debug=True)