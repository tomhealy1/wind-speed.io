import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__)
model = pickle.load(open('lin_model1.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict',methods=['POST'])
def predict():
    int_feature = [float(x) for x in request.form.values()]
    final_feature = [np.array(int_feature)]
    prediction = model.predict(final_feature)

    output = round(prediction[0], 2)

    return render_template('home.html', prediction_text="The predicted power output is {}".format(output))


if __name__ == "__main__":
    app.run(debug=True)