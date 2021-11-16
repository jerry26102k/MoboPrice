import numpy as np
import pickle
from flask import Flask, render_template, request

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))


@app.route('/')
def main():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def home():
    f1 = float(request.form['a'])
    f2 = float(request.form['b'])
    f3 = float(request.form['d'])
    f4 = float(request.form['e'])
    f5 = float(request.form['f'])
    f6 = float(request.form['g'])
    final_features = np.array([[f1, f2, f3, f4, f5, f6]])
    prediction = model.predict(final_features)
    output = round(prediction[0], 2)

    return render_template('index.html', prediction_text='Price of the mobile is: ' + str(output))


if __name__ == "__main__":
    app.run(debug=True)
