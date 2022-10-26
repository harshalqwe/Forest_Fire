from flask import Flask,request, url_for, redirect, render_template
import numpy as np
import pickle


app = Flask(__name__,template_folder='template')

model = pickle.load(open('model.pkl','rb'))

@app.route('/')
def hello_world():
    return render_template("Forest_Fire.html")


@app.route('/predict',methods=['POST','GET'])
def predict():
    int_features=[int(x) for x in request.form.values()]
    final=[np.array(int_features)]
    print(int_features)
    print(final)
    prediction=model.predict_proba(final)
    output='{0:.{1}f}'.format(prediction[0][1], 2)

    if output>str(0.5):
        return render_template('Forest_Fire.html',pred='Your Forest is in Danger.Prediction of fire occuring is {}'.format(float(output)*100),bhai='Danger')
    else:
        return render_template('Forest_Fire.html',pred='Your Forest is Safe.Prediction of fire occuring is {}'.format(float(output)*100),bhai='Safe')
if __name__ == '__main__':
    app.run(debug=True)
