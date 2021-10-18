from flask import Flask,render_template,request
import pickle
import numpy as np


load_model=pickle.load(open('linear_model.h5','rb'))

def pre(number):
    x_test=np.array(number)
    x_test=x_test.reshape(1,-1)
    res=load_model.predict(x_test)
    return res
app=Flask(__name__)
@app.route("/")
def Home():
    return render_template('index.html')

@app.route("/",methods=["POST"])
def deloy_pre():
    if request.method =="POST":
        num=request.form['number']
        res=float(pre(num))

    return render_template("index.html",predict_number=res)

if __name__ == '__main__':
    app.run(debug=True)
