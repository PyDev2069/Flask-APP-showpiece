from flask import Flask,render_template,request
import pandas as pd 
import pickle 

app = Flask(__name__)

with open("home.pkl","rb") as f:
    model=pickle.load(f)

@app.route('/',methods=['GET','POST'])
def index():
    prediction = None
    if request.method == "POST":
        try:
            Area = int(request.form['area'])
            Bedroom = int(request.form['bedroom'])
            newHome = pd.DataFrame([{
                'Area':Area,
                'Bedroom':Bedroom
            }])
            prediction_value = model.predict(newHome)[0]
            prediction = f"Your Home Price is {prediction_value:.2f} rupees"
        except Exception as e:
            prediction = f"Error : {str(e)}"
    return render_template('index.html',prediction=prediction)
if __name__ == '__main__':
    app.run(debug=True,port=8000)