from flask import Flask,render_template
import pickle as p
app=Flask(__name__)
model=p.load(open("Model1.pkl",'rb'))

##Home page
@app.route("/") 
def index():
    return render_template("index.html")     ## HTML Interface
@app.route("/Submit",methods=["POST"]) ##pass values to server
def predict():
    A=[]
    from flask import request  
    for i in request.form.values():  ## user values converted into list
        A.append(int(i))                     ## converted user values                                                    into integer ans pass into                                                             A
    Loan_Status = model.predict([A])
    return render_template("index.html",pred=Loan_Status)
 
if __name__ == "__main__":
    app.run(debug=True)
    
    
    