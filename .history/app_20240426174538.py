from flask import Flask, render_template, request
from car_dealer_chatboat import *


app = Flask(__name__) # define application name

@app.route("/", methods=["GET"])  # getting started
def hello_world():
    return render_template("index.html")

@app.route("/", methods = ["POST"]) # connenction via API
def communicate():
    input= request.form["You"]
    # user_experience = "C:\Users\uttam\OneDrive\Desktop\chat_bot\stored_experience" + input
    # input.save(user_experience)
    output = chatboat(input)
    return render_template("index.html", prediction = classification)


if __name__ == "__main__":
    app.run(port=5500, debug = True)