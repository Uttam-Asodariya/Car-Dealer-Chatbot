
from flask import Flask, render_template, request
from car_dealer_chatboat import *

#waitress-serve --host 127.0.0.1 app:app for hosting app # exclusively for windows

app = Flask(__name__) # define application name

@app.route("/", methods=["GET"])  # getting started
def hello_world():
    return render_template("index.html")

@app.route("/", methods = ["POST"]) # connenction via API
def communicate():
    input= request.form.get("You")
    # user_experience = "C:\Users\uttam\OneDrive\Desktop\chat_bot\stored_experience" + input
    # input.save(user_experience)
    response = chatboat(input)
    print(f"Chatbot Response: {response}")
    return render_template("index.html", prediction = str(response))

if __name__ == "__main__":
    app.run(debug = True)