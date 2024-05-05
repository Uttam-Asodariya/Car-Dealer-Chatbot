import requests
from flask import Flask, render_template, request
from car_dealer_chatboat import *
from flask_apscheduler import APScheduler

#waitress-serve --host 127.0.0.1 app:app for hosting app

app = Flask(__name__) # define application name
scheduler = APScheduler()

def trigger_url():
    url = "https://car-dealer-chatbot-1.onrender.com"
    response = requests.get(url)
    print("URL Triggered Successfully")

@app.route("/", methods=["GET"])  # getting started
def hello_world():
    return render_template("index.html")

@app.route("/", methods = ["POST"]) # connenction via API
def communicate():
    input= request.form.get("You")
    # user_experience = "C:\Users\uttam\OneDrive\Desktop\chat_bot\stored_experience" + input
    # input.save(user_experience)
    Response = chatboat(input)
    print(f"Chatbot Response: {Response}")
    return render_template("index.html", prediction = str(Response))


if __name__ == "__main__":
    scheduler.init_app(app)
    scheduler.start()
    scheduler.add_job(id='trigger_url', func=trigger_url, trigger='interval', minutes=14)

    app.run(host='0.0.0.0', debug = "True")