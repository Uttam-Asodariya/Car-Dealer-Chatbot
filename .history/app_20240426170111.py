from flask import Flask, render_template, request

app = Flask(__name__) # define application name

@app.route("/", methods=["GET"])  # getting started
def hello_world():
    return render_template("index.html")

@app.route("/", methods = ["GET"]) # connenction via API
def communicate():
    input= request.type["You"]
    user_experience = "C:\Users\uttam\OneDrive\Desktop\chat_bot\stored_experience" + input
    input.save()


if __name__ == "__main__":
    app.run(port=5500, debug = True)