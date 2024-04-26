#
from flask import Flask, render_template, request

app = Flask(__name__) # define application name

@app.route("/", methods=["GET"])  # getting started
def hello_world():
    return render_template("index.html")

@app.route("/", methods = ["post"]) # connenction via API
def communicate():
    pass

if __name__ == "__main__":
    app.run(port=5500, debug = True)