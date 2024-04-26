#

from flask import Flask, render_template

app = Flask(__name__) # define application name

@app.route("/", methods=["GET"])
def hello_world():
    return render_template("")

if __name__ == "__main__":
    app.run(port=5500, debug = True)