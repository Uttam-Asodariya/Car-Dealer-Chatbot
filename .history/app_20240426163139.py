#

from flask import Flask

app = Flask(__name__) # define application name

@app.route("/", methods=["GET"])
def hello_world():
    return "hello_world2"

if __name__ == "__main__":
    app.run(port=5500, debug = True)