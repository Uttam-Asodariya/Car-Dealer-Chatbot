from flask import Flask

app = Flask(__name__) # define application name

@app.route("/", method=["GET"])
def hello_world():
    return "hello_world"

if __name__ == "__main__":
    app.run(port=)