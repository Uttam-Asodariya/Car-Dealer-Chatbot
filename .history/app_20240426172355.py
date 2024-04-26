from flask import Flask, render_template, request

app = Flask(__name__) # define application name

@app.route("/", methods=["GET"])  # getting started
def hello_world():
    return render_template("index.html")

@app.route("/home/", methods = ["POST"]) # connenction via API
def communicate():
    input= request.form.get()"Submit"
    # user_experience = "C:\Users\uttam\OneDrive\Desktop\chat_bot\stored_experience" + input
    # input.save(user_experience)
    return f"You entered: {input}"


if __name__ == "__main__":
    app.run(port=5500, debug = True, use_reloader=False )