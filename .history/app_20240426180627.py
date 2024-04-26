from flask import Flask, render_template, request
from car_dealer_chatboat import *


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
    <!DOCTYPE html>

<html>
    <head>
        <title> Tutorial </title>

    </head>

    <body>

        <h1 class = "text-center">Car Dealer Chatboat</h1>
        <form action="/" method="post">
            <label for="You">You:</label>
            <input type="text" id="You" name="You">
            <input type="submit" value="Connect to BoAT">
        </form>

        {% if prediction %}
            <p class="text-center">BoAT: {{prediction}}</p>
            <p> Note: If this is not you are looking for, please rephrase your question. </p>
        {% endif %}
    </body>

</html>


    return render_template("index.html", prediction = str(response))


if __name__ == "__main__":
    app.run(port=5500, debug = True)