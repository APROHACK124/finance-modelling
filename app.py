# app.py
from flask import Flask

# Create an instance of the Flask class
app = Flask(__name__)

# Define a route for the homepage
@app.route('/')
def hello():
    return "Hello from Docker! Your app is running in a container."



# Run the app
if __name__ == '__main__':
    # Host='0.0.0.0' makes the app accessible from outside the container
    app.run(host='0.0.0.0', port=5000)



