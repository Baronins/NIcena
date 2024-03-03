from flask import Flask, render_template, request

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/run_model', methods=['POST'])
def run_model():
    # Handle model execution here
    # Retrieve input data from the request
    # Run your machine learning model
    # Return the results
    return 'Model has been executed'

if __name__ == '__main__':
    app.run(debug=True)