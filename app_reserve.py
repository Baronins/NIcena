from flask import Flask, render_template

app = Flask(__name__, static_url_path='/static')

@app.route('/')
def index():
    title = "NICena"
    subtitle = "Nekustamo īpašumu darījumu dati"
    return render_template('index.html', title=title, subtitle=subtitle)

@app.route('/run_model', methods=['POST'])
def run_model():
    # Handle model execution here
    # Retrieve input data from the request
    # Run your machine learning model
    # Return the results
    return 'Model has been executed'

if __name__ == '__main__':
    app.run(debug=True)
