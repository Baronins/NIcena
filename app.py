from flask import Flask, render_template
import pandas as pd
import json

app = Flask(__name__, static_url_path='/static')

# Read latitude, longitude, names, and price data from the dataset
data = pd.read_excel('Caka_longLat.xlsx')

@app.route('/')
def index():
    title = "NICena"
    subtitle = "Nekustamo īpašumu darījumu dati"
    # Convert latitude, longitude, names, and price data to lists
    latitude_json = data['Latitude'].tolist()
    longitude_json = data['Longitude'].tolist()
    names_json = data['Adreses pieraksts'].tolist()
    price_json = data['Darījuma summa, EUR'].tolist()
    return render_template('index.html', title=title, subtitle=subtitle, latitude=latitude_json, longitude=longitude_json, names=names_json, price=price_json)

@app.route('/run_model', methods=['POST'])
def run_model():
    # Handle model execution here
    # Retrieve input data from the request
    # Run your machine learning model
    # Return the results
    return 'Model has been executed'

if __name__ == '__main__':
    app.run(debug=True)
