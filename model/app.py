from flask import Flask, render_template, request, jsonify
import joblib
import numpy as np
import pandas as pd
import os

app = Flask(__name__)

# Load models
weather_model = joblib.load(os.path.join('models', 'weather_model.pkl'))
flight_model = joblib.load(os.path.join('models', 'flight_model.pkl'))
disaster_model = joblib.load(os.path.join('models', 'disaster_model.pkl'))
disaster_vectorizer = joblib.load(os.path.join('models', 'disaster_vectorizer.pkl'))

# Load sample data for dropdowns (for simplicity, hardcoded here)
places = ['New York', 'Los Angeles', 'Chicago', 'Houston', 'Phoenix']
days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']

@app.route('/')
def index():
    return render_template('index.html', places=places)

@app.route('/predict_weather', methods=['POST'])
def predict_weather():
    try:
        # Check if request is JSON (AJAX) or form data
        if request.is_json:
            data = request.get_json()
            temperature = float(data['temperature'])
            humidity = float(data['humidity'])
            pressure = float(data['pressure'])
        else:
            temperature = float(request.form['temperature'])
            humidity = float(request.form['humidity'])
            pressure = float(request.form['pressure'])
        
        # Create input array - adjusted to match your weather model's expected features
        # Based on Weather.py, the model expects: humidity, wind_kph, pressure_mb, cloud, uv_index, visibility_km, air_quality_PM2.5
        # Since you're only collecting temp, humidity, pressure, we'll use defaults for missing features
        input_data = np.array([[
            humidity,           # humidity
            10.0,              # wind_kph (default)
            pressure,          # pressure_mb
            50.0,              # cloud (default)
            5.0,               # uv_index (default)
            10.0,              # visibility_km (default)
            25.0               # air_quality_PM2.5 (default)
        ]])
        
        prediction = weather_model.predict(input_data)[0]
        
        if request.is_json:
            return jsonify({'prediction': round(prediction, 2), 'unit': '°C'})
        else:
            return render_template('index.html', 
                                 weather_prediction=f"{round(prediction, 2)}°C", 
                                 places=places)
    except Exception as e:
        if request.is_json:
            return jsonify({'error': str(e)}), 400
        else:
            return render_template('index.html', error=str(e), places=places)

@app.route('/predict_flight', methods=['POST'])
def predict_flight():
    try:
        # Check if request is JSON (AJAX) or form data
        if request.is_json:
            data = request.get_json()
            airline = int(data['airline'])
            from_airport = int(data['from_airport'])
            to_airport = int(data['to_airport'])
            day = int(data['day'])
            time = int(data['time'])
        else:
            airline = int(request.form['airline'])
            from_airport = int(request.form['from_airport'])
            to_airport = int(request.form['to_airport'])
            day = int(request.form['day'])
            time = int(request.form['time'])
        
        # Based on Flightdelay.py, the model expects 15 features:
        # ['Airline', 'Distance', 'Departure Delay', 'Passenger Load Factor', 'Airline Rating', 
        #  'Airport Rating', 'Market Share', 'OTP Index', 'weather__hourly__windspeedKmph', 
        #  'weather_desc', 'weather__hourly__precipMM', 'weather__hourly__humidity', 
        #  'weather__hourly__visibility', 'weather__hourly__pressure', 'weather__hourly__cloudcover']
        
        # Since you're only collecting basic info, we'll use reasonable defaults
        input_data = np.array([[
            airline,           # Airline
            500.0,            # Distance (default)
            0.0,              # Departure Delay (default)
            0.8,              # Passenger Load Factor (default)
            4.0,              # Airline Rating (default)
            4.0,              # Airport Rating (default)
            0.15,             # Market Share (default)
            0.85,             # OTP Index (default)
            15.0,             # wind speed (default)
            1,                # weather desc (default)
            0.0,              # precipitation (default)
            60.0,             # humidity (default)
            10.0,             # visibility (default)
            1013.0,           # pressure (default)
            30.0              # cloud cover (default)
        ]])
        
        prediction = flight_model.predict(input_data)[0]
        
        # Map prediction to readable format
        delay_status = {
            0: "On Time",
            1: "Delayed", 
            2: "Cancelled"
        }
        result = delay_status.get(prediction, f"Status Code: {prediction}")

        
        if request.is_json:
            return jsonify({'prediction': result})
        else:
            return render_template('index.html', 
                                 flight_prediction=result, 
                                 places=places)
    except Exception as e:
        if request.is_json:
            return jsonify({'error': str(e)}), 400
        else:
            # FIXED: Changed 'tindex.html' to 'index.html'
            return render_template('index.html', error=str(e), places=places)

@app.route('/predict_disaster', methods=['POST'])
def predict_disaster():
    try:
        # Check if request is JSON (AJAX) or form data
        if request.is_json:
            data = request.get_json()
            text = data['disaster_text']
        else:
            text = request.form['disaster_text']
        
        # Transform text using the saved vectorizer
        X_text = disaster_vectorizer.transform([text])
        prediction = disaster_model.predict(X_text)[0]
        
        if request.is_json:
            return jsonify({'prediction': prediction})
        else:
            return render_template('index.html', 
                                 disaster_prediction=prediction, 
                                 places=places)
    except Exception as e:
        if request.is_json:
            return jsonify({'error': str(e)}), 400
        else:
            # FIXED: Changed 'tindex.html' to 'index.html'
            return render_template('index.html', error=str(e), places=places)

if __name__ == '__main__':
    app.run(debug=True)
