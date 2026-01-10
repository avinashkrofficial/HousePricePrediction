import pandas as pd
from flask import Flask, render_template, request, jsonify
import pickle
import numpy as np
import os
import traceback

app = Flask(__name__)

# ------------------------
# Load Bengaluru dataset + model
# ------------------------
blr_data = pd.read_csv('datasets/Cleaned_Data.csv')
blr_model = pickle.load(open('model/RidgeModel.pkl','rb'))

# ------------------------
# Load Delhi dataset + model
# ------------------------
delhi_data = pd.read_csv('datasets/newdataset.csv')
delhi_model = pickle.load(open('model/RidgeModel_d.pkl','rb'))

# ------------------------
# Home route
# ------------------------
@app.route('/')
def index():
    cities = []
    locations = {}

    if blr_data is not None:
        cities.append("Bengaluru")
        locations["Bengaluru"] = sorted(blr_data['location'].unique().tolist())

    if delhi_data is not None:
        cities.append("Delhi")
        locations["Delhi"] = sorted(delhi_data['location'].unique().tolist())

    return render_template('index.html', cities=cities, locations=locations)


@app.route('/predict', methods=['POST'])
def predict():
    try:
        city = request.form.get('city')
        location = request.form.get('location')
        bhk = request.form.get('bhk')
        bath = request.form.get('bath')
        sqft = request.form.get('sqft')

        if not all([city, location, bhk, bath, sqft]):
            return "Missing required fields", 400

        bhk = int(bhk)
        bath = int(bath)
        sqft = float(sqft)

        if city == "Bengaluru":
            model = blr_model
        elif city == "Delhi":
            model = delhi_model
        else:
            return "Invalid city", 400

        if model is None:
            return f"{city} model not available", 500

        input_data = pd.DataFrame(
            [[location, sqft, bath, bhk]],
            columns=['location', 'total_sqft', 'bath', 'bhk']
        )

        predicted_price = model.predict(input_data)[0] * 1e5
        price_in_lakhs = predicted_price / 1e5

        if predicted_price >= 1e7:
            price_str = f"{predicted_price / 1e7:.2f} Cr"
        else:
            price_str = f"{price_in_lakhs:.2f} Lakh"

        return price_str

    except Exception as e:
        print("Error in prediction:", e)
        traceback.print_exc()
        return "Error predicting price", 500


@app.route('/predict-rent', methods=['POST'])
def predict_rent():
    try:
        city = request.form.get('city')
        location = request.form.get('location')
        bhk = request.form.get('bhk')
        bath = request.form.get('bath')
        sqft = request.form.get('sqft')

        if not all([city, location, bhk, bath, sqft]):
            return jsonify({"error": "Missing required fields"}), 400

        bhk = int(bhk)
        bath = int(bath)
        sqft = float(sqft)

        if city == "Bengaluru":
            model = blr_model
        elif city == "Delhi":
            model = delhi_model
        else:
            return jsonify({"error": "Invalid city"}), 400

        if model is None:
            return jsonify({"error": f"{city} model not available"}), 500

        input_data = pd.DataFrame(
            [[location, sqft, bath, bhk]],
            columns=['location', 'total_sqft', 'bath', 'bhk']
        )

        predicted_price = model.predict(input_data)[0] * 1e5
        price_in_lakhs = predicted_price / 1e5
        monthly_rent = calculate_rent_from_price(price_in_lakhs, bhk, city)

        if predicted_price >= 1e7:
            price_str = f"{predicted_price / 1e7:.2f} Cr"
        else:
            price_str = f"{price_in_lakhs:.2f} Lakh"

        rental_yield = (monthly_rent * 12 / predicted_price) * 100

        return jsonify({
            "price": price_str,
            "monthly_rent": f"₹{monthly_rent:,.0f}",
            "annual_rent": f"₹{monthly_rent * 12:,.0f}",
            "rental_yield": f"{rental_yield:.2f}%"
        })

    except Exception as e:
        print("Error in rent prediction:", e)
        traceback.print_exc()
        return jsonify({"error": "Error predicting rent"}), 500


@app.route('/api/emi', methods=['POST'])
def emi_calculator_api():
    try:
        data = request.json
        price = float(data.get('price', 0))
        down_payment = float(data.get('downPayment', 20))
        interest_rate = float(data.get('interestRate', 8.5))
        tenure = int(data.get('tenure', 20))

        if price <= 0:
            return jsonify({"error": "Invalid price"}), 400

        loan_amount = price * (1 - down_payment / 100) * 100000
        monthly_rate = interest_rate / (12 * 100)
        months = tenure * 12

        if monthly_rate == 0:
            emi = loan_amount / months
        else:
            emi = loan_amount * monthly_rate * ((1 + monthly_rate) ** months) / (((1 + monthly_rate) ** months) - 1)

        total_payment = emi * months
        total_interest = total_payment - loan_amount

        return jsonify({
            "emi": f"₹{emi:,.0f}",
            "total_payment": f"₹{total_payment:,.0f}",
            "total_interest": f"₹{total_interest:,.0f}"
        })

    except Exception as e:
        print("Error in EMI:", e)
        traceback.print_exc()
        return jsonify({"error": "Error calculating EMI"}), 500


@app.route('/api/calculate-rent', methods=['POST'])
def calculate_rent_manual():
    try:
        data = request.json
        price_lakhs = float(data.get('price', 0))
        bhk = int(data.get('bhk', 3))
        city = data.get('city', 'Bengaluru')

        if price_lakhs <= 0:
            return jsonify({"error": "Invalid price"}), 400

        monthly_rent = calculate_rent_from_price(price_lakhs, bhk, city)
        annual_rent = monthly_rent * 12
        price_rupees = price_lakhs * 100000
        rental_yield = (annual_rent / price_rupees) * 100

        return jsonify({
            "monthly_rent": f"₹{monthly_rent:,.0f}",
            "annual_rent": f"₹{annual_rent:,.0f}",
            "rental_yield": f"{rental_yield:.2f}%"
        })

    except Exception as e:
        print("Error in rent calculation:", e)
        traceback.print_exc()
        return jsonify({"error": "Error calculating rent"}), 500


@app.route('/health')
def health_check():
    return jsonify({
        "status": "healthy",
        "bengaluru_model": "available" if blr_model else "unavailable",
        "delhi_model": "available" if delhi_model else "unavailable"
    })


if __name__ == '__main__':
    print("\n" + "=" * 50)
    print("🏠 Real Estate Price + Rent + EMI Calculator")
    print("=" * 50)
    print(f"Server running at: http://127.0.0.1:5000")
    print("=" * 50 + "\n")
    app.run(debug=True, port=5000)

