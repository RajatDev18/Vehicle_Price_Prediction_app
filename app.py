import streamlit as st
import pickle
import numpy as np
import pandas as pd

# --------------------------------------------------
# Load Model
# --------------------------------------------------
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

st.title("🚗 Vehicle Price Prediction App")
st.write("Enter the vehicle details below to predict its price.")

# --------------------------------------------------
# Input Fields (ONLY FEATURES USED IN TRAINING)
# --------------------------------------------------

make = st.selectbox("Make",['Jeep', 'GMC', 'Dodge', 'RAM', 'Nissan', 'Ford', 'Hyundai',
       'Chevrolet', 'Volkswagen', 'Chrysler', 'Kia', 'Mazda', 'Acura',
       'Subaru', 'Audi', 'BMW', 'Toyota', 'Buick', 'Mercedes-Benz',
       'Honda', 'Lincoln', 'Cadillac', 'INFINITI', 'Lexus', 'Land Rover',
       'Volvo', 'Genesis', 'Jaguar'])

model_name = st.text_input("Model")

cylinders = st.number_input("Cylinders", min_value=1, max_value=16, step=1)

fuel = st.selectbox("Fuel Type", [
    'Gasoline', 'Diesel', 'Hybrid', 'Electric', 'E85 Flex Fuel','PHEV Hybrid Fuel', 'Diesel (B20 capable)'
])

mileage = st.number_input("Mileage (Miles Driven)", min_value=0, step=100)

transmission = st.selectbox("Transmission", ['automatic', 'cvt', 'single-speed (EV)', 'other', 'dual clutch'])
trim = st.text_input("Trim")

body = st.selectbox("Body Style", [
    'SUV', 'Pickup Truck', 'Sedan', 'Passenger Van', 'Cargo Van',
    'Hatchback', 'Convertible', 'Minivan'
])

doors = st.number_input("Doors", min_value=2, max_value=6, step=1)

exterior_color = st.text_input("Exterior Color")
interior_color = st.text_input("Interior Color")

drivetrain = st.selectbox("Drivetrain", [
    'Four-wheel Drive', 'All-wheel Drive', 'Rear-wheel Drive','Front-wheel Drive'
])

fuel_sys = st.selectbox("Fuel System", [
    'direct injection', 'multipoint injection', 'diesel direct injection', 'other'
])

turbo = st.selectbox("Turbocharged", ["Yes", "No"])

# --------------------------------------------------
# Predict Button
# --------------------------------------------------

if st.button("Predict Price"):
    try:
        # Apply log1p transformation to mileage (must match your training)
        mileage_log = np.log1p(mileage)
        if turbo == 'Yes':
            turbo = 1
        elif turbo == 'No':
            turbo = 0

        # Create input data exactly matching model's training columns
        input_data = pd.DataFrame([{
            "make": make,
            "model": model_name,
            "cylinders": cylinders,
            "fuel": fuel,
            "mileage": mileage_log,       # TRANSFORMED FEATURE
            "transmission": transmission,
            "trim": trim,
            "body": body,
            "doors": doors,
            "exterior_color": exterior_color,
            "interior_color": interior_color,
            "drivetrain": drivetrain,
            "fuel_sys": fuel_sys,
            "turbo": turbo
        }])

        # Predict price
        prediction = model.predict(input_data)[0]

        st.success(f"💰 Estimated Vehicle Price: **${prediction:,.2f}**")

    except Exception as e:
        st.error(f"Error: {e}")
