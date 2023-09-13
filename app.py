import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from prediction import get_prediction, ordinal_encoder

model = joblib.load(r'Model/RandomForestClassifier_model')



st.set_page_config(page_title="Accident Severity Prediction App", page_icon="🚧", layout="wide")


#creating option list for dropdown menu
options_day = ['Sunday', "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"]
options_age = ['18-30', '31-50', 'Over 51', 'Unknown', 'Under 18']

options_acc_area = ['Other', 'Office areas', 'Residential areas', ' Church areas',
       ' Industrial areas', 'School areas', '  Recreational areas',
       ' Outside rural areas', ' Hospital areas', '  Market areas',
       'Rural village areas', 'Unknown', 'Rural village areasOffice areas',
       'Recreational areas']
       
options_cause = ['No distancing', 'Changing lane to the right',
       'Changing lane to the left', 'Driving carelessly',
       'No priority to vehicle', 'Moving Backward',
       'No priority to pedestrian', 'Other', 'Overtaking',
       'Driving under the influence of drugs', 'Driving to the left',
       'Getting off the vehicle improperly', 'Driving at high speed',
       'Overturning', 'Turnover', 'Overspeed', 'Overloading', 'Drunk driving',
       'Unknown', 'Improper parking']
options_vehicle_type = ['Automobile', 'Lorry (41-100Q)', 'Other', 'Pick up upto 10Q',
       'Public (12 seats)', 'Stationwagen', 'Lorry (11-40Q)',
       'Public (13-45 seats)', 'Public (> 45 seats)', 'Long lorry', 'Taxi',
       'Motorcycle', 'Special vehicle', 'Ridden horse', 'Turbo', 'Bajaj', 'Bicycle']
options_driver_exp = ['5-10yr', '2-5yr', 'Above 10yr', '1-2yr', 'Below 1yr', 'No Licence', 'unknown']
options_lanes = ['Two-way (divided with broken lines road marking)', 'Undivided Two way',
       'other', 'Double carriageway (median)', 'One way',
       'Two-way (divided with solid lines road marking)', 'Unknown']

features = ['hour','day_of_week','casualties','accident_cause','vehicles_involved','vehicle_type','driver_age','accident_area','driving_experience','lanes']


st.markdown(""" <style> .font {font-size:40px ; font-family: 'Cooper Black'; color: #FF9633;} </style> """, unsafe_allow_html=True)
st.markdown('<p class="font">Road Accident Severity Prediction App 🚧</p>', unsafe_allow_html=True)


def main():
    with st.form('prediction_form'):

        st.subheader("Enter the input for following features:")

        col1, col2 = st.columns(2)
        with col1:
            hour = st.slider("Pickup Hour: ", 0, 23, value=0, format="%d")
        with col2:
            casualties = st.slider("Hour of Accident: ", 1, 8, value=0, format="%d")
            
       
        col1, col2 = st.columns(2)
        with col1:
            accident_cause = st.selectbox("Select Accident Cause: ", options=options_cause)
        with col2:
            day_of_week = st.selectbox("Select Day of the Week: ", options=options_day)

        col1, col2 = st.columns(2)
        with col1:
            vehicle_type = st.selectbox("Select Vehicle Type: ", options=options_vehicle_type)
        with col2:
            driver_age = st.selectbox("Select Driver Age: ", options=options_age)

        col1, col2 = st.columns(2)
        with col1:
            vehicles_involved = st.slider("Vehicles Involved: ", 1, 7, value=0, format="%d")
        with col2:
            accident_area = st.selectbox("Select Accident Area: ", options=options_acc_area)

        col1, col2 = st.columns(2)
        with col1:
            driving_experience = st.selectbox("Select Driving Experience: ", options=options_driver_exp)
        with col2:
            lanes = st.selectbox("Select Lanes: ", options=options_lanes)
        
        
        
        submit = st.form_submit_button("Predict")


    if submit:
        

        day_of_week = ordinal_encoder(day_of_week, options_day)
        accident_cause = ordinal_encoder(accident_cause, options_cause)
        vehicle_type = ordinal_encoder(vehicle_type, options_vehicle_type)
        driver_age =  ordinal_encoder(driver_age, options_age)
        accident_area =  ordinal_encoder(accident_area, options_acc_area)
        driving_experience = ordinal_encoder(driving_experience, options_driver_exp) 
        lanes = ordinal_encoder(lanes, options_lanes)


        data = np.array([hour,day_of_week,casualties,accident_cause,vehicles_involved, 
                            vehicle_type,driver_age,accident_area,driving_experience,lanes]).reshape(1,-1)

        pred = get_prediction(data=data, model=model)

        if pred[0]=="Slight Injury":
            new_title = '<p style="font-family:sans-serif; font-size: 35px;">The predicted severity is: Slight Injury</p>'
            st.markdown(new_title, unsafe_allow_html=True)
        elif pred[0]=="Serious Injury":
            new_title = '<p style="font-family:sans-serif;  font-size: 35px;">The predicted severity is: Serious Injury</p>'
            st.markdown(new_title, unsafe_allow_html=True)
        elif pred[0]=="Fatal Injury":
            new_title = '<p style="font-family:sans-serif;  font-size: 35px;">The predicted severity is: Fatal Injury</p>'
            st.markdown(new_title, unsafe_allow_html=True)
             

if __name__ == '__main__':
    main()