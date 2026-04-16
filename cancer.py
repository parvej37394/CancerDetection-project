import streamlit as st
import pickle
import numpy as np


# Load a mpodel 
model = pickle.load(open("model.pkl","rb"))
scaler = pickle.load(open("scaler.pkl",'rb'))

st.title("Cancer Detection App")
#st.write("Enter values: ")
# Example input 30 features 



features = [  'radius_mean', 'texture_mean', 'perimeter_mean',
       'area_mean', 'smoothness_mean', 'compactness_mean', 'concavity_mean',
       'concave points_mean', 'symmetry_mean', 'fractal_dimension_mean',
       'radius_se', 'texture_se', 'perimeter_se', 'area_se', 'smoothness_se',
       'compactness_se', 'concavity_se', 'concave points_se', 'symmetry_se',
       'fractal_dimension_se', 'radius_worst', 'texture_worst',
       'perimeter_worst', 'area_worst', 'smoothness_worst',
       'compactness_worst', 'concavity_worst', 'concave points_worst',
       'symmetry_worst', 'fractal_dimension_worst']
      

inputs = []

for feature in features:
    val = st.number_input(feature, value=0.0)
    inputs.append(val)

    
if st.button("Predict"):
    data = np.array(inputs).reshape(1 , -1)
    data = scaler.transform(data)

    prediction = model.predict(data)
    if prediction[0] ==1:
        st.error("Cncer Detected (maligant)")
    else:
        st.success("No Cancer (Bengign) ")
