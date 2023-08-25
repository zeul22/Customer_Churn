import pickle
import numpy as np
import streamlit as st
import pandas as pd

clf=pickle.load(open('clf.pkl',"rb"))
scaler = pickle.load(open('scaler.pkl',"rb"))
encoder = pickle.load(open('ohe.pkl',"rb"))


def reshape(input_data):
    data_inp=np.asarray(input_data)
    data_rshp=data_inp.reshape(1,-1)
    return data_rshp

def predict_value(input_data):
    data_rshp=reshape(input_data)
    prediction=clf.predict(data_rshp)
    print(prediction)
    if (prediction[0]==0):
        return 0
    else:
        return 1
    
def main():
    st.title("Churn Prediction Web app!")
    st.write('Created by @Rahul Anand')
    st.write("Enter the values for prediction!")
    
    location=st.selectbox('Location',['USA','Florida','Mexico'])
    subscription=st.number_input('Subscriptions')
    bill=st.number_input('Monthly_Bill')
    usage=st.number_input('Usage in GB')
    
    if st.button("Predict"):
        input_data=(subscription,	bill,	usage,	1.0	,0.0,	0.0	,0.0	,0.0	,1.0,	0.0)
        user_input=reshape(input_data)
        result=predict_value(input_data)
        st.title("Predicted")
        st.success(f"The Output is {result}")
        
        
if __name__=='__main__':
    main()


