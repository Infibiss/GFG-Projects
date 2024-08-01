import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from sklearn.preprocessing import StandardScaler


# Function to get user input
def get_user_input():
    st.sidebar.header('User Input Features')

    pclass = st.sidebar.selectbox('Passenger Class', (1, 2, 3))
    sex = st.sidebar.selectbox('Sex', ('Male', 'Female'))
    age = st.sidebar.slider('Age', 0, 100, 0)
    sibsp = st.sidebar.slider('Siblings/Spouses Aboard', 0, 8, 0)
    parch = st.sidebar.slider('Parents/Children Aboard', 0, 6, 0)
    fare = st.sidebar.slider('Fare', 1, 500, 1)
    embarked = st.sidebar.selectbox('Embarked', ('Southampton', 'Cherbourg', 'Queenstown'))

    user_data = {'Pclass': pclass,
                 'Sex': sex,
                 'Age': age,
                 'SibSp': sibsp,
                 'Parch': parch,
                 'Fare': fare,
                 'Embarked': embarked}

    features = pd.DataFrame(user_data, index=[0])
    return features


# Function to preprocess user input
def preprocess_input(data):
    genders = {"Male": 0, "Female": 1}
    data['Sex'] = data['Sex'].map(genders)

    ports = {"Southampton": 0, "Cherbourg": 1, "Queenstown": 2}
    data['Embarked'] = data['Embarked'].map(ports)
    return data

def predict(user_input):
    user_input = preprocess_input(user_input)

    path = os.path.dirname(__file__).replace('\\', '/') + '/model.joblib'
    model = joblib.load(path)
    prediction = model.predict(user_input)

    st.write('### Prediction:')
    if prediction[0] == 1:
        st.success('You survived!')
    else:
        st.error('You didn`t survive!')


# Main function
def main():
    # Custom CSS to create a blue container
    st.markdown(
        """
        <style>
        .blue-container {
            background-color: #00AAFF; /* Light blue color */
            padding: 10px;
            margin-bottom: 40px;
            border-radius: 5px;
            text-align: center;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    # Content of the blue container
    st.markdown('<div class="blue-container">'
                '<h1>Titanic Survival Prediction</h1>'
                '<h4>Logistic Regression</h4>',
                unsafe_allow_html=True)

    user_input = get_user_input()
    st.write('### User Input:')
    st.write(user_input)

    if st.button('Predict'):
        predict(user_input)


if __name__ == '__main__':
    main()
