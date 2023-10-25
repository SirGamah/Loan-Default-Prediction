import streamlit as st
from streamlit_option_menu import option_menu
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from plotly import graph_objs as go

from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

import warnings

warnings.filterwarnings("ignore")

#-----------Web page setting-------------------#
page_title = "Bank Loan Default Prediction"
page_icon = ":robot"
approve_icon = ":check_mark_button:"
not_approve_icon = ":prohibited:"
layout = "centered"

#--------------------Page configuration------------------#
st.set_page_config(page_title = page_title, page_icon = page_icon, layout = layout)



selected = option_menu(
    menu_title = "LoanDefaultPrediction",
    options = ['Home', 'Explore', 'Prediction', 'Contact'],
    icons = ["house-fill", "book-half", "robot", "envelope-fill"],
    default_index = 0,
    orientation = "horizontal"
)

# Load and clean data
@st.cache_data
def load_data():
    data = pd.read_csv('application_data.csv')
    data = data.dropna()
    data = data[[
            'TARGET',
            'NAME_CONTRACT_TYPE',
            'CODE_GENDER',
            'FLAG_OWN_REALTY',
            'AMT_INCOME_TOTAL',
            'AMT_CREDIT',
            'NAME_INCOME_TYPE',
            'NAME_EDUCATION_TYPE',
            'NAME_HOUSING_TYPE',
            'DAYS_BIRTH',
            'OCCUPATION_TYPE',
            'CNT_FAM_MEMBERS',
            'REGION_RATING_CLIENT',
            'ORGANIZATION_TYPE',
             ]]
        
    data['TARGET'] = data['TARGET'].apply(lambda x: 'Yes' if x == 1 else 'No')
    data = data[data['CODE_GENDER'] != 'XNA']
    data['CODE_GENDER'] = data['CODE_GENDER'].apply(lambda x: 'Male' if x == 'M' else 'Female')
        
    data['FLAG_OWN_REALTY'] = data['FLAG_OWN_REALTY'].apply(lambda x: 'Yes' if x == 'Y' else 'No')
    data['NAME_EDUCATION_TYPE'].replace(['Higher education', 'Incomplete higher'], 'Senior High', inplace=True)
    data['NAME_EDUCATION_TYPE'].replace('Secondary / secondary special', 'Junior High', inplace=True)
    data['NAME_EDUCATION_TYPE'].replace('Lower secondary', 'Primary or None', inplace=True)
    data['NAME_HOUSING_TYPE'].replace('House / apartment', 'Owns House/Apartment', inplace=True)
    data['DAYS_BIRTH'] = data['DAYS_BIRTH'] / 365.2
    data['DAYS_BIRTH'] = data['DAYS_BIRTH'].abs().astype(int)
    data['OCCUPATION_TYPE'].replace(['Laborers', 'Low-skill Laborers'], 'Laborers', inplace=True)
    data['OCCUPATION_TYPE'].replace(['High skill tech staff', 'IT staff'], 'IT staff', inplace=True)
    data['OCCUPATION_TYPE'].replace('Waiters/barmen staff', 'Waiters/barmen', inplace=True)
    data['REGION_RATING_CLIENT'] = data['REGION_RATING_CLIENT'].apply(lambda x: 'City' if x == 1 else ('Town' if x == 2 else 'Rural'))
    trade = ['Trade: type 4', 'Trade: type 1', 'Trade: type 3', 'Trade: type 7', 'Trade: type 2', 'Trade: type 6', 'Trade: type 5']
    business = ['Business Entity Type 3', 'Business Entity Type 2', 'Business Entity Type 1']
    industry = [
            'Industry: type 6',
            'Industry: type 8',
            'Industry: type 10',
            'Industry: type 12',
            'Industry: type 2',
            'Industry: type 4',
            'Industry: type 1',
            'Industry: type 5',
            'Industry: type 11',
            'Industry: type 3',
            'Industry: type 7',
            'Industry: type 9',
            'Industry: type 13'
            ]
    transport = [
            'Transport: type 1',
            'Transport: type 3',
            'Transport: type 2',
            'Transport: type 4'
        ]

    education = ['School', 'Kindergarten', 'University']
    data['ORGANIZATION_TYPE'].replace(trade, 'Trade', inplace=True)
    data['ORGANIZATION_TYPE'].replace(business, 'Business', inplace=True)
    data['ORGANIZATION_TYPE'].replace(industry, 'Industry', inplace=True)
    data['ORGANIZATION_TYPE'].replace(transport, 'Transport', inplace=True)
    data['ORGANIZATION_TYPE'].replace(education, 'Education', inplace=True)
                
    return data

data = load_data()


if selected == "Home":
    st.title(f"Welcome!")
    st.write("""LoanDefaultPrediction App is an app that uses Machine Learning algorithm to predict whether a loan applicant would default paying back the loan of not.
             The data is taken from the Kaggle (https://www.kaggle.com/datasets/gauravduttakiit/loan-defaulter/data). This is then processed, analyzed and used to train the model.
             Users can use details of loan applicants to then predict the possibility of defaulting the loan payment.""")

if selected == "Explore":
    st.write("""## Explore the Demography of Loan Defaulter Data""")

    chart_opt = [
        "Type of Loan Contract", 
        "Gender", 
        "Owns House", 
       "Source of Income",
        "Level of Education",
        "Type of Housing",
        "Type of Occupation",
        "Type of Residency",
        ]
    
    chart = st.selectbox("Select analysis:", chart_opt)

    # Ploting analysis
    if chart == "Type of Loan Contract":

        # Create a grouped bar chart using Plotly
        fig1 = go.Figure()
        # Group the data by 'NAME_CONTRACT_TYPE' and 'TARGET' and calculate the count
        grouped_data = data.groupby(['NAME_CONTRACT_TYPE', 'TARGET']).size().reset_index(name='count')
        # Create bars for each 'NAME_CONTRACT_TYPE'
        for contract_type in grouped_data['NAME_CONTRACT_TYPE'].unique():
            subset = grouped_data[grouped_data['NAME_CONTRACT_TYPE'] == contract_type]
            fig1.add_trace(go.Bar(x=subset['TARGET'], y=subset['count'], name=contract_type))
        # Customize the layout
        fig1.update_layout(
            title="Value Count of Loan Defaulter by Type of Loan Contract",
            xaxis_title="Loan Default",
            yaxis_title="Count [in Logarithm]",
            xaxis=dict(tickvals=[0, 1], ticktext=["No", "Yes"]),  # Customize x-axis labels
            yaxis_type="log",  # Logarithmic scale for the y-axis
        )


        # Display the Plotly chart using Streamlit
        st.plotly_chart(fig1)

    
    if chart == "Gender":
        # Create a grouped bar chart using Plotly
        fig2 = go.Figure()
        
        grouped_data = data.groupby(['CODE_GENDER', 'TARGET']).size().reset_index(name='count')
       
        for contract_type in grouped_data['CODE_GENDER'].unique():
            subset = grouped_data[grouped_data['CODE_GENDER'] == contract_type]
            fig2.add_trace(go.Bar(x=subset['TARGET'], y=subset['count'], name=contract_type))
        # Customize the layout
        fig2.update_layout(
            title="Value Count of Loan Defaulter by Gender of Applicant",
            xaxis_title="Loan Default",
            yaxis_title="Count [in Logarithm]",
            xaxis=dict(tickvals=[0, 1], ticktext=["No", "Yes"]),  # Customize x-axis labels
            yaxis_type="log",  # Logarithmic scale for the y-axis
        )


        st.plotly_chart(fig2)
    
    if chart == "Owns House":
        # Create a grouped bar chart using Plotly
        fig3 = go.Figure()
        
        grouped_data = data.groupby(['FLAG_OWN_REALTY', 'TARGET']).size().reset_index(name='count')
       
        for contract_type in grouped_data['FLAG_OWN_REALTY'].unique():
            subset = grouped_data[grouped_data['FLAG_OWN_REALTY'] == contract_type]
            fig3.add_trace(go.Bar(x=subset['TARGET'], y=subset['count'], name=contract_type))
        # Customize the layout
        fig3.update_layout(
            title="Value Count of Loan Defaulter by House Ownership",
            xaxis_title="Loan Default",
            yaxis_title="Count [in Logarithm]",
            xaxis=dict(tickvals=[0, 1], ticktext=["No", "Yes"]),  # Customize x-axis labels
            yaxis_type="log",  # Logarithmic scale for the y-axis
        )

        st.plotly_chart(fig3)
    
    if chart == "Source of Income":
        # Create a grouped bar chart using Plotly
        fig4 = go.Figure()
        
        grouped_data = data.groupby(['NAME_INCOME_TYPE', 'TARGET']).size().reset_index(name='count')
       
        for contract_type in grouped_data['NAME_INCOME_TYPE'].unique():
            subset = grouped_data[grouped_data['NAME_INCOME_TYPE'] == contract_type]
            fig4.add_trace(go.Bar(x=subset['TARGET'], y=subset['count'], name=contract_type))
        # Customize the layout
        fig4.update_layout(
            title="Value Count of Loan Defaulter by Souce of Income",
            xaxis_title="Loan Default",
            yaxis_title="Count [in Logarithm]",
            xaxis=dict(tickvals=[0, 1], ticktext=["No", "Yes"]),  # Customize x-axis labels
            yaxis_type="log",  # Logarithmic scale for the y-axis
        )


        st.plotly_chart(fig4)
    
    if chart == "Level of Education":
        # Create a grouped bar chart using Plotly
        fig5 = go.Figure()
        
        grouped_data = data.groupby(['NAME_EDUCATION_TYPE', 'TARGET']).size().reset_index(name='count')
       
        for contract_type in grouped_data['NAME_EDUCATION_TYPE'].unique():
            subset = grouped_data[grouped_data['NAME_EDUCATION_TYPE'] == contract_type]
            fig5.add_trace(go.Bar(x=subset['TARGET'], y=subset['count'], name=contract_type))
        # Customize the layout
        fig5.update_layout(
            title="Value Count of Loan Defaulter by Level of Education",
            xaxis_title="Loan Default",
            yaxis_title="Count [in Logarithm]",
            xaxis=dict(tickvals=[0, 1], ticktext=["No", "Yes"]),  # Customize x-axis labels
            yaxis_type="log",  # Logarithmic scale for the y-axis
        )


        st.plotly_chart(fig5)

    if chart == "Type of Housing":
        # Create a grouped bar chart using Plotly
        fig6 = go.Figure()
        
        grouped_data = data.groupby(['NAME_HOUSING_TYPE', 'TARGET']).size().reset_index(name='count')
       
        for contract_type in grouped_data['NAME_HOUSING_TYPE'].unique():
            subset = grouped_data[grouped_data['NAME_HOUSING_TYPE'] == contract_type]
            fig6.add_trace(go.Bar(x=subset['TARGET'], y=subset['count'], name=contract_type))
        # Customize the layout
        fig6.update_layout(
            title="Value Count of Loan Defaulter by Type of House Living in",
            xaxis_title="Loan Default",
            yaxis_title="Count [in Logarithm]",
            xaxis=dict(tickvals=[0, 1], ticktext=["No", "Yes"]),  # Customize x-axis labels
            yaxis_type="log",  # Logarithmic scale for the y-axis
        )

        st.plotly_chart(fig6)

    if chart == "Type of Occupation":
        # Create a grouped bar chart using Plotly
        fig7 = go.Figure()
        
        grouped_data = data.groupby(['OCCUPATION_TYPE', 'TARGET']).size().reset_index(name='count')
       
        for contract_type in grouped_data['OCCUPATION_TYPE'].unique():
            subset = grouped_data[grouped_data['OCCUPATION_TYPE'] == contract_type]
            fig7.add_trace(go.Bar(x=subset['TARGET'], y=subset['count'], name=contract_type))
        # Customize the layout
        fig7.update_layout(
            title="Value Count of Loan Defaulter by Type of Occupation",
            xaxis_title="Loan Default",
            yaxis_title="Count [in Logarithm]",
            xaxis=dict(tickvals=[0, 1], ticktext=["No", "Yes"]),  # Customize x-axis labels
            yaxis_type="log",  # Logarithmic scale for the y-axis
        )


        st.plotly_chart(fig7)

    if chart == "Type of Residency":
        # Create a grouped bar chart using Plotly
        fig8 = go.Figure()
        
        grouped_data = data.groupby(['REGION_RATING_CLIENT', 'TARGET']).size().reset_index(name='count')
       
        for contract_type in grouped_data['REGION_RATING_CLIENT'].unique():
            subset = grouped_data[grouped_data['REGION_RATING_CLIENT'] == contract_type]
            fig8.add_trace(go.Bar(x=subset['TARGET'], y=subset['count'], name=contract_type))
        # Customize the layout
        fig8.update_layout(
            title="Value Count of Loan Defaulter by Type of Residency",
            xaxis_title="Loan Default",
            yaxis_title="Count [in Logarithm]",
            xaxis=dict(tickvals=[0, 1], ticktext=["No", "Yes"]),  # Customize x-axis labels
            yaxis_type="log",  # Logarithmic scale for the y-axis
        )

        st.plotly_chart(fig8)




if selected == "Prediction":

    X = data.drop('TARGET', axis = 1)
    y = data['TARGET']
    y = y.apply(lambda x: 1 if x == 'Yes' else 0)
    y = y.values

    # Set encoders
    contract_encoder = LabelEncoder()
    gender_encoder = LabelEncoder()
    own_house_encoder = LabelEncoder()
    income_encoder = LabelEncoder()
    education_encoder = LabelEncoder()
    housing_encoder = LabelEncoder()
    occupation_encoder = LabelEncoder()
    region_encoder = LabelEncoder()
    organization_encoder = LabelEncoder()

    # Encode variables
    X['NAME_CONTRACT_TYPE'] = contract_encoder.fit_transform(X['NAME_CONTRACT_TYPE'])
    X['CODE_GENDER'] = gender_encoder.fit_transform(X['CODE_GENDER'])
    X['FLAG_OWN_REALTY'] = own_house_encoder.fit_transform(X['FLAG_OWN_REALTY'])
    X['NAME_INCOME_TYPE'] = income_encoder.fit_transform(X['NAME_INCOME_TYPE'])
    X['NAME_EDUCATION_TYPE'] = education_encoder.fit_transform(X['NAME_EDUCATION_TYPE'])
    X['NAME_HOUSING_TYPE'] = housing_encoder.fit_transform(X['NAME_HOUSING_TYPE'])
    X['OCCUPATION_TYPE'] = occupation_encoder.fit_transform(X['OCCUPATION_TYPE'])
    X['REGION_RATING_CLIENT'] = region_encoder.fit_transform(X['REGION_RATING_CLIENT'])
    X['ORGANIZATION_TYPE'] = organization_encoder.fit_transform(X['ORGANIZATION_TYPE'])

    dec = DecisionTreeClassifier()
    dec.fit(X, y)
    accuracy = dec.score(X, y)
    #print('Decision Treet Classifier Accuracy =',(accuracy*100).round(2),'%')



    # Building the app
    st.title("Bank Loan Default Prediction App")
    st.write("""#### Imput the following details to predict loan defaulting.""")

    # Define some terms
    loan_type = ['Cash loans', 'Revolving loans']
    gender = ['Male', 'Female']
    owns_house = ['Yes', 'No']
    income_type = ['Working', 
                   'State servant', 
                   'Commercial associate', 
                   'Pensioner',
                   'Unemployed', 
                   'Student', 
                   'Businessman', 
                   'Maternity leave']
    edu_level = ['Junior High', 'Senior High', 'Primary or None', 'Academic degree']
    housing_type = ['Owns House/Apartment', 
                  'Rented apartment', 
                  'With parents',
                  'Municipal apartment', 
                  'Office apartment', 
                  'Co-op apartment']
    ocupation_type = ['Laborers', 
                      'Core staff', 
                      'Accountants', 
                      'Managers',
                      'Drivers', 
                      'Sales staff', 
                      'Cleaning staff', 
                      'Cooking staff',
                    'Private service staff', 
                    'Medicine staff', 
                    'Security staff',
                    'IT staff', 
                    'Waiters/barmen', 
                    'Realty agents', 
                    'Secretaries',
                    'HR staff']
    residency = ['Town', 'City', 'Rural']
    organization_type = ['Business', 'Education', 'Government', 'Religion', 'Other',
                        'Electricity', 'Medicine', 'Self-employed', 'Transport',
                        'Construction', 'Housing', 'Trade', 'Industry', 'Military',
                        'Services', 'Security Ministries', 'Emergency', 'Security',
                        'Police', 'Postal', 'Agriculture', 'Restaurant', 'Culture',
                        'Hotel', 'Bank', 'Insurance', 'Mobile', 'Legal Services',
                        'Advertising', 'Cleaning', 'Telecom', 'Realtor']


    #Selections
    loan_type = st.selectbox("Select loan type: ", loan_type)
    gender = st.selectbox("Gender:", gender)
    owns_house = st.selectbox("Applicant owns house(s):", owns_house)
    income = st.slider("Total income:", 100, 50000, 1000, step = 50)
    loan_amt = st.slider("Loan amount:", 100, 500000, 1000, step = 50)
    income_type = st.selectbox("Current souce of income:", income_type)
    edu_level = st.selectbox("Select highest level of education: ", edu_level)
    housing_type = st.selectbox("House living in: ", housing_type)
    age = st.slider("Pick age:", 18, 80, 25)
    ocupation_type = st.selectbox("Type of Occupation: ", ocupation_type)
    household = st.slider("Number of people in household:", 1, 50, 2)
    residency = st.selectbox("Type of residencial area: ", residency)
    organization_type = st.selectbox("Type of organization working in: ", organization_type)
    pred_btn = st.button("Predict Default Possibility")

    # Get salary prediction
    if pred_btn:
        X_test = np.array([[
            loan_type,
            gender,
            owns_house,
            income,
            loan_amt,
            income_type,
            edu_level,
            housing_type,
            age,
            ocupation_type,
            household,
            residency,
            organization_type
        ]])

        X_test[:,0] = contract_encoder.transform(X_test[:,0])
        X_test[:,1] = gender_encoder.transform(X_test[:,1])
        X_test[:,2] = own_house_encoder.transform(X_test[:,2])
        X_test[:,5] = income_encoder.transform(X_test[:,5])
        X_test[:,6] = education_encoder.transform(X_test[:,6])
        X_test[:,7] = housing_encoder.transform(X_test[:,7])
        X_test[:,9] = occupation_encoder.transform(X_test[:,9])
        X_test[:,11] = region_encoder.transform(X_test[:,11])
        X_test[:,12] = organization_encoder.fit_transform(X_test[:,12])

        X_test = X_test.astype(float)


        pred = dec.predict(X_test)
        accuracy = (accuracy*100).round(2)
        if pred[0] == 0:
            st.write(f"""#### It is with {accuracy}% accuracy that the loan would NOT be defulated""")
        else:
            st.write(f"""#### It is with {accuracy}% accuracy that the loan would be defulated""")


if selected == "Contact":
    st.title("Get in touch")
    st.write("""#### Email: gamahrichard5@gmail.com""")
    st.write("""#### GitHub: someone@gmail.com""")
    st.write("""#### WhatsApp: https://wa.me/233542124371""")