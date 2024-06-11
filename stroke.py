import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt

# Set the page config
st.set_page_config(
    page_title="Stroke Prediction Dataset",
    layout="wide",
    page_icon=":brain:"
)

# Load the dataset
import pandas as pd
import numpy as np

# Read the DataFrame from the CSV file
old_df = pd.read_csv("cleaned_stroke.csv")

def round_decimals_to_whole_numbers(df):
    """
    Detect decimal values in numerical columns and round them to the nearest whole number.
    
    Parameters:
    df (pd.DataFrame): The input DataFrame containing numerical columns with decimal values.
    
    Returns:
    pd.DataFrame: The DataFrame with decimal values rounded to whole numbers.
    """
    # Select numerical columns
    numerical_columns = df.select_dtypes(include=[np.number]).columns
    
    # Round decimal values to the nearest whole number
    for column in numerical_columns:
        df[column] = df[column].round().astype(int)
    
    return df

import inflect

# Initialize the inflect engine
p = inflect.engine()

# Read the DataFrame from the CSV file
df = pd.read_csv("cleaned_stroke.csv")

# Define mappings for categorical values
gender_map = {0: "Male", 1: "Female", 2: "Other"}
hypertension_map = {0: "No", 1: "Yes"}
heart_disease_map = {0: "No", 1: "Yes"}
ever_married_map = {0: "No", 1: "Yes"}
work_type_map = {0: "children", 1: "Govt_job", 2: "Never_worked", 3: "Private", 4: "Self-employed"}
residence_type_map = {0: "Rural", 1: "Urban"}
smoking_status_map = {0: "formerly smoked", 1: "never smoked", 2: "smokes", 3: "Unknown"}

def convert_values_to_words(df):
    # Convert numerical values to words
    # for column in ["age", "avg_glucose_level", "bmi"]:
    #     df[column] = df[column].apply(lambda x: p.number_to_words(int(x)) if not pd.isnull(x) else x)
    
    # Convert categorical values to words
    df["gender"] = df["gender"].map(gender_map)
    df["hypertension"] = df["hypertension"].map(hypertension_map)
    df["heart_disease"] = df["heart_disease"].map(heart_disease_map)
    df["ever_married"] = df["ever_married"].map(ever_married_map)
    df["work_type"] = df["work_type"].map(work_type_map)
    df["Residence_type"] = df["Residence_type"].map(residence_type_map)
    df["smoking_status"] = df["smoking_status"].map(smoking_status_map)
    
    # Rename columns
    df.rename(columns={"hypertension": "Hypertension", "heart_disease": "Heart Disease", 
                       "ever_married": "Ever Married", "work_type": "Work Type",
                       "Residence_type": "Residence Type", "smoking_status": "Smoking Status",
                       "stroke": "Stroke"}, inplace=True)
    
    return df
df = round_decimals_to_whole_numbers(old_df)
# Call the function to convert values to words
df = convert_values_to_words(df)



st.markdown("""
    <style>
    div.stButton{
        display:flex;
        flex-direction:column;
        gap:0px;
        margin: 0;
        }
    div.stButton > button {
        width: 100%;
        text-align: left;
        display: block;
        # background-color: transparent;
        border:0px solid transparent;
        border-radius:5px;
    }
    # div.stButton > button:hover {
    #     background-color:red;
    # }
    </style>
    """, unsafe_allow_html=True)

# Sidebar title
st.sidebar.title("Stroke Prediction Dataset")
st.sidebar.write("Al Monsour M. Salida")
st.sidebar.write("Github: https://github.com/almonsour13/stroke_streamlit_final.git")

st.sidebar.write("Menu")

# Initialize session state if not already initialized
if 'dashboard' not in st.session_state:
    st.session_state.dashboard = True
if 'table' not in st.session_state:
    st.session_state.table = False

# Sidebar buttons
if st.sidebar.button("Dashboard"):
    st.session_state.dashboard = True
    st.session_state.table = False
if st.sidebar.button("Table"):
    st.session_state.table = True
    st.session_state.dashboard = False
    
st.sidebar.markdown("**Dataset Source:** [Stroke Prediction Dataset (Kaggle)](https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset)")


# Main page content based on button click
if st.session_state.dashboard:
    st.header("Dashboard")
    st.write("This dataset is used to predict whether a patient is likely to get stroke based on the input parameters like gender, age, various diseases, and smoking status. Each row in the data provides relavant information about the patient.")
    st.markdown("**Source:** [Stroke Prediction Dataset (Kaggle)](https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset)")
    st.divider()
    ##################################################################################
    df['Stroke_Label'] = df['Stroke'].map({0: 'Negative', 1: 'Positive'})
    stroke_counts = df['Stroke_Label'].value_counts().reset_index()
    stroke_counts.columns = ['Stroke', 'Count']
    fig_pie = px.pie(stroke_counts, values='Count', names='Stroke', title='Stroke Distribution')
    st.plotly_chart(fig_pie)
    # st.subheader('Stroke Distribution')
    st.write('This pie chart shows the proportion of participants who have experienced a stroke versus those who have not, using custom labels for better clarity.')
    st.divider()
    ##################################################################################
    fig_age = px.histogram(df, x='age', title='Age Distribution')
    st.plotly_chart(fig_age)
    # st.subheader('Age Distribution')
    st.write('This histogram shows the distribution of ages in the dataset, providing insights into the age range of the participants.')
    st.divider()
    ##################################################################################
    gender_counts = df['gender'].value_counts().reset_index()
    gender_counts.columns = ['Gender', 'Count']
    fig_gender = px.bar(gender_counts, x='Gender', y='Count', title='Gender Distribution')
    st.plotly_chart(fig_gender)
    # st.subheader('Gender Distribution')
    st.write('This bar chart displays the distribution of gender among the participants, allowing us to see the proportion of males, females, and others in the dataset.')
    st.divider()
    ##################################################################################
    hypertension_counts = df['Hypertension'].value_counts().reset_index()
    hypertension_counts.columns = ['Hypertension', 'Count']
    fig_hypertension = px.bar(hypertension_counts, x='Hypertension', y='Count', title='Hypertension Distribution')
    st.plotly_chart(fig_hypertension)
    # st.subheader('Hypertension Distribution')
    st.write('This bar chart shows the distribution of participants with and without hypertension, highlighting the prevalence of hypertension in the dataset.')
    st.divider()
    ##################################################################################
    fig_bmi = px.box(df, y='bmi', title='BMI Distribution')
    st.plotly_chart(fig_bmi)
    # st.subheader('BMI Distribution')
    st.write('This box plot shows the distribution of BMI values in the dataset, indicating the spread and central tendency of BMI among the participants.')
    st.divider()
    ##################################################################################
    age_bins = [0, 20, 40, 60, 80, 100]
    df['age_group'] = pd.cut(df['age'], bins=age_bins)
    df['age_group_str'] = df['age_group'].astype(str)
    stroke_by_age_group = df.groupby('age_group_str')['Stroke'].mean().reset_index()
    fig_stroke_age_group = px.bar(stroke_by_age_group, x='age_group_str', y='Stroke', title='Stroke Incidence by Age Group', labels={'age_group_str': 'Age Group', 'Stroke': 'Stroke Incidence Rate'})
    st.plotly_chart(fig_stroke_age_group)
    # st.subheader('Stroke Incidence by Age Group')
    st.write('This bar chart shows the incidence of stroke across different age groups, providing insights into which age groups have higher stroke rates.')
    st.divider()
    ##################################################################################
    avg_glucose_by_age = df.groupby('age_group_str')['avg_glucose_level'].mean().reset_index()
    fig_avg_glucose_line = px.line(avg_glucose_by_age, x='age_group_str', y='avg_glucose_level', 
                                   title='Average Glucose Level by Age Group',
                               labels={'age_group_str': 'Age Group', 'avg_glucose_level': 'Average Glucose Level'},
                               markers=True)
    st.plotly_chart(fig_avg_glucose_line)
    # st.subheader('Average Glucose Level by Age Group')
    st.write('This line chart shows the trend of average glucose levels across different age groups, providing insights into how glucose levels vary with age.')
    st.divider()
    ##################################################################################
    avg_bmi_by_age = df.groupby('age_group_str')['bmi'].mean().reset_index()
    fig_avg_bmi_line = px.line(avg_bmi_by_age, x='age_group_str', y='bmi', 
                               title='Average BMI by Age Group',
                               labels={'age_group_str': 'Age Group', 'bmi': 'Average BMI'},
                               markers=True)
    st.plotly_chart(fig_avg_bmi_line)
    # st.subheader('Average BMI by Age Group')
    st.write('This line chart shows the trend of average BMI across different age groups, providing insights into how BMI varies with age.')
    st.divider()
    ##################################################################################
    ##################################################################################
        
elif st.session_state.table:
    st.header("Table view")
    st.write("The table below is the represention of the stroke dataset, each row represents an individual, and each column provides specific information about the attributes and characteristics of that individual.")
    st.dataframe(df)
