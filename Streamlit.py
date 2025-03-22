import base64
import pickle
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder

def img_to_base64(image_path):
    with open(image_path, "rb") as img_file:
        img_data = img_file.read()
        return base64.b64encode(img_data).decode()

image_path = "F:\\Guvi Projects\\Mental_Health_Survey\\Image\\Neural_Networks.webp"

try:
    img_base64 = img_to_base64(image_path)

    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: linear-gradient(rgba(0, 0, 0, 0.7), rgba(0, 0, 0, 0.7)), url('data:image/jpeg;base64,{img_base64}');
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
            display: flex;
            justify-content: center;
            align-items: center;
            height: 100vh;
            padding: 0;
        }}
        .title-container {{
            text-align: center;
            color: white;
            font-size: 4em;
            margin-top: 350px;
            margin-bottom: 20px;
            
        }}
        
        </style>
        """, unsafe_allow_html=True
    )

except FileNotFoundError:
    st.error("Image not found at the specified path.")

class MentalHealthPreprocessor:

    def __init__(self):

        self.encoders = {}
        self.columns_to_drop = []
        self.numerical_features = None
        self.categorical_features = None

    def fit(self, data):

        self.numerical_features = data.select_dtypes(include=['int64', 'float64']).columns.tolist()
        self.categorical_features = data.select_dtypes(include=['object']).columns.tolist()

        corr_matrix = data[self.numerical_features].corr().abs()
        upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        self.columns_to_drop = [column for column in upper_tri.columns if any(upper_tri[column] > 0.6)]
        data = data.drop(columns=self.columns_to_drop, errors='ignore')
        
        self.numerical_features = data.select_dtypes(include=['int64', 'float64']).columns.tolist()
        
        for col in self.categorical_features:

            if col in data.columns:
                le = LabelEncoder()
                data[col] = le.fit_transform(data[col].astype(str))
                self.encoders[col] = le  

        return data

    def transform(self, data):

        data = data.drop(columns=self.columns_to_drop, errors='ignore')

        for col in self.categorical_features:
            if col in data.columns and col in self.encoders:
                data[col] = self.encoders[col].transform(data[col].astype(str))

        return data

@st.cache_resource
def load_preprocessor():

    with open("F:\\Guvi Projects\\Mental_Health_Survey\\Model\\preprocessor.pkl", "rb") as file:

        return pickle.load(file)
    
@st.cache_resource
def load_model():

    model = tf.keras.models.load_model("F:\\Guvi Projects\\Mental_Health_Survey\\Model\\neural_network.keras")

    return model

@st.cache_resource
def load_cleaning():

    with open("F:\\Guvi Projects\\Mental_Health_Survey\\Model\\cleaning.pkl", "rb") as file:
        cleaning_df = pickle.load(file)

    return cleaning_df

def predict_depression(input_data, model, preprocessor):

    processed_data = preprocessor.transform(pd.DataFrame([input_data]))
    
    expected_features = 13  
    current_features = processed_data.shape[1]
    
    if current_features < expected_features:
        missing_features = expected_features - current_features
        processed_data = np.pad(processed_data, ((0, 0), (0, missing_features)), mode='constant')
    
    processed_data = np.array(processed_data, dtype=np.float32)
    output = model.predict(processed_data)
    prediction = "Depression Detected" if output[0] > 0.5 else "No Depression"
    return prediction, output[0][0]

st.title("Mental Health Depression Prediction Application")
st.sidebar.header("Navigation Bar ->")
page = st.sidebar.radio("Pages", ["Home", "Upload Data", "Manual Entry", "Visualizations"])

if page == "Home":
    st.write("#### Predicting Depression using Artificial Intelligence")
    st.markdown("<br>", unsafe_allow_html=True)
    st.write("##### This app allows users to predict the likelihood of depression based on survey responses.")
    st.write("##### Depression is a mental health disorder that affects millions worldwide, impacting mood, energy levels, and daily life.")
    st.write("##### It can range from mild sadness to severe clinical depression, requiring professional intervention.")
    st.markdown("<br>", unsafe_allow_html=True)
    st.write("### Final Thought -")
    st.write("##### Depression is a serious but treatable condition. Small daily actions can improve mental health and prevent depression.")
    st.write("##### If you or someone you know struggles with depression, reach out for support always —> you're not alone !!!")

if "predicted_data" not in st.session_state:
    st.session_state["predicted_data"] = pd.DataFrame()

if page == "Upload Data":

    st.header("Upload Survey Data")
    uploaded_file = st.file_uploader("Upload CSV File", type=["csv"])

    if uploaded_file:
        data = pd.read_csv(uploaded_file)
        st.write("### Preview of Uploaded Data")
        st.dataframe(data.head())
        
        cleaned_data = load_cleaning()
        st.write("### Cleaned Data Preview")
        st.dataframe(cleaned_data.head())
        
        preprocessor = load_preprocessor()
        model = load_model()
        
        preprocessed_data = preprocessor.transform(cleaned_data.copy())

        expected_features = 13
        if preprocessed_data.shape[1] > expected_features:
            preprocessed_data = preprocessed_data.iloc[:, :expected_features]

        if st.button("Predict Depression Risk"):
            predictions = model.predict(preprocessed_data)
            prediction_labels = [1 if pred > 0.5 else 0 for pred in predictions]
            confidence_scores = [round(float(pred), 2) for pred in predictions]

            st.write("#### Prediction Labels \n ##### 1 -> Depression Detected \n ##### 0 -> No Depression")
            
            result_df = cleaned_data.copy()
            result_df["Prediction"] = prediction_labels
            result_df["Confidence Score"] = confidence_scores

            st.session_state["predicted_data"] = result_df
            st.write("#### Prediction Results")
            st.dataframe(result_df.head())

            csv = result_df.to_csv(index=False).encode('utf-8')
            st.download_button("Download Predictions", data=csv, file_name="predictions.csv", mime="text/csv")
        
elif page == "Manual Entry":

    st.header("Manual Data Entry")

    model = load_model()
    preprocessor = load_preprocessor()
    
    working_status = st.selectbox("Working Professional or Student", ["Professional", "Student"])

    sleep = ["6-8 hours", "Less than 5 hours", "5-6 hours", "More than 8 hours"]

    city = ["Mumbai", "Delhi", "Bangalore", "Hyderabad", "Ahmedabad", "Chennai", "Kolkata", "Pune", "Jaipur", "Lucknow",
            "Kanpur", "Nagpur", "Indore", "Thane", "Bhopal", "Visakhapatnam", "Patna", "Vadodara", "Vasai-Virar", "Ghaziabad", "Ludhiana",
            "Agra", "Nashik", "Faridabad", "Meerut", "Rajkot", "Varanasi", "Srinagar", "Aurangabad", "Dhanbad", "Amritsar", "Kalyan"]
    
    profession = ["Teacher", "Content Writer", "Architect", "Consultant", "HR Manager", "Pharmacist", "Doctor", "Business Analyst", "Entrepreneur",
                   "Chemist", "Educational Consultant", "Chef", "Data Scientist", "Researcher", "Lawyer", "Customer Support", "Marketing Manager", "Pilot", "Travel Consultant",
                   "Plumber", "Sales Executive", "Manager", "Judge", "Electrician", "Financial Analyst", "Software Engineer", "Civil Engineer", "UX/UI Designer", "Digital Marketer", 
                   "Accountant", "Finanancial Analyst", "Mechanical Engineer", "Graphic Designer", "Research Analyst", "Investment Banker", "Family Consultant", "Dev", "Analyst"]
    
    degree = ["Class 12", "B.Ed", "B.Arch", "B.Com", "BCA", "B.Pharm", "M.Ed", "MCA", "BBA", "BSc", "MSc", "LLM", "M.Tech", "B.Tech", "M.Pharm", "LLB", "BHM", "MBA", "BA",
              "ME", "MD", "MHM", "BE", "PhD", "M.Com", "MBBS", "MA", "M.Arch"]
    
    diet = ["Moderate", "Unhealthy", "Healthy"]
    
    input_data = {
        "Gender": st.selectbox("Select Gender", ["Male", "Female"]),
        "Age": st.slider("Select Age", 18, 80, 1),
        "City": st.selectbox("Select City", city),
        "Sleep Duration": st.selectbox("Select Sleep Duration", sleep),
        "Dietary Habits": st.selectbox("Select Dietary Habits", diet),
        "Have you ever had suicidal thoughts ?": st.selectbox("Have you ever had suicidal thoughts?", ["Yes", "No"]),
        "Work/Study Hours": st.slider("Enter Work/Study Hours", 0, 24, 1),
        "Financial Stress": st.slider("Financial Stress", 0, 5, 1),
        "Family History of Mental Illness": st.selectbox("Family History of Mental Illness", ["Yes", "No"]),
    }
    
    if working_status == "Professional":
        input_data.update({
            "Profession": st.selectbox("Enter Profession", profession),
            "Work Pressure": st.slider("Work Pressure", 0, 10, 5),
            "Job Satisfaction": st.slider("Job Satisfaction", 0, 10, 5),
        })
    else:
        input_data.update({
            "Academic Pressure": st.slider("Academic Pressure", 0, 10, 5),
            "CGPA": st.number_input("CGPA", min_value=0.0, max_value=10.0, value=6.0),
            "Study Satisfaction": st.slider("Study Satisfaction", 0, 10, 5),
            "Degree": st.selectbox("Enter Degree", degree),
        })
    
    if st.button("Predict Depression Risk"):
        prediction, confidence = predict_depression(input_data, model, preprocessor)
        st.write(f"#### Prediction: \n {prediction}")
        st.write(f"#### Confidence Score: \n {confidence:.2f}")

elif page == "Visualizations":

    st.header("Data Insights & Visualizations")

    preprocessor = load_preprocessor()
    model = load_model()
    cleaned_data = load_cleaning()
        
    preprocessed_data = preprocessor.transform(cleaned_data.copy())

    expected_features = 13

    if preprocessed_data.shape[1] > expected_features:
        preprocessed_data = preprocessed_data.iloc[:, :expected_features]
    
    if not st.session_state["predicted_data"].empty:
        st.write("#### Predicted Data Overview")
        st.dataframe(st.session_state["predicted_data"].head())

        numeric_data = st.session_state["predicted_data"].select_dtypes(include=[np.number])

        if not numeric_data.empty:
            st.write("#### Correlation Heatmap (Predicted Data)")
            plt.figure(figsize=(10, 6))
            sns.heatmap(numeric_data.corr(), annot=True, cmap="coolwarm")
            st.pyplot(plt)

        else:
            st.warning("No numeric columns available for correlation heatmap.")

        st.write("#### Depression Distribution in Predictions")

        fig, ax = plt.subplots()
        sns.countplot(x="Prediction", data=st.session_state["predicted_data"], ax=ax)
        st.pyplot(fig)

    else:
        st.warning("No predicted data available. Please run predictions in the Upload Data section first.")

    