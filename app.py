import streamlit as st
import pandas as pd
import numpy as np
import joblib
import sqlite3
import plotly.express as px
from math import radians, sin, cos, asin, sqrt

# ==================== PAGE CONFIG ====================
st.set_page_config(page_title="Amazon Delivery Dashboard - Aqua Theme", layout="wide")

# ==================== CUSTOM STYLES ====================
st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;600;800&display=swap');

        * {font-family: 'Poppins', sans-serif;}

        /* Animated gradient background */
        @keyframes pulseGradient {
            0% {background-position: 0% 50%;}
            50% {background-position: 100% 50%;}
            100% {background-position: 0% 50%;}
        }

        .stApp {
            background: linear-gradient(-45deg, #001f3f, #00796b, #26c6da, #6a1b9a);
            background-size: 300% 300%;
            animation: pulseGradient 15s ease infinite;
            color: #e0f7fa !important;
        }

        [data-testid="stSidebar"] {
            background: linear-gradient(to bottom, #0d2b45, #1a3c58);
            color: #b2ebf2;
        }

        h1, h2, h3 {
            text-shadow: 1px 1px 4px rgba(0, 0, 0, 0.3);
            color: #e0f7fa;
        }

        /* Predict Button Styling */
        div.stButton > button:first-child {
            background: linear-gradient(90deg, #00acc1, #26c6da, #00acc1);
            background-size: 200% 100%;
            animation: shimmer 3s infinite linear;
            color: white !important;
            border-radius: 16px !important;
            padding: 1.1em 1.6em !important;
            font-weight: 900 !important;
            font-size: 1.4em !important;
            border: none !important;
            width: 100% !important;
            transition: all 0.3s ease-in-out;
            box-shadow: 0px 0px 12px rgba(38, 198, 218, 0.7);
        }

        @keyframes shimmer {
            0% {background-position: 0%;}
            100% {background-position: 200%;}
        }

        div.stButton > button:hover {
            transform: scale(1.05);
            box-shadow: 0px 0px 18px rgba(0, 255, 255, 0.9);
        }

        /* Table text contrast fix */
        .stDataFrame {color: black !important;}
    </style>
""", unsafe_allow_html=True)

# ==================== DATABASE ====================
conn = sqlite3.connect('amazon_delivery.db', check_same_thread=False)
cursor = conn.cursor()
cursor.execute('''
    CREATE TABLE IF NOT EXISTS predictions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        distance REAL,
        agent_age INTEGER,
        agent_rating REAL,
        weather TEXT,
        traffic TEXT,
        vehicle TEXT,
        area TEXT,
        category TEXT,
        predicted_time REAL
    )
''')
conn.commit()

# ==================== HELPER FUNCTIONS ====================
def haversine(lat1, lon1, lat2, lon2):
    """Calculate distance in kilometers between two coordinates."""
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
    dlon, dlat = lon2 - lon1, lat2 - lat1
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a))
    return 6371 * c

@st.cache_resource
def load_model():
    """Load trained ML model."""
    return joblib.load("best_delivery_model.pkl")

model = load_model()

# ==================== SIDEBAR ====================
st.sidebar.image("https://cdn-icons-png.flaticon.com/512/1048/1048313.png", width=100)
st.sidebar.title("📦 Navigation")
page = st.sidebar.radio("Go to", ["Overview", "Prediction", "Search & Filters", "Analytics", "About"])

# ==================== OVERVIEW PAGE ====================
if page == "Overview":
    st.title(" Amazon Delivery Overview")
    st.markdown("Visualize key delivery insights and performance metrics.")

    df = pd.read_csv("amazon_delivery.csv")

    if 'distance_km' not in df.columns and all(col in df.columns for col in ['Store_Latitude','Store_Longitude','Drop_Latitude','Drop_Longitude']):
        df['distance_km'] = df.apply(lambda row: haversine(
            row['Store_Latitude'], row['Store_Longitude'],
            row['Drop_Latitude'], row['Drop_Longitude']), axis=1)

    col1, col2 = st.columns(2)
    with col1:
        fig1 = px.histogram(df, x='Delivery_Time', nbins=30, title="Delivery Time Distribution", color_discrete_sequence=['#26c6da'])
        st.plotly_chart(fig1, use_container_width=True)
    with col2:
        fig2 = px.box(df, x='Traffic', y='Delivery_Time', color='Traffic', title="Delivery Time by Traffic")
        st.plotly_chart(fig2, use_container_width=True)

    if 'distance_km' in df.columns:
        fig3 = px.scatter(df, x='distance_km', y='Delivery_Time', color='Weather',
                          title="Distance vs Delivery Time by Weather")
        st.plotly_chart(fig3, use_container_width=True)
    else:
        st.warning("Distance data unavailable — skipping Distance vs Delivery Time plot.")

# ==================== PREDICTION PAGE ====================
elif page == "Prediction":
    st.title("🕒 Predict Delivery Time")
    st.markdown("Input your delivery details below to predict delivery time (in hours).")

    if "prediction" not in st.session_state:
        st.session_state.prediction = None

    col1, col2 = st.columns(2)
    distance = col1.number_input("Distance from Store to Drop (km)", min_value=0.1, max_value=200.0, value=5.0)
    agent_age = col1.slider("Agent Age", 18, 70, 30)
    agent_rating = col2.slider("Agent Rating", 0.0, 5.0, 4.5, 0.1)
    weather = col1.selectbox("Weather", ["Clear", "Cloudy", "Rainy", "Stormy"])
    traffic = col2.selectbox("Traffic", ["Low", "Medium", "High"])
    vehicle = col1.selectbox("Vehicle", ["Bike", "Car", "Van"])
    area = col2.selectbox("Area", ["Urban", "Semi-Urban", "Metropolitan"])
    category = st.selectbox("Product Category", ["Food", "Grocery", "Electronics", "Fashion", "Books", "Clothing", "Other"])

    predict_clicked = st.button(" PREDICT DELIVERY TIME")

    if predict_clicked:
        try:
            input_data = {
                'distance_km': [distance],
                'Agent_Age': [agent_age],
                'Agent_Rating': [agent_rating],
                'Weather_' + weather: [1],
                'Traffic_' + traffic: [1],
                'Vehicle_' + vehicle: [1],
                'Area_' + area: [1],
                'Category_' + category: [1]
            }
            X_input = pd.DataFrame(input_data)
            expected_cols = model.feature_names_in_
            for col in expected_cols:
                if col not in X_input.columns:
                    X_input[col] = 0
            X_input = X_input[expected_cols]

            prediction = float(model.predict(X_input)[0])
            st.session_state.prediction = prediction

            st.success(f" Estimated Delivery Time: **{prediction:.2f} hours**")

            cursor.execute('''
                INSERT INTO predictions (distance, agent_age, agent_rating, weather, traffic, vehicle, area, category, predicted_time)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (distance, agent_age, agent_rating, weather, traffic, vehicle, area, category, prediction))
            conn.commit()

        except Exception as e:
            st.error(f"Prediction error: {e}")

    if st.session_state.prediction is not None and not predict_clicked:
        st.info(f"Last Predicted Delivery Time: **{st.session_state.prediction:.2f} hours**")

# ==================== SEARCH & FILTERS PAGE ====================
elif page == "Search & Filters":
    st.title(" Search & Filter Predictions")
    df_db = pd.read_sql_query("SELECT * FROM predictions", conn)
    col1, col2 = st.columns(2)
    weather_filter = col1.multiselect("Filter by Weather", df_db['weather'].unique())
    traffic_filter = col2.multiselect("Filter by Traffic", df_db['traffic'].unique())

    df_filtered = df_db.copy()
    if weather_filter:
        df_filtered = df_filtered[df_filtered['weather'].isin(weather_filter)]
    if traffic_filter:
        df_filtered = df_filtered[df_filtered['traffic'].isin(traffic_filter)]

    st.dataframe(df_filtered, use_container_width=True)

# ==================== ANALYTICS PAGE ====================
elif page == "Analytics":
    st.title(" Advanced Analytics")
    st.markdown("Explore interactive visual insights.")

    df = pd.read_csv("amazon_delivery.csv")
    if 'distance_km' not in df.columns and all(col in df.columns for col in ['Store_Latitude','Store_Longitude','Drop_Latitude','Drop_Longitude']):
        df['distance_km'] = df.apply(lambda row: haversine(
            row['Store_Latitude'], row['Store_Longitude'],
            row['Drop_Latitude'], row['Drop_Longitude']), axis=1)

    col1, col2 = st.columns(2)
    with col1:
        fig = px.box(df, x='Weather', y='Delivery_Time', color='Weather', title="Delivery Time by Weather")
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        fig = px.box(df, x='Area', y='Delivery_Time', color='Area', title="Delivery Time by Area")
        st.plotly_chart(fig, use_container_width=True)

    corr = df.select_dtypes(include=[np.number]).corr()
    fig_corr = px.imshow(corr, text_auto=True, color_continuous_scale="Tealgrn", title="Correlation Heatmap")
    st.plotly_chart(fig_corr, use_container_width=True)

# ==================== ABOUT PAGE ====================
elif page == "About":
    st.title(" About This Project")
    st.markdown("""
    **Amazon Delivery Time Prediction Dashboard (Aqua Edition)**  
    This project predicts delivery times using ML, interactive charts, and a modern UI.  
    - **Tech:** Python, Pandas, Scikit-learn, Plotly, Streamlit, SQLite  
    - **Goal:** Optimize delivery logistics and improve agent efficiency  
    - **Theme:** Aqua gradient with glowing effects and smooth animations  
    - **Developer:** Mrityunjay Acharya 
    """)
