import streamlit as st
import pandas as pd
import joblib
import os

# Define the path to your trained model
MODEL_PATH = 'models/student_model.pkl'

# Set up the web page
st.set_page_config(page_title="Student Predictor", page_icon="🎓", layout="wide")

# Check if the model exists
if not os.path.exists(MODEL_PATH):
    st.error("🚨 Model not found! Please run your training script first to generate 'student_model.pkl' inside the 'models' folder.")
else:
    # Load the trained model
    model = joblib.load(MODEL_PATH)
    
    st.title("🎓 Student Performance Prediction System")

    # Create tabs
    tab1, tab2 = st.tabs(["👤 Individual Prediction", "📊 Educator Dashboard (Bulk & Analytics)"])

    # ==========================================
    # TAB 1: INDIVIDUAL STUDENT PREDICTION
    # ==========================================
    with tab1:
        st.write("Enter a single student's study habits and previous scores below.")
        
        col1, col2 = st.columns(2)
        with col1:
            hours_studied = st.number_input("Hours Studied per Day", min_value=0.0, max_value=24.0, value=5.0)
            sleep_hours = st.number_input("Hours of Sleep per Night", min_value=0.0, max_value=24.0, value=7.0)
            
        with col2:
            previous_scores = st.number_input("Previous Scores (0-100)", min_value=0.0, max_value=100.0, value=75.0)
            sample_papers = st.number_input("Sample Papers Practiced", min_value=0, max_value=100, value=2)

        if st.button("Predict Performance", type="primary"):
            input_data = pd.DataFrame({
                'Hours_Studied': [hours_studied],
                'Previous_Scores': [previous_scores],
                'Sleep_Hours': [sleep_hours],
                'Sample_Question_Papers_Practiced': [sample_papers]
            })

            prediction = model.predict(input_data)[0]
            final_score = max(0, min(100, prediction))
            st.success(f"### Expected Performance Index: {final_score:.2f} / 100")

    # ==========================================
    # TAB 2: EDUCATOR DASHBOARD (BULK & GRAPHS)
    # ==========================================
    with tab2:
        st.write("Upload a CSV file containing your class roster to generate predictions and instant analytics.")
        
        uploaded_file = st.file_uploader("Upload Class Data (CSV)", type="csv")

        if uploaded_file is not None:
            bulk_data = pd.read_csv(uploaded_file)
            
            if st.button("Generate Class Predictions & Analytics", key="bulk_predict"):
                try:
                    # 1. Generate predictions
                    features = bulk_data[['Hours_Studied', 'Previous_Scores', 'Sleep_Hours', 'Sample_Question_Papers_Practiced']]
                    predictions = model.predict(features)
                    
                    # 2. Add to dataframe and clean up
                    bulk_data['Predicted_Performance_Index'] = predictions
                    bulk_data['Predicted_Performance_Index'] = bulk_data['Predicted_Performance_Index'].clip(0, 100).round(2)
                    
                    st.success("✅ Predictions generated successfully!")
                    
                    # 3. Display Data
                    st.dataframe(bulk_data, use_container_width=True)
                    
                    st.divider()
                    st.header("📈 Class Performance Analytics")
                    
                    # 4. Generate Interactive Graphs
                    graph_col1, graph_col2 = st.columns(2)
                    
                    with graph_col1:
                        st.subheader("Distribution of Predicted Scores")
                        st.write("See how the class grades are spread out.")
                        # We use a simple line/area chart or bar chart for distribution
                        st.bar_chart(bulk_data['Predicted_Performance_Index'])
                        
                        
                    with graph_col2:
                        st.subheader("Hours Studied vs. Predicted Score")
                        st.write("Analyze the direct impact of study time on final performance.")
                        # Scatter plot mapping study time to the final score
                        st.scatter_chart(
                            data=bulk_data,
                            x='Hours_Studied',
                            y='Predicted_Performance_Index'
                        )
                        
                    
                    st.divider()
                    
                    # 5. Export Data
                    csv_export = bulk_data.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="📥 Download Extended Data (CSV)",
                        data=csv_export,
                        file_name='class_predictions_analytics.csv',
                        mime='text/csv',
                    )
                except KeyError:

                    st.error("🚨 Error: Your CSV is missing required columns. Please check the column names.")
