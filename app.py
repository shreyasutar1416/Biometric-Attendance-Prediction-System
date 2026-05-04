import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import timedelta

# Set page config
st.set_page_config(page_title="Biometric Attendance Prediction", layout="wide")

@st.cache_resource
def load_model():
    model_path = r"c:\Users\HP\Desktop\biometric attendance prediction\models\rf_attendance_model.joblib"
    if os.path.exists(model_path):
        return joblib.load(model_path)
    return None

@st.cache_data
def load_data():
    data_path = r"c:\Users\HP\Desktop\biometric attendance prediction\data\processed_attendance.csv"
    if os.path.exists(data_path):
        df = pd.read_csv(data_path)
        df['Date'] = pd.to_datetime(df['Date'])
        return df
    return None

def main():
    st.title("🎓 Biometric Attendance Prediction Dashboard")
    st.markdown("Predict the likelihood of a student's future attendance using historical biometric data.")
    
    model = load_model()
    df = load_data()
    
    if model is None or df is None:
        st.error("Model or data not found! Please run the training script first.")
        return
        
    st.sidebar.header("Search Student")
    search_type = st.sidebar.radio("Search by:", ["Name", "Admission No"])
    search_query = st.sidebar.text_input(f"Enter {search_type}")
    
    if search_query:
        if search_type == "Name":
            student_df = df[df['Name'].str.contains(search_query, case=False, na=False)]
        else:
            student_df = df[df['Admission No'].astype(str).str.contains(search_query, case=False, na=False)]
            
        if student_df.empty:
            st.warning("No student found with that query.")
        else:
            # Let user select exact student if multiple matches
            unique_students = student_df[['student_session_id', 'Name', 'Admission No']].drop_duplicates()
            if len(unique_students) > 1:
                selected_idx = st.sidebar.selectbox(
                    "Multiple students found. Please select one:",
                    options=range(len(unique_students)),
                    format_func=lambda x: f"{unique_students.iloc[x]['Name']} ({unique_students.iloc[x]['Admission No']})"
                )
                selected_student = unique_students.iloc[selected_idx]
            else:
                selected_student = unique_students.iloc[0]
                st.sidebar.success(f"Found: {selected_student['Name']}")
                
            # Filter data for selected student
            s_data = df[df['student_session_id'] == selected_student['student_session_id']].sort_values(by='Date')
            
            st.header(f"Student: {selected_student['Name']}")
            st.subheader(f"Admission No: {selected_student['Admission No']}")
            
            # --- Predictions ---
            st.markdown("---")
            st.subheader("🔮 Predictions for the Upcoming Week")
            
            # Get last known state
            last_record = s_data.iloc[-1]
            last_date = last_record['Date']
            
            # Generate features for next 7 days
            future_dates = [last_date + timedelta(days=i) for i in range(1, 8)]
            
            curr_prev_day_attendance = last_record['is_present']
            curr_cumulative_absences = last_record['cumulative_absences']
            curr_rolling_7_day_rate = last_record['rolling_7_day_rate']
            
            predictions = []
            
            for d in future_dates:
                day_of_week = d.dayofweek
                is_weekend = 1 if day_of_week in [5, 6] else 0
                
                # Create feature array
                features = pd.DataFrame([{
                    'day_of_week': day_of_week,
                    'is_weekend': is_weekend,
                    'prev_day_attendance': curr_prev_day_attendance,
                    'rolling_7_day_rate': curr_rolling_7_day_rate,
                    'cumulative_absences': curr_cumulative_absences
                }])
                
                # Predict probability
                prob_present = model.predict_proba(features)[0][1]
                predicted_class = 1 if prob_present >= 0.5 else 0
                
                predictions.append({
                    'Date': d.strftime("%Y-%m-%d"),
                    'Day': d.strftime("%A"),
                    'Probability of Attendance': f"{prob_present * 100:.1f}%",
                    'Prediction': 'Present' if predicted_class == 1 else 'Absent'
                })
                
                # Update features for next iteration (simulation)
                curr_prev_day_attendance = predicted_class
                if predicted_class == 0:
                    curr_cumulative_absences += 1
                # Simplified rolling update for demo
                curr_rolling_7_day_rate = (curr_rolling_7_day_rate * 6 + predicted_class) / 7.0
                
            pred_df = pd.DataFrame(predictions)
            
            # Style the dataframe
            def color_prediction(val):
                color = 'green' if val == 'Present' else 'red'
                return f'color: {color}'
                
            st.dataframe(pred_df.style.map(color_prediction, subset=['Prediction']), use_container_width=True)
            
            # --- Visualizations ---
            st.markdown("---")
            st.subheader("📊 Historical Attendance Trends")
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Overall Attendance pie chart
                presents = s_data['is_present'].sum()
                absents = len(s_data) - presents
                fig, ax = plt.subplots(figsize=(4, 4))
                ax.pie([presents, absents], labels=['Present', 'Absent'], autopct='%1.1f%%', 
                       colors=['#2ecc71', '#e74c3c'], startangle=90)
                ax.axis('equal')
                st.pyplot(fig)
                
            with col2:
                # Attendance by Day of Week
                # Map day_of_week to names
                day_names = {0: 'Mon', 1: 'Tue', 2: 'Wed', 3: 'Thu', 4: 'Fri', 5: 'Sat', 6: 'Sun'}
                s_data['Day Name'] = s_data['day_of_week'].map(day_names)
                attendance_by_day = s_data.groupby('Day Name')['is_present'].mean() * 100
                
                # Reorder days
                order = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
                attendance_by_day = attendance_by_day.reindex(order).dropna()
                
                fig, ax = plt.subplots(figsize=(6, 4))
                sns.barplot(x=attendance_by_day.index, y=attendance_by_day.values, ax=ax, palette="viridis")
                ax.set_ylabel("Attendance Rate (%)")
                ax.set_title("Attendance Rate by Day of Week")
                ax.set_ylim(0, 100)
                st.pyplot(fig)
            
            # Line chart for rolling attendance
            st.subheader("Rolling 7-Day Attendance Rate Over Time")
            fig, ax = plt.subplots(figsize=(10, 3))
            ax.plot(s_data['Date'], s_data['rolling_7_day_rate'] * 100, marker='o', linestyle='-', color='b')
            ax.set_ylabel("Rate (%)")
            ax.grid(True, alpha=0.3)
            # Format x-axis
            plt.xticks(rotation=45)
            st.pyplot(fig)

if __name__ == "__main__":
    main()
