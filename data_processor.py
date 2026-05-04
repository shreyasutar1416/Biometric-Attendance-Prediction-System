import pandas as pd
import numpy as np
import os

def load_and_preprocess_data(file_path):
    print(f"Loading data from {file_path}...")
    df = pd.read_csv(file_path)
    
    # Identify date columns
    date_columns = [col for col in df.columns if col.startswith('2026-')]
    id_vars = ['student_session_id', 'Admission No', 'Roll Number', 'Name', 'm-Y']
    
    # Melt dataframe to long format
    print("Reshaping data to long format...")
    df_long = pd.melt(df, id_vars=id_vars, value_vars=date_columns, 
                      var_name='Date', value_name='Status')
    
    # Convert Date column to datetime
    df_long['Date'] = pd.to_datetime(df_long['Date'])
    
    # Sort by student and date
    df_long = df_long.sort_values(by=['student_session_id', 'Date']).reset_index(drop=True)
    
    # Map numeric status codes to binary: 4 -> 1 (Present), 1 -> 0 (Absent)
    # Assuming 4 is present because it's more frequent
    df_long['is_present'] = df_long['Status'].map({4: 1, 1: 0})
    
    # Drop rows where Status is NaN or not 4/1
    df_long = df_long.dropna(subset=['is_present'])
    df_long['is_present'] = df_long['is_present'].astype(int)
    
    return df_long

def extract_features(df):
    print("Extracting features...")
    # Date-based features
    df['day_of_week'] = df['Date'].dt.dayofweek
    df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
    
    # Previous day attendance
    df['prev_day_attendance'] = df.groupby('student_session_id')['is_present'].shift(1)
    
    # Rolling 7-day attendance rate
    # Using shift(1) to avoid data leakage (current day shouldn't be in the rolling rate)
    df['rolling_7_day_rate'] = df.groupby('student_session_id')['is_present'].transform(
        lambda x: x.shift(1).rolling(window=7, min_periods=1).mean()
    )
    
    # Total monthly absences up to the current day
    # Cumulative sum of absents (0)
    df['is_absent'] = 1 - df['is_present']
    df['cumulative_absences'] = df.groupby('student_session_id')['is_absent'].transform(
        lambda x: x.shift(1).cumsum()
    )
    df.drop(columns=['is_absent'], inplace=True)
    
    # Drop rows with NaN from shifting (first day of the month)
    # Or fill them with defaults
    df['prev_day_attendance'] = df['prev_day_attendance'].fillna(1) # Assume present by default
    df['rolling_7_day_rate'] = df['rolling_7_day_rate'].fillna(1.0)
    df['cumulative_absences'] = df['cumulative_absences'].fillna(0)
    
    return df

def main():
    file_path = r"c:\Users\HP\Desktop\biometric attendance prediction\data\student_details_on_20260504 (2).csv"
    output_path = r"c:\Users\HP\Desktop\biometric attendance prediction\data\processed_attendance.csv"
    
    # Ensure data directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    if not os.path.exists(file_path):
        print(f"Error: Could not find data file at {file_path}")
        return
        
    df = load_and_preprocess_data(file_path)
    df = extract_features(df)
    
    print(f"Saving processed data to {output_path}...")
    df.to_csv(output_path, index=False)
    print("Data processing complete!")
    print(df.head())

if __name__ == "__main__":
    main()
