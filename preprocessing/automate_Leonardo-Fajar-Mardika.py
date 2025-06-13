import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
import os

def load_data(filepath):
    df = pd.read_csv(filepath)
    return df

def load_data(path):
    df = pd.read_csv(path)
    return df

def preprocess_data(df):
    # Buang kolom yang tidak berguna
    df= df.drop(columns=['student_id'])
    # Menghapus baris yang memiliki nilai NaN
    df=df.dropna()
    # Menghapus duplikat
    df = df.drop_duplicates()

    

    # Scaling dengan MinMaxScaler
    scaler = MinMaxScaler()
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

    df = outlier(df,numeric_cols)

    # Label Encoding untuk kolom kategorikal
    categorical_cols = df.select_dtypes(include=['object']).columns
    label_encoders = {}
    for col in categorical_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        label_encoders[col] = le

    

    return df

def outlier(df, numerical_cols, threshold=1.5):    
    df_winsorized = df.copy()
    
    # Winsorizing untuk numerical_cols biasa
    Q1 = df[numerical_cols].quantile(0.25)
    Q3 = df[numerical_cols].quantile(0.75)
    IQR = Q3 - Q1

    lower_bound = Q1 - threshold * IQR
    upper_bound = Q3 + threshold * IQR

    for col in numerical_cols:
        df_winsorized[col] = np.where(df[col] < lower_bound[col], lower_bound[col], df[col])
        df_winsorized[col] = np.where(df[col] > upper_bound[col], upper_bound[col], df_winsorized[col])
    
    return df_winsorized

def save_data(df, output_path):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)

def main():
    input_path = "./student_habits_performance_raw.csv"
    output_path = "./preprocessing/student_habits_performance_preprocessed.csv"

    df = load_data(input_path)
    df_clean = preprocess_data(df)
    save_data(df_clean, output_path)

if __name__ == "__main__":
    main()
