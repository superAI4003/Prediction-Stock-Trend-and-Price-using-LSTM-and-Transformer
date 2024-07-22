import pandas as pd
import os

path_datasets_raw = "./datasets_raw/"
path_datasets_processed = "./datasets/"

# Ensure the processed datasets directory exists
os.makedirs(path_datasets_processed, exist_ok=True)

for filename in os.listdir(path_datasets_raw):
    if filename.endswith(".csv"):
        df = pd.read_csv(os.path.join(path_datasets_raw, filename))
        df['rolling_mean_14'] = df['close'].shift(1).rolling(window=14).mean()
        
        # Add the 'predict' column based on the condition
        df['predict'] = df.apply(
            lambda row: 'high' if pd.notna(row['rolling_mean_14']) and row['close'] > row['rolling_mean_14'] else (' ' if pd.isna(row['rolling_mean_14']) else 'low'),
            axis=1
        )
        
        # Drop the temporary 'rolling_mean_14' column
        df.drop(columns=['rolling_mean_14'], inplace=True)
        
        # Remove rows with ' ' in any column
        df = df[~df.apply(lambda x: x.str.contains(' ').any(), axis=1)]
        
        # Save the processed file
        df.to_csv(os.path.join(path_datasets_processed, filename), index=False)

        print("Preprocessed "+filename+".")