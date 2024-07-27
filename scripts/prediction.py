import torch
import numpy as np
import pandas as pd

def save_predictions(predictions_df, file_path):
    # Create a new DataFrame with the desired columns
    new_df = pd.DataFrame({
        'Player': predictions_df['Player'],
        'Predicted_FantPt/G': predictions_df['Predicted_FantPt/G']
    })

    # Save the new DataFrame to an Excel file
    new_df.to_excel(file_path, index=False)

def predict(model, X):
    model.eval()
    with torch.no_grad():
        predictions = model(torch.tensor(X, dtype=torch.float32)).numpy()
    return predictions