import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from scripts.training import train_model
from scripts.evaluation import evaluate_model, plot_correlation_matrix, plot_distribution
from scripts.prediction import predict, save_predictions
from scripts.feature_engineering import preprocess_data, add_rolling_averages, shift_target, add_season_flags
from scripts.model import FantasyFootballLSTM
import torch
import numpy as np

# Define paths
data_path = 'data/cleaned_fantasy_football_data.xlsx'
results_path = 'results'
model_path = 'models/trained_model.pth'
predictions_path = os.path.join(results_path, 'predictions_2024.xlsx')

# Load and preprocess the data
df = pd.read_excel(data_path)
df = preprocess_data(df)
df = add_rolling_averages(df)
df = add_season_flags(df)

# Save 2023 data before shifting the target
df_2023 = df[df['Year'] == 2023].copy()
player_names_2023 = df_2023['Player'].reset_index(drop=True)
df_2023 = df_2023.drop(columns=['Player'])

# Apply shift_target to remove rows with NaN target values
df = shift_target(df)

# Reset the index to ensure sequential indexing
df = df.reset_index(drop=True)

# Define features and target
feature_names = ['Year', 'Age', 'G', 'Tgt', 'Rec', 'RecYds', 'RecTD', 'TD/G', 'RecYds/G',
                 'FantPt/G', 'Tgt/G', 'FantPt/GLast2Y', 'FantPt/GLast3Y',
                 'Tgt/GLast2Y', 'Tgt/GLast3Y', 'RecYds/GLast2Y', 'RecYds/GLast3Y',
                 '#ofY']
target = 'NextYearFantPt/G'

# Split the data into training and test sets
X = df[feature_names]
y = df[target]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Filter data for 2023 to be used for prediction later
X_2023 = df_2023[feature_names]

# Standardize the data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
X_2023 = scaler.transform(X_2023)

# Ensure X_train, X_test, and X_2023 are correctly shaped for LSTM
X_train = X_train.reshape(X_train.shape[0], 1, X_train.shape[1])
X_test = X_test.reshape(X_test.shape[0], 1, X_test.shape[1])
X_2023 = X_2023.reshape(X_2023.shape[0], 1, X_2023.shape[1])

# Train the model
input_dim = X_train.shape[2]
hidden_dim = 128  # Set hidden dimension
num_layers = 2  # Set number of LSTM layers
model = FantasyFootballLSTM(input_dim, hidden_dim, num_layers)
train_model(model, X_train, y_train, epochs=1000, learning_rate=0.001)

# Save the model
torch.save(model.state_dict(), model_path)

# Evaluate the model
evaluate_model(model, X_test, y_test, X_train, y_train)

# Plot correlation matrix
plot_correlation_matrix(df, feature_names + [target], results_path)

# Make predictions for the test data
y_test_pred = predict(model, X_test)

# Make predictions for 2024 data
predictions_2023 = predict(model, X_2023)
predictions_df = pd.DataFrame(predictions_2023, columns=['Predicted_FantPt/G'])

# Ensure player names are included in predictions
predictions_df['Player'] = player_names_2023
save_predictions(predictions_df, predictions_path)
print(f"Predictions for 2024 saved to '{predictions_path}'")

# Plot distribution curve for actual vs predicted values
plot_distribution(y_test, y_test_pred, results_path)

print(f"Distribution curve plot saved to '{results_path}/distribution_curve.png'")
