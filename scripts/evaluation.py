import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def evaluate_model(model, X_test, y_test, X_train, y_train):
    model.eval()
    
    # Convert DataFrame to numpy array if needed
    if isinstance(X_test, pd.DataFrame):
        X_test = X_test.values
    if isinstance(X_train, pd.DataFrame):
        X_train = X_train.values
    
    # Convert to torch tensors
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32).view(-1, 1)
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).view(-1, 1)
    
    with torch.no_grad():
        y_train_pred = model(X_train_tensor).numpy()
        y_test_pred = model(X_test_tensor).numpy()

    rmse_train = np.sqrt(mean_squared_error(y_train, y_train_pred))
    mae_train = mean_absolute_error(y_train, y_train_pred)
    r2_train = r2_score(y_train, y_train_pred)

    rmse_test = np.sqrt(mean_squared_error(y_test, y_test_pred))
    mae_test = mean_absolute_error(y_test, y_test_pred)
    r2_test = r2_score(y_test, y_test_pred)

    print(f'Training RMSE: {rmse_train:.4f}, MAE: {mae_train:.4f}, R²: {r2_train:.4f}')
    print(f'Testing RMSE: {rmse_test:.4f}, MAE: {mae_test:.4f}, R²: {r2_test:.4f}')

    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_test_pred, alpha=0.6)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.title('Actual vs. Predicted Values')
    
    # Ensure the 'results' directory exists
    os.makedirs('results', exist_ok=True)
    plt.savefig(os.path.join('results', 'actual_vs_predicted.png'))
    plt.close()

def permutation_importance(model, X_test, y_test, metric=mean_squared_error):
    X_test_copy = X_test.copy()  # Make a copy of X_test to avoid modifying the original data
    baseline_score = metric(y_test, model(torch.tensor(X_test, dtype=torch.float32)).detach().numpy())
    importances = []
    for col in range(X_test_copy.shape[1]):
        # Save the original feature column
        original_feature = X_test_copy[:, col].copy()
        
        # Shuffle the feature column
        np.random.shuffle(X_test_copy[:, col])
        
        # Recalculate the score with the shuffled feature
        shuffled_score = metric(y_test, model(torch.tensor(X_test_copy, dtype=torch.float32)).detach().numpy())
        
        # Calculate the importance as the difference in scores
        importance = shuffled_score - baseline_score
        
        # Restore the original feature column
        X_test_copy[:, col] = original_feature
        
        importances.append(importance)
        
    return np.array(importances)

def plot_feature_importances(model, X_test, y_test, feature_names, results_path):
    importances = permutation_importance(model, X_test, y_test)
    feature_importances = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
    feature_importances = feature_importances.sort_values(by='Importance', ascending=False)
    
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Importance', y='Feature', data=feature_importances)
    plt.title('Feature Importances')
    plt.savefig(os.path.join(results_path, 'feature_importances.png'))
    plt.close()

def plot_correlation_matrix(df, columns, results_path):
    correlation_matrix = df[columns].corr()
    
    plt.figure(figsize=(12, 14))
    sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap="coolwarm")
    plt.title('Correlation Matrix of Features and Target')
    plt.savefig(os.path.join(results_path, 'correlation_matrix.png'))
    plt.close()

def plot_distribution(y_test, y_test_pred, results_path):
    # Plot distribution curve for actual vs predicted values
    plt.figure(figsize=(10, 6))
    sns.kdeplot(y_test, label='Actual', color='red', fill=True)
    sns.kdeplot(y_test_pred, label='Predicted', color='blue', fill=True)
    plt.title('Distribution Curve: Actual vs Predicted Fantasy Points/Game')
    plt.xlabel('Fantasy Points/Game')
    plt.ylabel('Density')
    plt.legend()
    plt.savefig(os.path.join(results_path, 'distribution_curve.png'))
    plt.close()

    # Calculate the histogram intersection metric
    hist_actual, bins = np.histogram(y_test, bins=30, density=True)
    hist_pred, _ = np.histogram(y_test_pred, bins=bins, density=True)
    overlap = np.sum(np.minimum(hist_actual, hist_pred) * np.diff(bins))

    print(f"Distribution Overlap: {overlap:.4f}")

    return overlap