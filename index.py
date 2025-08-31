import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

# --- Step 1: Load and explore the dataset ---
print("Loading calories dataset...")
try:
    df = pd.read_csv("calories.csv")
    if 'User_ID' in df.columns:
        try:
            df_exercise = pd.read_csv("exercise.csv")
            df = pd.merge(df, df_exercise, on='User_ID')
            print("Successfully merged with exercise.csv")
        except FileNotFoundError:
            print("exercise.csv not found. Proceeding with weak features.")
            pass
except FileNotFoundError:
    print("calories.csv not found. Please ensure the file is in the correct directory.")
    exit()

# --- Step 2: Select input features and target variable ---
if 'Duration' in df.columns and 'Heart_Rate' in df.columns:
    print("Using powerful features: Duration and Heart_Rate included.")
    input_features = ['Age', 'Height', 'Weight', 'Gender', 'Duration', 'Heart_Rate']
else:
    print("Using weak features: Age, Height, Weight, Gender.")
    input_features = ['Age', 'Height', 'Weight', 'Gender']

target_feature = 'Calories'
X_raw = df[input_features].copy()
y = df[target_feature].values.reshape(-1, 1)

# --- Step 3: Handle categorical features (Gender) ---
X_raw['Gender'] = X_raw['Gender'].apply(lambda x: 1 if x == 'male' else 0)

# --- Step 4: Feature scaling (Standardization) ---
def fit_standard_scaler(data):
    mean = np.mean(data, axis=0)
    std = np.std(data, axis=0)
    std[std == 0] = 1.0
    return mean, std

def transform_standard_scaler(data, mean, std):
    return (data - mean) / std

def inverse_transform_standard_scaler(scaled_data, mean, std):
    return scaled_data * std + mean

X_mean, X_std = fit_standard_scaler(X_raw.values)
y_mean, y_std = fit_standard_scaler(y)
X_scaled = transform_standard_scaler(X_raw.values, X_mean, X_std)
y_scaled = transform_standard_scaler(y, y_mean, y_std)

# --- Step 5: Split data ---
train_size = int(0.8 * len(X_scaled))
X_train, X_test = X_scaled[:train_size], X_scaled[train_size:]
y_train, y_test = y_scaled[:train_size], y_scaled[train_size:]
print(f"\nData split into {len(X_train)} training and {len(X_test)} testing samples.")

# --- Step 6: Initialize NN parameters ---
np.random.seed(42)
input_dim = X_train.shape[1]
hidden_dim_1 = 64
hidden_dim_2 = 32
output_dim = 1

W1 = np.random.randn(input_dim, hidden_dim_1) * np.sqrt(2.0 / input_dim)
b1 = np.zeros((1, hidden_dim_1))
W2 = np.random.randn(hidden_dim_1, hidden_dim_2) * np.sqrt(2.0 / hidden_dim_1)
b2 = np.zeros((1, hidden_dim_2))
W3 = np.random.randn(hidden_dim_2, output_dim) * np.sqrt(1.0 / hidden_dim_2)
b3 = np.zeros((1, output_dim))

print(f"\nUpgraded Neural Network Architecture: {input_dim} -> {hidden_dim_1} -> {hidden_dim_2} -> {output_dim}")

# --- Step 7: Activations & Loss ---
def leaky_relu(x, alpha=0.01):
    return np.where(x > 0, x, x * alpha)

def leaky_relu_derivative(x, alpha=0.01):
    return np.where(x > 0, 1, alpha)

def linear(x):
    return x

def mse_loss(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

def mse_derivative(y_true, y_pred):
    return 2 * (y_pred - y_true) / len(y_true)

# --- Step 8: Training loop with Adam + Dropout ---
epochs = 2000
learning_rate = 0.001
batch_size = 64
dropout_rate = 0.2

beta1, beta2, epsilon = 0.9, 0.999, 1e-8
m_W1, v_W1, m_b1, v_b1 = 0, 0, 0, 0
m_W2, v_W2, m_b2, v_b2 = 0, 0, 0, 0
m_W3, v_W3, m_b3, v_b3 = 0, 0, 0, 0

# --- Checkpoint setup ---
os.makedirs("checkpoints", exist_ok=True)
best_loss = float("inf")

print(f"\nStarting training with Adam Optimizer...")
print(f"Epochs: {epochs}, Learning Rate: {learning_rate}, Dropout: {dropout_rate*100}%")

for epoch in range(epochs):
    permutation = np.random.permutation(len(X_train))
    X_train_shuffled, y_train_shuffled = X_train[permutation], y_train[permutation]

    for i in range(0, len(X_train), batch_size):
        batch_X = X_train_shuffled[i:i+batch_size]
        batch_y = y_train_shuffled[i:i+batch_size]

        # Forward pass
        z1 = np.dot(batch_X, W1) + b1
        a1 = leaky_relu(z1)
        dropout_mask1 = (np.random.rand(*a1.shape) > dropout_rate) / (1.0 - dropout_rate)
        a1 *= dropout_mask1

        z2 = np.dot(a1, W2) + b2
        a2 = leaky_relu(z2)
        dropout_mask2 = (np.random.rand(*a2.shape) > dropout_rate) / (1.0 - dropout_rate)
        a2 *= dropout_mask2

        z3 = np.dot(a2, W3) + b3
        y_pred = linear(z3)

        # Backward pass
        d_loss = mse_derivative(batch_y, y_pred)
        d_z3 = d_loss
        dW3 = np.dot(a2.T, d_z3)
        db3 = np.sum(d_z3, axis=0)

        d_a2 = np.dot(d_z3, W3.T)
        d_a2 *= dropout_mask2
        d_z2 = d_a2 * leaky_relu_derivative(z2)
        dW2 = np.dot(a1.T, d_z2)
        db2 = np.sum(d_z2, axis=0)

        d_a1 = np.dot(d_z2, W2.T)
        d_a1 *= dropout_mask1
        d_z1 = d_a1 * leaky_relu_derivative(z1)
        dW1 = np.dot(batch_X.T, d_z1)
        db1 = np.sum(d_z1, axis=0)

        # Adam updates
        t = epoch + 1
        m_W1 = beta1 * m_W1 + (1 - beta1) * dW1
        v_W1 = beta2 * v_W1 + (1 - beta2) * (dW1**2)
        W1 -= learning_rate * (m_W1 / (1 - beta1**t)) / (np.sqrt(v_W1 / (1 - beta2**t)) + epsilon)

        m_b1 = beta1 * m_b1 + (1 - beta1) * db1
        v_b1 = beta2 * v_b1 + (1 - beta2) * (db1**2)
        b1 -= learning_rate * (m_b1 / (1 - beta1**t)) / (np.sqrt(v_b1 / (1 - beta2**t)) + epsilon)

        m_W2 = beta1 * m_W2 + (1 - beta1) * dW2
        v_W2 = beta2 * v_W2 + (1 - beta2) * (dW2**2)
        W2 -= learning_rate * (m_W2 / (1 - beta1**t)) / (np.sqrt(v_W2 / (1 - beta2**t)) + epsilon)

        m_b2 = beta1 * m_b2 + (1 - beta1) * db2
        v_b2 = beta2 * v_b2 + (1 - beta2) * (db2**2)
        b2 -= learning_rate * (m_b2 / (1 - beta1**t)) / (np.sqrt(v_b2 / (1 - beta2**t)) + epsilon)

        m_W3 = beta1 * m_W3 + (1 - beta1) * dW3
        v_W3 = beta2 * v_W3 + (1 - beta2) * (dW3**2)
        W3 -= learning_rate * (m_W3 / (1 - beta1**t)) / (np.sqrt(v_W3 / (1 - beta2**t)) + epsilon)

        m_b3 = beta1 * m_b3 + (1 - beta1) * db3
        v_b3 = beta2 * v_b3 + (1 - beta2) * (db3**2)
        b3 -= learning_rate * (m_b3 / (1 - beta1**t)) / (np.sqrt(v_b3 / (1 - beta2**t)) + epsilon)

    # --- Evaluation + Checkpoint ---
    if epoch % 200 == 0 or epoch == epochs - 1:
        z1_test = np.dot(X_test, W1) + b1
        a1_test = leaky_relu(z1_test)
        z2_test = np.dot(a1_test, W2) + b2
        a2_test = leaky_relu(z2_test)
        z3_test = np.dot(a2_test, W3) + b3
        y_pred_scaled = linear(z3_test)
        y_pred_original = inverse_transform_standard_scaler(y_pred_scaled, y_mean, y_std)
        y_test_original = inverse_transform_standard_scaler(y_test, y_mean, y_std)
        test_loss = mse_loss(y_test_original, y_pred_original)

        print(f"Epoch {epoch:4d} | Test Loss (MSE): {test_loss:,.2f}")

        # Save best checkpoint
        if test_loss < best_loss:
            best_loss = test_loss
            checkpoint_path = f"checkpoints/checkpoint_epoch_{epoch}.npz"
            np.savez(checkpoint_path, W1=W1, b1=b1, W2=W2, b2=b2, W3=W3, b3=b3,
                     X_mean=X_mean, X_std=X_std, y_mean=y_mean, y_std=y_std)
            print(f"✅ Saved new best checkpoint: {checkpoint_path} (loss={best_loss:.2f})")

# --- Step 9: Final Model Evaluation ---
z1_final = np.dot(X_test, W1) + b1
a1_final = leaky_relu(z1_final)
z2_final = np.dot(a1_final, W2) + b2
a2_final = leaky_relu(z2_final)
z3_final = np.dot(a2_final, W3) + b3
y_pred_scaled = linear(z3_final)
y_pred_original = inverse_transform_standard_scaler(y_pred_scaled, y_mean, y_std)
y_test_original = inverse_transform_standard_scaler(y_test, y_mean, y_std)

mse = np.mean((y_test_original - y_pred_original) ** 2)
rmse = np.sqrt(mse)
mae = np.mean(np.abs(y_test_original - y_pred_original))
r2_num = np.sum((y_test_original - y_pred_original) ** 2)
r2_den = np.sum((y_test_original - np.mean(y_test_original)) ** 2)
r2 = 1 - (r2_num / r2_den)

print(f"\n=== FINAL MODEL EVALUATION ===")
print(f"Root Mean Squared Error (RMSE): {rmse:.2f} calories")
print(f"Mean Absolute Error (MAE): {mae:.2f} calories")
print(f"R-squared (R²): {r2:.4f}")

# --- Step 10: Plotting Results ---
plt.figure(figsize=(10, 6))
plt.scatter(y_test_original, y_pred_original, alpha=0.6, s=20, label='Actual vs. Predicted')
plt.plot([y_test_original.min(), y_test_original.max()], [y_test_original.min(), y_test_original.max()], 'r--', lw=2, label='Perfect Prediction')
plt.xlabel('Actual Calories')
plt.ylabel('Predicted Calories')
plt.title('Model Performance on Test Set')
plt.legend()
plt.grid(True)
plt.show()
