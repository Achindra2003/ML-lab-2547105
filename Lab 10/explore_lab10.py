"""Exploratory script for ML Lab 10 (MLP for XOR).

Executes all baseline models (Keras, PyTorch, TF Low-Level),
hyperparameter sweeps (Learning Rate, Activations, Topologies),
Pure NumPy Scratch Analytical Backprop, and Latent Hidden Space Transformations.
Prints exact numerical metrics for inclusion in cell interpretations.
"""
import numpy as np
import time
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import warnings
warnings.filterwarnings('ignore')

SEED = 42
np.random.seed(SEED)

print("="*80)
print("LAB 10: LEARNING THE XOR BOOLEAN FUNCTION USING AN MLP")
print("="*80)

# XOR Dataset
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float32)
y = np.array([[0], [1], [1], [0]], dtype=np.float32)

print("Inputs (X):\n", X)
print("Targets (y):\n", y)
print("-"*80)

# ---------------------------------------------------------
# [1] Keras Implementation
# ---------------------------------------------------------
print("\n[1] KERAS BASELINE")
try:
    import tensorflow as tf
    from tensorflow import keras
    tf.random.set_seed(SEED)
    k_model = keras.Sequential([
        keras.layers.Dense(8, input_dim=2, activation='tanh'),
        keras.layers.Dense(1, activation='sigmoid')
    ])
    k_model.compile(loss='binary_crossentropy', optimizer=keras.optimizers.Adam(learning_rate=0.05), metrics=['accuracy'])
    start = time.time()
    k_model.fit(X, y, epochs=200, verbose=0)
    k_time = time.time() - start
    k_loss, k_acc = k_model.evaluate(X, y, verbose=0)
    k_preds = k_model.predict(X, verbose=0)
    print(f"Keras Accuracy: {k_acc*100:.2f}% | Loss: {k_loss:.4f} | Latency: {k_time:.4f}s")
    print("Keras Predictions:\n", np.round(k_preds, 4))
except Exception as e:
    print(f"Keras execution skipped (missing tensorflow): {e}")

# ---------------------------------------------------------
# [2] PyTorch Implementation
# ---------------------------------------------------------
print("\n[2] PYTORCH BASELINE")
import torch
import torch.nn as nn
import torch.optim as optim

torch.manual_seed(SEED)
X_pt = torch.tensor(X)
y_pt = torch.tensor(y)

class XOR_MLP(nn.Module):
    def __init__(self, hidden_neurons=8):
        super(XOR_MLP, self).__init__()
        self.hidden = nn.Linear(2, hidden_neurons)
        self.output = nn.Linear(hidden_neurons, 1)
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        return self.sigmoid(self.output(self.tanh(self.hidden(x))))

pt_model = XOR_MLP(hidden_neurons=8)
optimizer_pt = optim.Adam(pt_model.parameters(), lr=0.05)
criterion_pt = nn.BCELoss()

start_pt = time.time()
pt_losses = []
for epoch in range(200):
    optimizer_pt.zero_grad()
    l = criterion_pt(pt_model(X_pt), y_pt)
    l.backward()
    optimizer_pt.step()
    pt_losses.append(l.item())
pt_time = time.time() - start_pt

with torch.no_grad():
    pt_preds = pt_model(X_pt)
    pt_acc = ((pt_preds > 0.5).float() == y_pt).float().mean().item()

print(f"PyTorch Accuracy: {pt_acc*100:.2f}% | Loss: {pt_losses[-1]:.4f} | Latency: {pt_time:.4f}s")
print("PyTorch Predictions:\n", np.round(pt_preds.numpy(), 4))

# ---------------------------------------------------------
# [3] SLI 10.1: Pure NumPy Analytical Backprop from Scratch
# ---------------------------------------------------------
print("\n[3] SLI 10.1: PURE NUMPY ANALYTICAL BACKPROP FROM SCRATCH (NO LIBRARIES)")
np.random.seed(SEED)
W1_np = np.random.randn(2, 8).astype(np.float32) * 0.5
b1_np = np.zeros((1, 8), dtype=np.float32)
W2_np = np.random.randn(8, 1).astype(np.float32) * 0.5
b2_np = np.zeros((1, 1), dtype=np.float32)

lr_np = 0.1
scratch_losses = []
for epoch in range(500):
    # Forward
    Z1 = np.dot(X, W1_np) + b1_np
    A1 = np.tanh(Z1)
    Z2 = np.dot(A1, W2_np) + b2_np
    y_hat = 1.0 / (1.0 + np.exp(-Z2))
    
    # Loss
    l_sc = -np.mean(y * np.log(y_hat + 1e-8) + (1 - y) * np.log(1 - y_hat + 1e-8))
    scratch_losses.append(l_sc)
    
    # Backprop
    dZ2 = y_hat - y
    dW2 = np.dot(A1.T, dZ2) / 4.0
    db2 = np.sum(dZ2, axis=0, keepdims=True) / 4.0
    
    dA1 = np.dot(dZ2, W2_np.T)
    dZ1 = dA1 * (1.0 - A1**2)
    dW1 = np.dot(X.T, dZ1) / 4.0
    db1 = np.sum(dZ1, axis=0, keepdims=True) / 4.0
    
    # Update
    W2_np -= lr_np * dW2
    b2_np -= lr_np * db2
    W1_np -= lr_np * dW1
    b1_np -= lr_np * db1

acc_sc = np.mean((y_hat > 0.5) == y)
print(f"NumPy Scratch Accuracy: {acc_sc*100:.2f}% | Loss: {scratch_losses[-1]:.4f}")
print("NumPy Scratch Predictions:\n", np.round(y_hat, 4))

# ---------------------------------------------------------
# [4] SLI 10.2: Latent Hidden Space Transformation
# ---------------------------------------------------------
print("\n[4] SLI 10.2: LATENT HIDDEN SPACE FEATURE TRANSFORMATIONS")
with torch.no_grad():
    h_activations = pt_model.tanh(pt_model.hidden(X_pt)).numpy()

print("Extracted Hidden Space Representation Vector Coordinates (h1, h2, ... h8):\n", np.round(h_activations, 4))
print("First 2 Hidden Dimensions for Cartesian Plotting:")
for i, txt in enumerate(['(0,0)', '(0,1)', '(1,0)', '(1,1)']):
    print(f"  Input {txt} -> Target y={int(y[i][0])} -> Hidden Coordinates: ({h_activations[i, 0]:.4f}, {h_activations[i, 1]:.4f})")

print("="*80)
