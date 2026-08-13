"""Exploratory script for ML Lab 10 (MLP for XOR).

Runs all MLP architectures across three distinct libraries (Keras, PyTorch, TensorFlow Low-Level),
and iteratively tests Learning Rates, Topologies, and Activations, printing exact numerical
results and loss decay values for inclusion in markdown interpretation cells.
"""
import numpy as np
import time

# Suppress warnings for clean output
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("LAB 10: LEARNING THE XOR BOOLEAN FUNCTION USING AN MLP")
print("="*80)

# Dataset
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float32)
y = np.array([[0], [1], [1], [0]], dtype=np.float32)

print("Inputs (X):\n", X)
print("Targets (y):\n", y)
print("-"*80)

SEED = 42

# ---------------------------------------------------------
# Keras Implementation Baseline
# ---------------------------------------------------------
print("\n[1] KERAS (HIGH-LEVEL API) BASELINE")
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

tf.random.set_seed(SEED)
np.random.seed(SEED)

keras_base = Sequential([
    Dense(8, input_dim=2, activation='tanh'),
    Dense(1, activation='sigmoid')
])
keras_base.compile(loss='binary_crossentropy', optimizer=keras.optimizers.Adam(learning_rate=0.05), metrics=['accuracy'])
keras_base.fit(X, y, epochs=200, verbose=0)
k_loss, k_acc = keras_base.evaluate(X, y, verbose=0)
print(f"Keras Baseline Accuracy: {k_acc*100:.2f}% | Loss: {k_loss:.4f}")

# ---------------------------------------------------------
# PyTorch Implementation Baseline
# ---------------------------------------------------------
print("\n[2] PYTORCH IMPLEMENTATION BASELINE")
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

pt_base = XOR_MLP()
optimizer = optim.Adam(pt_base.parameters(), lr=0.05)
criterion = nn.BCELoss()

pt_losses = []
for epoch in range(200):
    optimizer.zero_grad()
    loss = criterion(pt_base(X_pt), y_pt)
    loss.backward()
    optimizer.step()
    pt_losses.append(loss.item())
with torch.no_grad():
    pt_acc = ((pt_base(X_pt) > 0.5).float() == y_pt).float().mean().item()
print(f"PyTorch Baseline Accuracy: {pt_acc*100:.2f}% | Loss: {pt_losses[-1]:.4f}")

# ---------------------------------------------------------
# TensorFlow Low-Level API Baseline
# ---------------------------------------------------------
print("\n[3] TENSORFLOW (LOW-LEVEL API) BASELINE")
tf.random.set_seed(SEED)

W1 = tf.Variable(tf.random.normal([2, 8], stddev=0.1, seed=SEED))
b1 = tf.Variable(tf.zeros([8]))
W2 = tf.Variable(tf.random.normal([8, 1], stddev=0.1, seed=SEED))
b2 = tf.Variable(tf.zeros([1]))

def tf_forward(x):
    return tf.math.sigmoid(tf.matmul(tf.math.tanh(tf.matmul(x, W1) + b1), W2) + b2)

tf_opt = tf.optimizers.Adam(learning_rate=0.05)

tf_losses = []
for epoch in range(200):
    with tf.GradientTape() as tape:
        loss = tf.reduce_mean(tf.keras.losses.binary_crossentropy(y, tf_forward(X)))
    grads = tape.gradient(loss, [W1, b1, W2, b2])
    tf_opt.apply_gradients(zip(grads, [W1, b1, W2, b2]))
    tf_losses.append(loss.numpy())

tf_acc = np.mean((tf_forward(X).numpy() > 0.5) == y)
print(f"TF Low-Level Baseline Accuracy: {tf_acc*100:.2f}% | Loss: {tf_losses[-1]:.4f}")

print("="*80)
# ---------------------------------------------------------
# SLI 10.1: Effect of Learning Rate (Keras)
# ---------------------------------------------------------
print("\n[SLI 10.1] EFFECT OF LEARNING RATE (Adam, 200 Epochs)")
lr_list = [0.001, 0.01, 0.05, 0.1, 0.5]
for lr in lr_list:
    tf.random.set_seed(SEED)
    model = Sequential([Dense(8, input_dim=2, activation='tanh'), Dense(1, activation='sigmoid')])
    model.compile(loss='binary_crossentropy', optimizer=keras.optimizers.Adam(learning_rate=lr), metrics=['accuracy'])
    model.fit(X, y, epochs=200, verbose=0)
    l, a = model.evaluate(X, y, verbose=0)
    print(f"  LR = {lr:<5} | Final Accuracy: {a*100:>6.2f}% | Final Loss: {l:.4f}")

# ---------------------------------------------------------
# SLI 10.2: Effect of Hidden Neurons (PyTorch)
# ---------------------------------------------------------
print("\n[SLI 10.2] EFFECT OF HIDDEN NEURONS (LR=0.05, 200 Epochs)")
for neurons in [2, 4, 8, 16]:
    torch.manual_seed(SEED)
    model = XOR_MLP(hidden_neurons=neurons)
    optimizer = optim.Adam(model.parameters(), lr=0.05)
    criterion = nn.BCELoss()
    for epoch in range(200):
        optimizer.zero_grad()
        loss = criterion(model(X_pt), y_pt)
        loss.backward()
        optimizer.step()
    with torch.no_grad():
        a = ((model(X_pt) > 0.5).float() == y_pt).float().mean().item()
    print(f"  Neurons = {neurons:<2} | Final Accuracy: {a*100:>6.2f}% | Final Loss: {loss.item():.4f}")

# ---------------------------------------------------------
# SLI 10.3: Effect of Activation Functions (Keras)
# ---------------------------------------------------------
print("\n[SLI 10.3] EFFECT OF ACTIVATION FUNCTIONS (LR=0.05, 200 Epochs)")
for act in ['relu', 'sigmoid', 'tanh']:
    tf.random.set_seed(SEED)
    model = Sequential([Dense(8, input_dim=2, activation=act), Dense(1, activation='sigmoid')])
    model.compile(loss='binary_crossentropy', optimizer=keras.optimizers.Adam(learning_rate=0.05), metrics=['accuracy'])
    model.fit(X, y, epochs=200, verbose=0)
    l, a = model.evaluate(X, y, verbose=0)
    print(f"  Activation = {act:<7} | Final Accuracy: {a*100:>6.2f}% | Final Loss: {l:.4f}")
print("="*80)
