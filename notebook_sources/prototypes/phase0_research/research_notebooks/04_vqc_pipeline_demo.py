# Extracted first-party notebook code; execution outputs intentionally omitted.

# Notebook cell
# Install required packages
import subprocess
import sys

packages = ['pennylane', 'pennylane-lightning', 'numpy', 'scipy', 'pandas', 'matplotlib', 'seaborn']
for package in packages:
 try:
 __import__(package.replace('-', '_'))
 except ImportError:
 print(f"Installing {package}...")
 subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q', package])

print("All packages ready.")

# Notebook cell
# Core imports
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import norm
import json
import warnings
from pathlib import Path
import time

warnings.filterwarnings('ignore')

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (14, 6)

# Quantum
PENNYLANE_AVAILABLE = False
try:
 import pennylane as qml
 from pennylane import numpy as qnp
 PENNYLANE_AVAILABLE = True
 print("PennyLane loaded successfully.")
except ImportError:
 print("PennyLane not available. Using NumPy fallback.")

# Machine learning baselines
SKLEARN_AVAILABLE = False
try:
 from sklearn.preprocessing import StandardScaler
 from sklearn.linear_model import Ridge
 from sklearn.ensemble import RandomForestRegressor
 from sklearn.metrics import mean_squared_error, r2_score
 SKLEARN_AVAILABLE = True
 print("scikit-learn loaded successfully.")
except ImportError:
 print("scikit-learn not available. Using manual standardization.")

print("\n" + "="*60)
print("ENVIRONMENT READY")
print(f"PennyLane: {PENNYLANE_AVAILABLE}")
print(f"scikit-learn: {SKLEARN_AVAILABLE}")
print("="*60)

# Notebook cell
np.random.seed(42)

n_samples = 500
n_features = 8

# Initialize arrays
X = np.zeros((n_samples, n_features))
y = np.zeros(n_samples)

# Feature 0: Momentum (AR(1) process)
X[0, 0] = np.random.normal(0, 0.2)
for t in range(1, n_samples):
 X[t, 0] = 0.7 * X[t-1, 0] + 0.3 * np.random.normal(0, 0.2)

# Feature 1: Volatility Regime (GARCH-like)
vol = 0.15
for t in range(n_samples):
 vol = 0.9 * vol + 0.1 * abs(np.random.normal(0, 0.5))
 X[t, 1] = min(vol, 1.0) # Clip to [-1, 1]

# Feature 2: Carry / Value (slow mean-reverting)
X[0, 2] = 0.0
for t in range(1, n_samples):
 X[t, 2] = 0.95 * X[t-1, 2] + 0.05 * np.random.normal(0, 0.3)

# Feature 3: Sentiment (noisy, random walk)
X[0, 3] = 0.0
for t in range(1, n_samples):
 X[t, 3] = X[t-1, 3] + np.random.normal(0, 0.1)
 X[t, 3] = np.clip(X[t, 3], -1, 1)

# Feature 4: Mean Reversion (Z-score of recent returns)
X[0, 4] = np.random.normal(0, 0.2)
for t in range(1, n_samples):
 X[t, 4] = 0.85 * X[t-1, 4] + 0.15 * np.random.normal(0, 0.2)

# Feature 5: Cross-Asset Correlation
X[0, 5] = np.random.normal(0, 0.1)
for t in range(1, n_samples):
 X[t, 5] = 0.92 * X[t-1, 5] + 0.08 * np.random.normal(0, 0.15)

# Feature 6: Technical Indicator (e.g., RSI normalized)
X[0, 6] = np.random.normal(0, 0.15)
for t in range(1, n_samples):
 X[t, 6] = 0.8 * X[t-1, 6] + 0.2 * np.random.normal(0, 0.2)

# Feature 7: Liquidity / Spread
X[0, 7] = 0.1 + np.random.normal(0, 0.05)
for t in range(1, n_samples):
 X[t, 7] = 0.95 * X[t-1, 7] + 0.05 * np.random.normal(0, 0.1)
 X[t, 7] = np.clip(X[t, 7], -1, 1)

# ============================================================================
# Generate returns with KNOWN signal structure
# ============================================================================

# True signal weights (what the VQC should learn)
true_weights = np.array([0.4, -0.3, 0.2, 0.15, 0.25, 0.1, 0.2, -0.1])
true_weights = true_weights / np.linalg.norm(true_weights) # Normalize

# Regime-dependent weights: high vol reduces momentum, increases mean reversion
for t in range(n_samples):
 vol_regime = X[t, 1] # Volatility feature in [0, 1]
 
 # Adjust weights based on volatility regime
 adjusted_weights = true_weights.copy
 adjusted_weights[0] *= (1 - 0.5 * vol_regime) # Reduce momentum weight in high vol
 adjusted_weights[4] *= (1 + 0.3 * vol_regime) # Increase mean reversion weight in high vol
 
 # Normalize
 adjusted_weights = adjusted_weights / np.linalg.norm(adjusted_weights)
 
 # Predicted signal
 predicted_signal = np.dot(X[t], adjusted_weights)
 
 # Return = signal coefficient * predicted signal + noise
 y[t] = 0.03 * predicted_signal + 0.01 * np.random.normal(0, 0.15)

# ============================================================================
# Train/test split
# ============================================================================
split_idx = int(0.7 * n_samples)
X_train, X_test = X[:split_idx], X[split_idx:]
y_train, y_test = y[:split_idx], y[split_idx:]

print(f"\nSynthetic Data Generated")
print(f" Total samples: {n_samples}")
print(f" Features: {n_features}")
print(f" Training set: {len(X_train)} samples")
print(f" Test set: {len(X_test)} samples")
print(f" Return range: [{y.min:.4f}, {y.max:.4f}]")
print(f" Return mean: {y.mean:.6f}, std: {y.std:.6f}")
print(f"\nTrue signal weights (normalized): {true_weights}")

# Notebook cell
# Manual standardization to [-1, 1] for quantum encoding
def standard_scale_to_range(X, X_ref=None, target_range=(-1, 1)):
 """Standard scale to target range, using reference data for fit."""
 if X_ref is None:
 X_ref = X
 
 mean = X_ref.mean(axis=0)
 std = X_ref.std(axis=0)
 std[std == 0] = 1.0 # Avoid division by zero
 
 X_scaled = (X - mean) / std
 # Clip to [-1, 1] to ensure valid quantum angles
 X_scaled = np.clip(X_scaled, target_range[0], target_range[1])
 
 return X_scaled, (mean, std)

# Fit scaler on training data
X_train_scaled, (feature_mean, feature_std) = standard_scale_to_range(X_train)
X_test_scaled, _ = standard_scale_to_range(X_test, X_ref=X_train, target_range=(-1, 1))

print("Feature Preprocessing Complete")
print(f" Training features mean: {X_train_scaled.mean(axis=0)}")
print(f" Training features std: {X_train_scaled.std(axis=0)}")
print(f" Training features range: [{X_train_scaled.min:.3f}, {X_train_scaled.max:.3f}]")

# Notebook cell
# Feature correlation matrix
corr_matrix = np.corrcoef(X_train_scaled.T)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Heatmap
ax = axes[0]
im = ax.imshow(corr_matrix, cmap='RdBu_r', vmin=-1, vmax=1)
ax.set_xticks(range(n_features))
ax.set_yticks(range(n_features))
ax.set_xlabel('Feature Index')
ax.set_ylabel('Feature Index')
ax.set_title('Feature Correlation Matrix')
plt.colorbar(im, ax=ax)

# Feature distributions
ax = axes[1]
for i in range(n_features):
 ax.hist(X_train_scaled[:, i], bins=20, alpha=0.6, label=f'F{i}')
ax.set_xlabel('Normalized Value')
ax.set_ylabel('Count')
ax.set_title('Feature Distributions (Training Set)')
ax.legend(fontsize=8)

plt.tight_layout
plt.show

print("\nFeature Correlation (absolute)")
print(f" Max correlation: {np.max(np.abs(corr_matrix - np.eye(n_features))):.3f}")
print(f" Mean correlation: {np.mean(np.abs(corr_matrix - np.eye(n_features))):.3f}")

# Notebook cell
if PENNYLANE_AVAILABLE:
 # PennyLane VQC Circuit
 dev = qml.device('default.qubit', wires=8)
 
 @qml.qnode(dev)
 def vqc_circuit(signal_vector, params):
 """
 Variational Quantum Classifier for signal combination.
 
 Args:
 signal_vector: 8-dim signal features in [-1, 1]
 params: (3, 8) array of trainable rotation angles
 
 Returns:
 Expectation value of PauliZ on qubit 0
 """
 # Encoding layer: angle encode each feature
 for i in range(8):
 qml.RY(np.pi * signal_vector[i], wires=i)
 
 # Variational layers
 n_layers = params.shape[0]
 for layer in range(n_layers):
 # Single-qubit rotations
 for i in range(8):
 qml.RY(params[layer, i], wires=i)
 
 # Linear entanglement (chain)
 for i in range(7):
 qml.CNOT(wires=[i, i+1])
 
 # Measurement: expectation of PauliZ on qubit 0
 return qml.expval(qml.PauliZ(0))
 
 print("PennyLane VQC Circuit Ready")
 print(f" Wires (qubits): 8")
 print(f" Variational layers: 3")
 print(f" Trainable parameters: 3 x 8 = 24")
else:
 print("PennyLane not available. Using NumPy implementation below.")

# Notebook cell
# Pure NumPy quantum simulation using state vectors and matrix exponentials

def matrix_ry(theta):
 """RY(theta) gate as 2x2 matrix."""
 return np.array([
 [np.cos(theta/2), -np.sin(theta/2)],
 [np.sin(theta/2), np.cos(theta/2)]
 ])

def matrix_cnot:
 """CNOT(control, target) as 4x4 matrix (control on first qubit)."""
 return np.array([
 [1, 0, 0, 0],
 [0, 1, 0, 0],
 [0, 0, 0, 1],
 [0, 0, 1, 0]
 ])

def apply_single_qubit_gate(state, gate, qubit_idx, n_qubits):
 """Apply single-qubit gate to state vector.
 
 Builds: I ⊗ ... ⊗ gate ⊗ ... ⊗ I
 """
 # Build full unitary using tensor products
 full_gate = 1.0
 for i in range(n_qubits):
 if i == qubit_idx:
 full_gate = np.kron(full_gate, gate)
 else:
 full_gate = np.kron(full_gate, np.eye(2))
 
 return full_gate @ state

def apply_two_qubit_gate(state, gate, control, target, n_qubits):
 """Apply two-qubit gate to state vector.
 
 For CNOT, we build: I ⊗ ... ⊗ CNOT ⊗ ... ⊗ I
 """
 # Simple approach: enumerate all basis states and apply gate element-wise
 dim = 2 ** n_qubits
 new_state = np.zeros_like(state)
 
 for basis_idx in range(dim):
 # Determine if we apply the gate
 bits = [(basis_idx >> i) & 1 for i in range(n_qubits)]
 control_bit = bits[control]
 target_bit = bits[target]
 
 if control_bit == 1:
 # Flip target qubit
 new_bits = bits.copy
 new_bits[target] = 1 - new_bits[target]
 new_basis_idx = sum(bit << i for i, bit in enumerate(new_bits))
 new_state[new_basis_idx] += state[basis_idx]
 else:
 new_state[basis_idx] += state[basis_idx]
 
 return new_state

def vqc_circuit_numpy(signal_vector, params):
 """
 VQC circuit in pure NumPy.
 
 Args:
 signal_vector: (8,) array of features in [-1, 1]
 params: (3, 8) array of trainable angles
 
 Returns:
 Expectation value of Z on qubit 0
 """
 n_qubits = 8
 
 # Initialize state |00...0>
 state = np.zeros(2 ** n_qubits)
 state[0] = 1.0
 
 # Encoding layer: RY(pi * s_i) on qubit i
 for i in range(n_qubits):
 angle = np.pi * signal_vector[i]
 ry_gate = matrix_ry(angle)
 state = apply_single_qubit_gate(state, ry_gate, i, n_qubits)
 
 # Variational layers
 n_layers = params.shape[0]
 for layer in range(n_layers):
 # Single-qubit rotations
 for i in range(n_qubits):
 ry_gate = matrix_ry(params[layer, i])
 state = apply_single_qubit_gate(state, ry_gate, i, n_qubits)
 
 # Linear entanglement
 for i in range(n_qubits - 1):
 state = apply_two_qubit_gate(state, matrix_cnot, i, i+1, n_qubits)
 
 # Measurement: expectation of Z on qubit 0
 # E[Z_0] = sum_basis (amplitude^2) * (-1)^(bit_0)
 expectation = 0.0
 for basis_idx in range(2 ** n_qubits):
 qubit_0_bit = (basis_idx >> 0) & 1
 z_eigenvalue = 1 if qubit_0_bit == 0 else -1
 expectation += (np.abs(state[basis_idx]) ** 2) * z_eigenvalue
 
 return expectation

print("NumPy Quantum Simulator Ready")
print(f" Implementation: State vector + matrix exponentials")
print(f" Supported: RY, CNOT")

# Notebook cell
# Test both implementations for consistency
print("Testing VQC Implementations\n" + "="*50)

# Random signal and parameters
test_signal = np.random.uniform(-1, 1, 8)
test_params = np.random.uniform(-np.pi, np.pi, (3, 8))

# NumPy version
t0 = time.time
output_numpy = vqc_circuit_numpy(test_signal, test_params)
time_numpy = time.time - t0
print(f"NumPy output: {output_numpy:.6f}")
print(f"NumPy time: {time_numpy*1000:.2f} ms")

# PennyLane version (if available)
if PENNYLANE_AVAILABLE:
 t0 = time.time
 output_pl = vqc_circuit(test_signal, test_params)
 time_pl = time.time - t0
 print(f"PennyLane output: {output_pl:.6f}")
 print(f"PennyLane time: {time_pl*1000:.2f} ms")
 print(f"\nDifference: {abs(output_numpy - float(output_pl)):.8f}")
 

# Notebook cell
def compute_sharpe_ratio(returns, threshold=1e-6):
 """Compute Sharpe ratio. Returns 0 if std is too small."""
 mean_ret = np.mean(returns)
 std_ret = np.std(returns)
 
 if std_ret < threshold:
 return 0.0
 
 return mean_ret / std_ret

def loss_function(params_flat, X_data, y_data, use_pennylane=False):
 """
 Loss function: negative Sharpe ratio of weighted returns.
 
 Args:
 params_flat: (24,) flattened parameters
 X_data: (n_samples, 8) features
 y_data: (n_samples,) returns
 use_pennylane: bool, use PennyLane or NumPy
 
 Returns:
 Negative Sharpe ratio (to minimize)
 """
 # Reshape parameters
 params = params_flat.reshape(3, 8)
 
 # Predict weights for each sample
 predictions = np.zeros(len(X_data))
 
 if use_pennylane and PENNYLANE_AVAILABLE:
 for i in range(len(X_data)):
 predictions[i] = float(vqc_circuit(X_data[i], params))
 else:
 for i in range(len(X_data)):
 predictions[i] = vqc_circuit_numpy(X_data[i], params)
 
 # Weighted returns
 weighted_returns = predictions * y_data
 
 # Compute negative Sharpe ratio
 sharpe = compute_sharpe_ratio(weighted_returns)
 
 return -sharpe # Negative because we minimize

# Initialize parameters
params_init = np.random.uniform(-np.pi/4, np.pi/4, (3, 8))
params_flat_init = params_init.flatten

print("Training Configuration")
print(f" Optimizer: COBYLA (gradient-free)")
print(f" Initial parameters shape: {params_init.shape}")
print(f" Loss function: Negative Sharpe ratio")
print(f" Training samples: {len(X_train_scaled)}")
print(f" Max iterations: 200")

# Notebook cell
# Track training progress
loss_history = []
call_count = [0]

def loss_with_tracking(params_flat):
 """Loss function with iteration tracking."""
 call_count[0] += 1
 loss = loss_function(params_flat, X_train_scaled, y_train, use_pennylane=PENNYLANE_AVAILABLE)
 loss_history.append(loss)
 
 if call_count[0] % 10 == 0:
 print(f" Iteration {call_count[0]:3d} | Loss: {loss:8.6f}")
 
 return loss

print("\nStarting Training...\n")
t_start = time.time

# Optimize
result = minimize(
 loss_with_tracking,
 params_flat_init,
 method='COBYLA',
 options={'maxiter': 200, 'rhobeg': 0.1, 'tol': 1e-4}
)

t_train = time.time - t_start

print(f"\nTraining Complete")
print(f" Total iterations: {call_count[0]}")
print(f" Training time: {t_train:.2f} seconds")
print(f" Optimization success: {result.success}")
print(f" Final loss: {result.fun:.6f}")
print(f" Final Sharpe: {-result.fun:.6f}")

# Extract trained parameters
params_trained = result.x.reshape(3, 8)

# Notebook cell
# Plot convergence
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Loss curve
ax = axes[0]
ax.plot(loss_history, linewidth=2, color='#1f77b4')
ax.set_xlabel('Iteration')
ax.set_ylabel('Loss (Negative Sharpe Ratio)')
ax.set_title('Training Convergence')
ax.grid(True, alpha=0.3)

# Sharpe ratio
ax = axes[1]
sharpe_history = [-l for l in loss_history]
ax.plot(sharpe_history, linewidth=2, color='#ff7f0e')
ax.set_xlabel('Iteration')
ax.set_ylabel('Sharpe Ratio')
ax.set_title('Sharpe Ratio Improvement')
ax.grid(True, alpha=0.3)

plt.tight_layout
plt.show

# Notebook cell
# Generate predictions on test set
def predict_vqc(X, params, use_pennylane=False):
 """Generate VQC predictions."""
 predictions = np.zeros(len(X))
 
 if use_pennylane and PENNYLANE_AVAILABLE:
 for i in range(len(X)):
 predictions[i] = float(vqc_circuit(X[i], params))
 else:
 for i in range(len(X)):
 predictions[i] = vqc_circuit_numpy(X[i], params)
 
 return predictions

# VQC predictions
y_train_pred_vqc = predict_vqc(X_train_scaled, params_trained, use_pennylane=PENNYLANE_AVAILABLE)
y_test_pred_vqc = predict_vqc(X_test_scaled, params_trained, use_pennylane=PENNYLANE_AVAILABLE)

# Compute metrics
def compute_metrics(y_true, y_pred, name):
 """Compute evaluation metrics."""
 # Direction accuracy
 direction_correct = np.mean(np.sign(y_true) == np.sign(y_pred))
 
 # Sharpe ratio
 weighted_returns = y_pred * y_true
 sharpe = compute_sharpe_ratio(weighted_returns)
 
 # Correlation
 if np.std(y_pred) > 1e-6 and np.std(y_true) > 1e-6:
 correlation = np.corrcoef(y_true, y_pred)[0, 1]
 else:
 correlation = 0.0
 
 # MSE
 mse = np.mean((y_true - y_pred) ** 2)
 
 print(f"\n{name}")
 print(f" Direction Accuracy: {direction_correct:.4f}")
 print(f" Sharpe Ratio: {sharpe:.4f}")
 print(f" Correlation: {correlation:.4f}")
 print(f" MSE: {mse:.6f}")
 
 return {
 'direction_accuracy': direction_correct,
 'sharpe': sharpe,
 'correlation': correlation,
 'mse': mse
 }

metrics_train = compute_metrics(y_train, y_train_pred_vqc, "VQC Training Metrics")
metrics_test = compute_metrics(y_test, y_test_pred_vqc, "VQC Test Metrics")

# Notebook cell
baseline_results = {}

# Ridge Regression
try:
 from sklearn.linear_model import Ridge
 ridge = Ridge(alpha=1.0)
 ridge.fit(X_train_scaled, y_train)
 y_test_pred_ridge = ridge.predict(X_test_scaled)
 metrics_ridge = compute_metrics(y_test, y_test_pred_ridge, "Ridge Regression Test Metrics")
 baseline_results['ridge'] = metrics_ridge
except Exception as e:
 print(f"Ridge Regression failed: {e}")

# Random Forest
try:
 from sklearn.ensemble import RandomForestRegressor
 rf = RandomForestRegressor(n_estimators=50, max_depth=10, random_state=42, n_jobs=-1)
 rf.fit(X_train_scaled, y_train)
 y_test_pred_rf = rf.predict(X_test_scaled)
 metrics_rf = compute_metrics(y_test, y_test_pred_rf, "Random Forest Test Metrics")
 baseline_results['rf'] = metrics_rf
except Exception as e:
 print(f"Random Forest failed: {e}")

# Buy-and-hold (always 1, i.e., always buy)
buy_hold_pred = np.ones_like(y_test)
metrics_bh = compute_metrics(y_test, buy_hold_pred, "Buy-and-Hold (Baseline) Test Metrics")
baseline_results['buy_hold'] = metrics_bh

# Notebook cell
# Comparison summary
print("\n" + "="*60)
print("PERFORMANCE COMPARISON")
print("="*60)

comparison_data = {
 'VQC': metrics_test,
 **baseline_results
}

comparison_df = pd.DataFrame(comparison_data).T
print(comparison_df)

print("\nKey Observations:")
if comparison_df.loc['VQC', 'sharpe'] > comparison_df['sharpe'].drop('VQC').max:
 print(" ✓ VQC achieves the highest Sharpe ratio among all methods")
else:
 print(f" - VQC Sharpe: {comparison_df.loc['VQC', 'sharpe']:.4f}")

if comparison_df.loc['VQC', 'direction_accuracy'] > 0.5:
 print(f" ✓ VQC direction accuracy: {comparison_df.loc['VQC', 'direction_accuracy']:.4f} (better than random)")
else:
 print(f" - VQC direction accuracy: {comparison_df.loc['VQC', 'direction_accuracy']:.4f}")

# Notebook cell
# Learned parameters
print("Learned VQC Parameters\n" + "="*60)
print(f"Shape: {params_trained.shape} (layers, qubits)")
print(f"\nParameter Range: [{params_trained.min:.4f}, {params_trained.max:.4f}]")
print(f"Parameter Mean: {params_trained.mean:.4f}")
print(f"Parameter Std: {params_trained.std:.4f}")

# Per-layer analysis
for layer in range(params_trained.shape[0]):
 print(f"\nLayer {layer}:")
 for qubit in range(params_trained.shape[1]):
 angle = params_trained[layer, qubit]
 print(f" Qubit {qubit}: {angle:7.4f} rad = {np.degrees(angle):7.2f}°")

# Feature importance (heuristic: magnitude of encoding * first layer weights)
encoding_strength = np.abs(params_trained[0, :]) # First layer weights (post-encoding)
print(f"\nEffective Feature Importance (from encoding + first layer):")
for i in range(8):
 print(f" Feature {i}: {encoding_strength[i]:.4f}")

most_important = np.argmax(encoding_strength)
print(f"\nMost important feature: Feature {most_important} (magnitude {encoding_strength[most_important]:.4f})")

# Notebook cell
# Visualize parameters
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

# Heatmap of all parameters
ax = axes[0]
im = ax.imshow(params_trained, cmap='RdBu_r', aspect='auto')
ax.set_xlabel('Qubit Index')
ax.set_ylabel('Layer Index')
ax.set_title('Learned Rotation Angles (Radians)')
plt.colorbar(im, ax=ax)

# Per-layer magnitude
ax = axes[1]
for layer in range(params_trained.shape[0]):
 ax.bar(range(8), np.abs(params_trained[layer, :]), alpha=0.7, label=f'Layer {layer}')
ax.set_xlabel('Qubit Index')
ax.set_ylabel('|Parameter|')
ax.set_title('Parameter Magnitude by Qubit')
ax.legend
ax.grid(True, alpha=0.3, axis='y')

# Feature importance
ax = axes[2]
ax.bar(range(8), encoding_strength, color='steelblue', alpha=0.7)
ax.set_xlabel('Feature Index')
ax.set_ylabel('Effective Importance')
ax.set_title('Learned Feature Importance')
ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout
plt.show

# Notebook cell
# Generate baseline predictions
if 'ridge' in baseline_results:
 y_test_pred_ridge = ridge.predict(X_test_scaled)
else:
 y_test_pred_ridge = None

if 'rf' in baseline_results:
 y_test_pred_rf = rf.predict(X_test_scaled)
else:
 y_test_pred_rf = None

# Compute cumulative returns
cumulative_vqc = np.cumsum(y_test * y_test_pred_vqc)
cumulative_bh = np.cumsum(y_test)

cumulative_others = {}
if y_test_pred_ridge is not None:
 cumulative_others['Ridge'] = np.cumsum(y_test * y_test_pred_ridge)
if y_test_pred_rf is not None:
 cumulative_others['RF'] = np.cumsum(y_test * y_test_pred_rf)

# Plot cumulative returns
fig, axes = plt.subplots(2, 1, figsize=(14, 8))

# Predictions vs Actuals
ax = axes[0]
test_indices = np.arange(len(y_test))
ax.scatter(test_indices, y_test, alpha=0.5, s=20, label='Actual Returns', color='gray')
ax.plot(test_indices, y_test_pred_vqc, linewidth=2, label='VQC Prediction', color='#1f77b4')
if y_test_pred_ridge is not None:
 ax.plot(test_indices, y_test_pred_ridge, linewidth=2, label='Ridge Prediction', color='#ff7f0e', alpha=0.7)
ax.set_xlabel('Test Sample Index')
ax.set_ylabel('Return / Prediction')
ax.set_title('Actual Returns vs Predictions')
ax.legend
ax.grid(True, alpha=0.3)

# Cumulative returns
ax = axes[1]
ax.plot(test_indices, cumulative_vqc, linewidth=2.5, label='VQC', color='#1f77b4')
ax.plot(test_indices, cumulative_bh, linewidth=2.5, label='Buy-and-Hold', color='gray', linestyle='--')
for name, cum_ret in cumulative_others.items:
 ax.plot(test_indices, cum_ret, linewidth=2, label=name, alpha=0.7)
ax.set_xlabel('Test Sample Index')
ax.set_ylabel('Cumulative Return')
ax.set_title('Cumulative Returns Comparison')
ax.legend
ax.grid(True, alpha=0.3)

plt.tight_layout
plt.show

print(f"\nCumulative Returns Summary")
print(f" VQC: {cumulative_vqc[-1]:.4f}")
print(f" Buy-and-Hold: {cumulative_bh[-1]:.4f}")
if y_test_pred_ridge is not None:
 print(f" Ridge: {cumulative_others['Ridge'][-1]:.4f}")
if y_test_pred_rf is not None:
 print(f" Random Forest: {cumulative_others['RF'][-1]:.4f}")

# Notebook cell
# Benchmark circuit scaling with increasing qubits
qubit_counts = [4, 6, 8, 10, 12]
timings = []

print("Quantum Scaling Benchmark (NumPy)")
print("="*50)

for n_qubits in qubit_counts:
 # Random signal and parameters
 test_signal_q = np.random.uniform(-1, 1, n_qubits)
 test_params_q = np.random.uniform(-np.pi, np.pi, (3, n_qubits))
 
 # Adapt circuit for this qubit count
 def vqc_circuit_n_qubits(signal_vector, params, n_qubits):
 state = np.zeros(2 ** n_qubits)
 state[0] = 1.0
 
 # Encoding
 for i in range(n_qubits):
 angle = np.pi * signal_vector[i]
 ry_gate = matrix_ry(angle)
 state = apply_single_qubit_gate(state, ry_gate, i, n_qubits)
 
 # Variational
 n_layers = params.shape[0]
 for layer in range(n_layers):
 for i in range(n_qubits):
 ry_gate = matrix_ry(params[layer, i])
 state = apply_single_qubit_gate(state, ry_gate, i, n_qubits)
 
 for i in range(n_qubits - 1):
 state = apply_two_qubit_gate(state, matrix_cnot, i, i+1, n_qubits)
 
 # Measurement
 expectation = 0.0
 for basis_idx in range(2 ** n_qubits):
 qubit_0_bit = (basis_idx >> 0) & 1
 z_eigenvalue = 1 if qubit_0_bit == 0 else -1
 expectation += (np.abs(state[basis_idx]) ** 2) * z_eigenvalue
 
 return expectation
 
 # Time single evaluation
 t0 = time.time
 _ = vqc_circuit_n_qubits(test_signal_q, test_params_q, n_qubits)
 elapsed = time.time - t0
 timings.append(elapsed)
 
 state_dim = 2 ** n_qubits
 print(f" {n_qubits:2d} qubits ({state_dim:5d} states): {elapsed*1000:8.2f} ms")

print("\nNote: Runtime grows exponentially (classical simulation).")
print("Real quantum hardware would break even around 20-30 qubits.")

# Notebook cell
# Plot scaling
fig, ax = plt.subplots(figsize=(10, 6))

ax.semilogy(qubit_counts, timings, 'o-', linewidth=2, markersize=10, color='#d62728')
ax.set_xlabel('Number of Qubits')
ax.set_ylabel('Evaluation Time (seconds, log scale)')
ax.set_title('VQC Circuit Scaling (NumPy State Vector Simulator)')
ax.grid(True, alpha=0.3, which='both')
ax.set_xticks(qubit_counts)

# Annotate
for q, t in zip(qubit_counts, timings):
 ax.annotate(f'{t*1000:.0f}ms', xy=(q, t), xytext=(0, 10),
 textcoords='offset points', ha='center', fontsize=9)

plt.tight_layout
plt.show

# Exponential fit
log_times = np.log(np.array(timings))
coeffs = np.polyfit(np.array(qubit_counts), log_times, 1)
print(f"\nExponential Fit: log(time) = {coeffs[0]:.4f} * n_qubits + {coeffs[1]:.4f}")
print(f" Doubling factor per qubit: {np.exp(coeffs[0]):.2f}x")

# Notebook cell
# Create output directory
output_dir = Path('../results')
output_dir.mkdir(parents=True, exist_ok=True)

print(f"Exporting to: {output_dir.resolve}\n")

# Save trained parameters
params_dict = {
 'trained_parameters': params_trained.tolist,
 'shape': list(params_trained.shape),
 'training_loss': float(result.fun),
 'training_sharpe': float(-result.fun),
 'num_iterations': call_count[0],
 'training_time_seconds': t_train
}

with open(output_dir / 'vqc_params.json', 'w') as f:
 json.dump(params_dict, f, indent=2)
print(f"✓ Saved VQC parameters to vqc_params.json")

# Save performance metrics
metrics_dict = {
 'vqc': {
 'train': metrics_train,
 'test': metrics_test
 },
 'baselines': baseline_results,
 'scaling': {
 'qubit_counts': qubit_counts,
 'timings_ms': [t*1000 for t in timings],
 'exponential_doubling_factor': float(np.exp(coeffs[0]))
 }
}

with open(output_dir / 'vqc_metrics.json', 'w') as f:
 json.dump(metrics_dict, f, indent=2)
print(f"✓ Saved metrics to vqc_metrics.json")

# Save synthetic data info
data_dict = {
 'n_samples': int(n_samples),
 'n_features': int(n_features),
 'train_test_split': 0.7,
 'train_size': len(X_train),
 'test_size': len(X_test),
 'return_statistics': {
 'mean': float(y.mean),
 'std': float(y.std),
 'min': float(y.min),
 'max': float(y.max)
 },
 'true_signal_weights': true_weights.tolist
}

with open(output_dir / 'data_info.json', 'w') as f:
 json.dump(data_dict, f, indent=2)
print(f"✓ Saved data info to data_info.json")

print(f"\nAll results saved to {output_dir.resolve}")

# Notebook cell


# Notebook cell

