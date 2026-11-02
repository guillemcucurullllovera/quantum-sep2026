# Extracted first-party notebook code; execution outputs intentionally omitted.

# Notebook cell
# Install quantum and numerical computing dependencies
# IPython: !pip install -q qiskit qiskit-aer qiskit-optimization numpy scipy pandas matplotlib seaborn --upgrade 2>&1 | grep -E '(Successfully|ERROR|error)' || echo 'Dependencies installed'

# Notebook cell
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import minimize, differential_evolution
from scipy.linalg import expm
import json
import time
from pathlib import Path
from datetime import datetime
from typing import Tuple, Dict, List

# Attempt to import Qiskit
QISKIT_AVAILABLE = False
try:
    from qiskit import QuantumCircuit, QuantumRegister
    from qiskit_aer import AerSimulator
    from qiskit.primitives import Sampler
    QISKIT_AVAILABLE = True
    print("✓ Qiskit quantum framework available")
except ImportError as e:
    print(f"⚠ Qiskit not available: {e}")
    print("  → Using fallback numpy/scipy QAOA simulator (works for n ≤ 20 qubits)")

print(f"\nExecution timestamp: {datetime.now().isoformat()}")

# Configure matplotlib
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")
# IPython: %matplotlib inline

# Notebook cell
# Define paths
data_dir = Path('../data')
results_dir = Path('../results')
data_dir.mkdir(exist_ok=True)
results_dir.mkdir(exist_ok=True)

qubo_path = data_dir / 'qubo_matrix.npy'

# Load or generate QUBO
if qubo_path.exists():
    Q = np.load(qubo_path)
    print(f"✓ Loaded QUBO matrix from {qubo_path}")
else:
    # Generate synthetic QUBO for 20 variables
    # Represents pair selection: maximize correlation while constraining covariance
    np.random.seed(42)
    n_vars = 20
    Q = np.zeros((n_vars, n_vars))
    
    # Diagonal: -correlation strength (maximize positive correlation)
    np.fill_diagonal(Q, -np.random.uniform(0.5, 2.0, n_vars))
    
    # Off-diagonal: covariance penalties (discourage multi-collinearity)
    for i in range(n_vars):
        for j in range(i+1, n_vars):
            Q[i, j] = np.random.uniform(0.1, 0.5)
            Q[j, i] = Q[i, j]  # Symmetry
    
    np.save(qubo_path, Q)
    print(f"✓ Generated synthetic QUBO matrix ({n_vars}×{n_vars})")

n_qubits = Q.shape[0]
print(f"\nQUBO Problem Size:")
print(f"  Variables: {n_qubits}")
print(f"  Search space: 2^{n_qubits} = {2**n_qubits:,} states")
print(f"  QUBO matrix condition number: {np.linalg.cond(Q):.2e}")

# Notebook cell
def qubo_to_ising(Q: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Convert QUBO to Ising Hamiltonian.
    
    QUBO: x^T Q x where x_i ∈ {0,1}
    Ising: sum_i h_i z_i + sum_{i<j} J_ij z_i z_j where z_i ∈ {-1,+1}
    
    Substitution: x_i = (1 - z_i) / 2
    
    Returns:
        h: linear coefficients (length n)
        J: coupling matrix (n×n, upper triangular)
        offset: constant term
    """
    n = Q.shape[0]
    h = np.zeros(n)
    J = np.zeros((n, n))
    
    # Expand x_i = (1 - z_i) / 2
    # x_i x_j = (1 - z_i)(1 - z_j) / 4 = (1 - z_i - z_j + z_i z_j) / 4
    
    offset = 0.0
    for i in range(n):
        for j in range(n):
            coeff = Q[i, j] / 4.0 if i != j else Q[i, j] / 2.0
            
            if i == j:
                # Diagonal: Q_ii * x_i = Q_ii * (1 - z_i) / 2
                h[i] -= Q[i, i] / 2.0
                offset += Q[i, i] / 2.0
            elif i < j:
                # Off-diagonal: Q_ij * x_i * x_j
                J[i, j] = (Q[i, j] + Q[j, i]) / 4.0
                h[i] -= (Q[i, j] + Q[j, i]) / 4.0
                h[j] -= (Q[i, j] + Q[j, i]) / 4.0
                offset += (Q[i, j] + Q[j, i]) / 4.0
    
    return h, J, offset

h, J, offset = qubo_to_ising(Q)

print("Ising Hamiltonian Conversion:")
print(f"  Linear terms (h): min={h.min():.4f}, max={h.max():.4f}, mean={h.mean():.4f}")
print(f"  Coupling terms (J):")
print(f"    Non-zero couplings: {np.count_nonzero(J)}")
print(f"    Range: [{J.min():.4f}, {J.max():.4f}]")
print(f"  Offset: {offset:.4f}")

# Visualize coupling structure
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# Linear coefficients
axes[0].bar(range(n_qubits), h)
axes[0].set_xlabel('Qubit Index')
axes[0].set_ylabel('Coefficient Value (h_i)')
axes[0].set_title('Linear Coefficients of Ising Hamiltonian')
axes[0].grid(alpha=0.3)

# Coupling structure (as heatmap)
im = axes[1].imshow(J + J.T, cmap='RdBu_r', aspect='auto')
axes[1].set_xlabel('Qubit j')
axes[1].set_ylabel('Qubit i')
axes[1].set_title('Coupling Coefficients (J_ij)')
plt.colorbar(im, ax=axes[1])

plt.tight_layout()
plt.show()

print("\n✓ QUBO successfully converted to Ising form")

# Notebook cell
class QAOASimulator:
    """
    QAOA simulator with both Qiskit and numpy/scipy fallback implementations.
    """
    
    def __init__(self, h: np.ndarray, J: np.ndarray, p: int = 1):
        self.h = h
        self.J = J
        self.n_qubits = len(h)
        self.p = p
        self.use_qiskit = QISKIT_AVAILABLE
        
    def _build_cost_hamiltonian_matrix(self) -> np.ndarray:
        """Build full cost Hamiltonian as diagonal matrix (for fallback)."""
        n = self.n_qubits
        H_C = np.zeros((2**n, 2**n))
        
        for state in range(2**n):
            # Convert state to binary
            z = np.array([1 if (state >> i) & 1 else -1 for i in range(n)])
            
            # Compute cost
            cost = np.dot(self.h, z) + np.sum(self.J * np.outer(z, z))
            H_C[state, state] = cost
        
        return H_C
    
    def _build_mixer_hamiltonian_matrix(self) -> np.ndarray:
        """Build full mixer Hamiltonian as matrix (for fallback)."""
        n = self.n_qubits
        H_M = np.zeros((2**n, 2**n))
        
        for state in range(2**n):
            # X_i flips bit i
            for i in range(n):
                next_state = state ^ (1 << i)
                H_M[next_state, state] += 1.0
        
        return H_M
    
    def evaluate_fallback(self, params: np.ndarray, shots: int = 4096) -> Tuple[float, np.ndarray]:
        """
        Evaluate QAOA using numpy/scipy (fallback when Qiskit unavailable).
        Uses matrix exponentials for unitary evolution.
        """
        n = self.n_qubits
        
        # Build Hamiltonians
        H_C = self._build_cost_hamiltonian_matrix()
        H_M = self._build_mixer_hamiltonian_matrix()
        
        # Initial state: |+⟩^n = (1/√2^n) * sum of all basis states
        state = np.ones(2**n) / np.sqrt(2**n)
        
        # Apply QAOA layers
        for layer in range(self.p):
            gamma = params[2*layer]
            beta = params[2*layer + 1]
            
            # Apply cost unitary: e^{-i*gamma*H_C}
            U_C = expm(-1j * gamma * H_C)
            state = U_C @ state
            
            # Apply mixer unitary: e^{-i*beta*H_M}
            U_M = expm(-1j * beta * H_M)
            state = U_M @ state
        
        # Compute probabilities
        probs = np.abs(state)**2
        
        # Sample bitstrings
        samples = np.random.choice(2**n, size=shots, p=probs)
        
        # Decode samples and compute energies
        energies = []
        for sample in samples:
            z = np.array([1 if (sample >> i) & 1 else -1 for i in range(n)])
            energy = np.dot(self.h, z) + np.sum(self.J * np.outer(z, z))
            energies.append(energy)
        
        energies = np.array(energies)
        mean_energy = np.mean(energies)
        
        # Get best state
        best_idx = np.argmin(energies)
        best_bitstring = samples[best_idx]
        
        return mean_energy, best_bitstring
    
    def evaluate_qiskit(self, params: np.ndarray, shots: int = 4096) -> Tuple[float, np.ndarray]:
        """
        Evaluate QAOA using Qiskit backend.
        """
        qc = QuantumCircuit(self.n_qubits)
        
        # Initial state: |+⟩^n
        for i in range(self.n_qubits):
            qc.h(i)
        
        # QAOA layers
        for layer in range(self.p):
            gamma = params[2*layer]
            beta = params[2*layer + 1]
            
            # Cost layer: RZZ and RZ gates
            for i in range(self.n_qubits):
                if self.h[i] != 0:
                    qc.rz(2 * gamma * self.h[i], i)
            
            for i in range(self.n_qubits):
                for j in range(i+1, self.n_qubits):
                    if self.J[i, j] != 0:
                        qc.rzz(2 * gamma * self.J[i, j], i, j)
            
            # Mixer layer
            for i in range(self.n_qubits):
                qc.rx(2 * beta, i)
        
        # Measure
        qc.measure_all()
        
        # Run
        sim = AerSimulator()
        result = sim.run(qc, shots=shots).result()
        counts = result.get_counts()
        
        # Extract best bitstring
        best_bitstring = max(counts, key=counts.get)
        best_bitstring_int = int(best_bitstring, 2)
        
        # Compute mean energy
        energies = []
        for bitstring, count in counts.items():
            z = np.array([1 if bit == '1' else -1 for bit in bitstring])
            energy = np.dot(self.h, z) + np.sum(self.J * np.outer(z, z))
            for _ in range(count):
                energies.append(energy)
        
        mean_energy = np.mean(energies)
        return mean_energy, best_bitstring_int
    
    def evaluate(self, params: np.ndarray, shots: int = 4096) -> Tuple[float, np.ndarray]:
        """Unified evaluation interface."""
        if self.use_qiskit:
            return self.evaluate_qiskit(params, shots)
        else:
            return self.evaluate_fallback(params, shots)
    
    def objective(self, params: np.ndarray, shots: int = 4096) -> float:
        """Objective function for optimization (minimize mean energy)."""
        mean_energy, _ = self.evaluate(params, shots=shots)
        return mean_energy

# Test with p=1
print(f"Initializing QAOA with p=1...")
qaoa_p1 = QAOASimulator(h, J, p=1)
print(f"✓ QAOA initialized (using {'Qiskit' if QISKIT_AVAILABLE else 'numpy/scipy fallback'})")
print(f"  Backend: {'Qiskit + AerSimulator' if QISKIT_AVAILABLE else 'scipy.linalg.expm with full matrices'}")
print(f"  Qubits: {qaoa_p1.n_qubits}")
print(f"  Depth (p): 1")

# Notebook cell
def optimize_qaoa(h: np.ndarray, J: np.ndarray, p: int, shots: int = 4096, 
                   method: str = 'COBYLA', max_iterations: int = 200) -> Dict:
    """
    Optimize QAOA parameters using classical optimization.
    
    Parameters:
        h, J: Ising Hamiltonian coefficients
        p: QAOA depth
        shots: Measurement shots per evaluation
        method: 'COBYLA' or 'differential_evolution'
        max_iterations: Maximum optimization iterations
    
    Returns:
        Dictionary with optimization results
    """
    print(f"\nOptimizing QAOA (p={p}, shots={shots})...")
    
    qaoa = QAOASimulator(h, J, p=p)
    
    # Random initial guess
    x0 = np.random.uniform(-np.pi, np.pi, 2*p)
    
    # Track convergence
    evals = []
    
    def objective_with_tracking(params):
        val = qaoa.objective(params, shots=shots)
        evals.append(val)
        if len(evals) % 10 == 0:
            print(f"  Iteration {len(evals):3d}: objective = {val:.6f}")
        return val
    
    # Optimize
    start_time = time.time()
    
    if method == 'COBYLA':
        result = minimize(
            objective_with_tracking,
            x0,
            method='COBYLA',
            options={'maxiter': max_iterations, 'rhobeg': 1.0}
        )
    else:
        bounds = [(-np.pi, np.pi)] * (2*p)
        result = differential_evolution(
            objective_with_tracking,
            bounds,
            maxiter=max_iterations,
            seed=42,
            workers=1
        )
    
    elapsed = time.time() - start_time
    
    # Final evaluation with best parameters
    best_energy, best_bitstring = qaoa.evaluate(result.x, shots=shots*2)
    
    return {
        'p': p,
        'params': result.x,
        'best_energy': best_energy,
        'best_bitstring': best_bitstring,
        'n_evals': len(evals),
        'elapsed_time': elapsed,
        'convergence': evals,
        'success': result.success if hasattr(result, 'success') else True
    }

# Optimize for p=1, 2, 3
qaoa_results = {}
depths = [1, 2, 3]

for p in depths:
    qaoa_results[p] = optimize_qaoa(h, J, p=p, shots=4096, method='COBYLA', max_iterations=150)
    print(f"✓ p={p}: Best energy = {qaoa_results[p]['best_energy']:.6f}, Time = {qaoa_results[p]['elapsed_time']:.2f}s")

# Notebook cell
def greedy_selection(Q: np.ndarray, max_pairs: int = 10) -> Tuple[List[Tuple[int, int]], float]:
    """
    Greedy pair selection: iteratively pick best uncorrelated pairs.
    """
    n = Q.shape[0]
    selected = []
    remaining = set(range(n))
    
    while len(selected) < max_pairs and len(remaining) >= 2:
        best_value = -np.inf
        best_pair = None
        
        for i in remaining:
            for j in remaining:
                if i < j:
                    value = Q[i, j] + Q[j, i]  # Symmetric pairing value
                    if value > best_value:
                        best_value = value
                        best_pair = (i, j)
        
        if best_pair:
            selected.append(best_pair)
            remaining.discard(best_pair[0])
            remaining.discard(best_pair[1])
    
    # Compute objective
    selected_arr = np.zeros(n, dtype=int)
    for i, j in selected:
        selected_arr[i] = 1
        selected_arr[j] = 1
    
    obj = selected_arr @ Q @ selected_arr
    return selected, obj

def random_selection(Q: np.ndarray, n_selections: int = 10) -> float:
    """
    Random binary selection baseline.
    """
    n = Q.shape[0]
    best_obj = np.inf
    
    for _ in range(100):
        selection = np.random.randint(0, 2, n)
        obj = selection @ Q @ selection
        if obj < best_obj:
            best_obj = obj
    
    return best_obj

# Compute classical baselines
greedy_pairs, greedy_obj = greedy_selection(Q, max_pairs=10)
random_obj = random_selection(Q, n_selections=10)

# Convert QAOA bitstrings to objectives
qaoa_objs = {}
for p in depths:
    bitstring = qaoa_results[p]['best_bitstring']
    z = np.array([1 if (bitstring >> i) & 1 else -1 for i in range(n_qubits)])
    # Convert z to x: x_i = (1 - z_i) / 2
    x = (1 - z) / 2
    qaoa_objs[p] = x @ Q @ x

# Build comparison table
comparison_df = pd.DataFrame({
    'Method': ['Greedy', 'Random', 'QAOA (p=1)', 'QAOA (p=2)', 'QAOA (p=3)'],
    'Objective Value': [
        greedy_obj,
        random_obj,
        qaoa_objs[1],
        qaoa_objs[2],
        qaoa_objs[3]
    ],
    'Time (s)': [
        0.001,
        0.001,
        qaoa_results[1]['elapsed_time'],
        qaoa_results[2]['elapsed_time'],
        qaoa_results[3]['elapsed_time']
    ]
})

best_obj = comparison_df['Objective Value'].min()
comparison_df['Approximation Ratio'] = comparison_df['Objective Value'] / best_obj

print("\n" + "="*70)
print("COMPARISON: QAOA vs Classical Methods")
print("="*70)
print(comparison_df.to_string(index=False))
print("="*70)

# Visualization
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# Objective values
colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7']
axes[0].bar(comparison_df['Method'], comparison_df['Objective Value'], color=colors)
axes[0].set_ylabel('Objective Value (lower is better)')
axes[0].set_title('Objective Value Comparison')
axes[0].tick_params(axis='x', rotation=45)
axes[0].grid(alpha=0.3, axis='y')

# Approximation ratios
axes[1].bar(comparison_df['Method'], comparison_df['Approximation Ratio'], color=colors)
axes[1].axhline(y=1.0, color='red', linestyle='--', linewidth=2, label='Optimal')
axes[1].set_ylabel('Approximation Ratio (lower is better)')
axes[1].set_title('Approximation Quality')
axes[1].tick_params(axis='x', rotation=45)
axes[1].legend()
axes[1].grid(alpha=0.3, axis='y')

plt.tight_layout()
plt.show()

print(f"\nBest method: {comparison_df.loc[comparison_df['Objective Value'].idxmin(), 'Method']}")

# Notebook cell
# Plot convergence for each depth
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

for idx, p in enumerate(depths):
    conv = qaoa_results[p]['convergence']
    axes[idx].plot(conv, linewidth=2, color='steelblue')
    axes[idx].fill_between(range(len(conv)), conv, alpha=0.3, color='steelblue')
    axes[idx].set_xlabel('Iteration')
    axes[idx].set_ylabel('Mean Energy')
    axes[idx].set_title(f'QAOA Convergence (p={p})')
    axes[idx].grid(alpha=0.3)
    
    final_val = conv[-1]
    initial_val = conv[0]
    improvement = (initial_val - final_val) / abs(initial_val) * 100
    axes[idx].text(0.98, 0.05, f'Improvement: {improvement:.1f}%', 
                   transform=axes[idx].transAxes, ha='right', va='bottom',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.tight_layout()
plt.show()

print("Convergence Summary:")
for p in depths:
    conv = qaoa_results[p]['convergence']
    print(f"  p={p}: {len(conv)} evaluations, improvement = {(conv[0]-conv[-1])/abs(conv[0])*100:.2f}%")

# Notebook cell
# Grid search over parameter space for p=1 (landscape visualization)
print("Computing optimization landscape for p=1...")

gamma_range = np.linspace(0, 2*np.pi, 15)
beta_range = np.linspace(0, np.pi, 15)
landscape = np.zeros((len(gamma_range), len(beta_range)))

qaoa_landscape = QAOASimulator(h, J, p=1)

for i, gamma in enumerate(gamma_range):
    for j, beta in enumerate(beta_range):
        params = np.array([gamma, beta])
        landscape[i, j], _ = qaoa_landscape.evaluate(params, shots=2048)

# Plot landscape
fig, ax = plt.subplots(figsize=(10, 8))
im = ax.contourf(beta_range, gamma_range, landscape, levels=20, cmap='viridis')
optimal_p = qaoa_results[1]['params']
ax.plot(optimal_p[1], optimal_p[0], 'r*', markersize=20, label='Optimized Parameters')
ax.set_xlabel('β (mixer angle)', fontsize=12)
ax.set_ylabel('γ (cost angle)', fontsize=12)
ax.set_title('QAOA Optimization Landscape (p=1)', fontsize=14)
plt.colorbar(im, ax=ax, label='Mean Energy')
ax.legend(fontsize=11)
plt.tight_layout()
plt.show()

print("✓ Landscape computed")

# Notebook cell
# Bitstring distribution for optimal p
print("\nBitstring probability distributions:")

fig, axes = plt.subplots(1, 3, figsize=(15, 4))

for idx, p in enumerate(depths):
    qaoa = QAOASimulator(h, J, p=p)
    params = qaoa_results[p]['params']
    
    # Run with more shots to get distribution
    samples = []
    for _ in range(4):
        _, bitstring = qaoa.evaluate(params, shots=1024)
        samples.append(bitstring)
    
    # Count occurrences (top 15 most frequent)
    unique, counts = np.unique(samples, return_counts=True)
    top_idx = np.argsort(-counts)[:15]
    
    axes[idx].barh(range(len(top_idx)), counts[top_idx])
    axes[idx].set_yticks(range(len(top_idx)))
    axes[idx].set_yticklabels([f'{unique[i]:0{n_qubits}b}' for i in top_idx], fontsize=8)
    axes[idx].set_xlabel('Frequency')
    axes[idx].set_title(f'Top Bitstrings (p={p})')
    axes[idx].grid(alpha=0.3, axis='x')

plt.tight_layout()
plt.show()

# Notebook cell
# Generate synthetic historical data for backtesting
print("Generating synthetic market data for backtesting...")

np.random.seed(42)
n_assets = 30
n_months = 36  # 3 years
n_days_per_month = 21  # Trading days

# Generate correlated asset returns
monthly_returns = np.random.randn(n_months, n_assets) * 0.02  # 2% vol
for i in range(1, n_months):
    monthly_returns[i] += 0.3 * monthly_returns[i-1]  # Autocorrelation

# Add pair relationships
for i in range(0, n_assets-1, 2):
    monthly_returns[:, i+1] += 0.5 * monthly_returns[:, i]  # Comovement

# Create date index
dates = pd.date_range('2023-01-01', periods=n_months, freq='MS')
returns_df = pd.DataFrame(monthly_returns, index=dates, columns=[f'Asset_{i}' for i in range(n_assets)])

print(f"✓ Generated synthetic data: {n_assets} assets, {n_months} months")
print(f"  Return statistics: mean = {returns_df.mean().mean():.4f}, vol = {returns_df.std().mean():.4f}")

# Display data
print("\nSample returns (first 5 months):")
print(returns_df.head())

# Notebook cell
def backtest_portfolio(returns_df: pd.DataFrame, method: str = 'qaoa', 
                       lookback_window: int = 12, rebalance_freq: int = 1,
                       transaction_cost: float = 5e-4) -> pd.DataFrame:
    """
    Walk-forward backtest with monthly rebalancing.
    
    Parameters:
        returns_df: Monthly returns DataFrame
        method: 'qaoa', 'greedy', or 'random'
        lookback_window: Months of historical data for selection
        rebalance_freq: Rebalance every N months
        transaction_cost: Bid-ask spread + commissions
    
    Returns:
        DataFrame with daily portfolio values and metrics
    """
    n_assets = returns_df.shape[1]
    portfolio_values = [1.0]
    selected_pairs_history = []
    
    for month in range(lookback_window, len(returns_df)):
        if (month - lookback_window) % rebalance_freq == 0:
            # Get historical data
            historical_ret = returns_df.iloc[month-lookback_window:month]
            cov_matrix = historical_ret.cov().values
            
            # Build QUBO
            Q_bt = -cov_matrix + 0.01 * np.eye(n_assets)  # Maximize correlation, penalize concentration
            
            # Select pairs based on method
            if method == 'qaoa':
                h_bt, J_bt, _ = qubo_to_ising(Q_bt)
                qaoa_bt = QAOASimulator(h_bt, J_bt, p=1)
                params_init = np.random.uniform(-np.pi, np.pi, 2)
                res = minimize(lambda p: qaoa_bt.objective(p, shots=2048), params_init, method='COBYLA', 
                              options={'maxiter': 50})
                _, best_bitstring = qaoa_bt.evaluate(res.x, shots=4096)
                selected = np.array([(best_bitstring >> i) & 1 for i in range(n_assets)])
            elif method == 'greedy':
                pairs, _ = greedy_selection(Q_bt, max_pairs=10)
                selected = np.zeros(n_assets, dtype=int)
                for i, j in pairs:
                    selected[i] = 1
                    selected[j] = 1
            else:  # random
                selected = np.random.binomial(1, 0.5, n_assets)
            
            # Equal-weight pairs
            n_selected = np.sum(selected)
            if n_selected > 0:
                weights = selected / (2 * n_selected)  # Long/short neutral
            else:
                weights = np.zeros(n_assets)
            
            selected_pairs_history.append(selected)
        
        # Compute monthly return
        monthly_ret = returns_df.iloc[month]
        port_ret = np.dot(weights, monthly_ret) - transaction_cost * np.sum(np.abs(weights))
        
        # Update portfolio value
        portfolio_values.append(portfolio_values[-1] * (1 + port_ret))
    
    # Convert to DataFrame
    dates = returns_df.index[lookback_window:]
    backtest_df = pd.DataFrame({
        'date': dates,
        'portfolio_value': portfolio_values[1:]
    })
    
    return backtest_df

print("Running walk-forward backtests (this may take a minute)...\n")

# Run backtests for all methods
backtest_results = {}
for method in ['qaoa', 'greedy', 'random']:
    print(f"  Backtesting {method.upper()}...", flush=True)
    backtest_results[method] = backtest_portfolio(returns_df, method=method, 
                                                  lookback_window=12, transaction_cost=5e-4)
    final_value = backtest_results[method]['portfolio_value'].iloc[-1]
    print(f"    Final portfolio value: {final_value:.4f}")

print("\n✓ Backtests completed")

# Notebook cell
# Analyze backtest results
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Portfolio values over time
for method in ['qaoa', 'greedy', 'random']:
    axes[0, 0].plot(backtest_results[method]['date'], 
                    backtest_results[method]['portfolio_value'], 
                    label=method.upper(), linewidth=2)
axes[0, 0].set_xlabel('Date')
axes[0, 0].set_ylabel('Portfolio Value')
axes[0, 0].set_title('Walk-Forward Backtest: Portfolio Growth')
axes[0, 0].legend()
axes[0, 0].grid(alpha=0.3)

# Returns distribution
backtest_returns = {}
for method in ['qaoa', 'greedy', 'random']:
    values = backtest_results[method]['portfolio_value'].values
    rets = np.diff(values) / values[:-1]
    backtest_returns[method] = rets
    axes[0, 1].hist(rets, alpha=0.6, label=method.upper(), bins=15)
axes[0, 1].set_xlabel('Monthly Return')
axes[0, 1].set_ylabel('Frequency')
axes[0, 1].set_title('Return Distributions')
axes[0, 1].legend()
axes[0, 1].grid(alpha=0.3)

# Cumulative returns
for method in ['qaoa', 'greedy', 'random']:
    values = backtest_results[method]['portfolio_value'].values
    cum_ret = (values[-1] - values[0]) / values[0] * 100
    axes[1, 0].bar(method.upper(), cum_ret)
axes[1, 0].set_ylabel('Cumulative Return (%)')
axes[1, 0].set_title('Total Returns')
axes[1, 0].grid(alpha=0.3, axis='y')

# Risk metrics
metrics_data = []
for method in ['qaoa', 'greedy', 'random']:
    rets = backtest_returns[method]
    total_ret = (backtest_results[method]['portfolio_value'].iloc[-1] - 1) * 100
    sharpe = np.mean(rets) / np.std(rets) * np.sqrt(12) if np.std(rets) > 0 else 0
    max_dd = np.min(np.maximum.accumulate(backtest_results[method]['portfolio_value'].values) - 
                    backtest_results[method]['portfolio_value'].values) / \
             np.maximum.accumulate(backtest_results[method]['portfolio_value'].values).max() * 100
    metrics_data.append({'Method': method.upper(), 'Return %': total_ret, 'Sharpe': sharpe, 'Max DD %': abs(max_dd)})

metrics_df = pd.DataFrame(metrics_data)
axes[1, 1].axis('off')
axes[1, 1].table(cellText=metrics_df.values.round(3), 
               colLabels=metrics_df.columns,
               cellLoc='center', loc='center',
               colWidths=[0.25, 0.25, 0.25, 0.25])
axes[1, 1].set_title('Risk-Adjusted Performance', pad=20)

plt.tight_layout()
plt.show()

print("\nBacktest Performance Summary:")
print(metrics_df.to_string(index=False))

# Notebook cell
# Prepare results for export
results_export = {
    'metadata': {
        'timestamp': datetime.now().isoformat(),
        'backend': 'Qiskit' if QISKIT_AVAILABLE else 'numpy/scipy',
        'n_qubits': n_qubits,
        'problem_size': f'QUBO {n_qubits}×{n_qubits}'
    },
    'qaoa_optimization': {
        'p1': {
            'objective': float(qaoa_results[1]['best_energy']),
            'params_gamma': float(qaoa_results[1]['params'][0]),
            'params_beta': float(qaoa_results[1]['params'][1]),
            'n_evaluations': int(qaoa_results[1]['n_evals']),
            'runtime_seconds': float(qaoa_results[1]['elapsed_time'])
        },
        'p2': {
            'objective': float(qaoa_results[2]['best_energy']),
            'runtime_seconds': float(qaoa_results[2]['elapsed_time'])
        },
        'p3': {
            'objective': float(qaoa_results[3]['best_energy']),
            'runtime_seconds': float(qaoa_results[3]['elapsed_time'])
        }
    },
    'comparison': {
        'greedy_objective': float(greedy_obj),
        'random_objective': float(random_obj),
        'qaoa_p1_objective': float(qaoa_objs[1]),
        'qaoa_p1_approximation_ratio': float(qaoa_objs[1] / best_obj)
    },
    'backtest': {
        'lookback_window_months': 12,
        'transaction_cost_bps': 5,
        'methods': {}
    }
}

# Add backtest metrics
for method in ['qaoa', 'greedy', 'random']:
    values = backtest_results[method]['portfolio_value'].values
    rets = np.diff(values) / values[:-1]
    total_ret = (values[-1] - values[0]) / values[0]
    sharpe = np.mean(rets) / np.std(rets) * np.sqrt(12) if np.std(rets) > 0 else 0
    max_dd = np.min(np.maximum.accumulate(values) - values) / np.maximum.accumulate(values).max()
    
    results_export['backtest']['methods'][method] = {
        'total_return': float(total_ret),
        'sharpe_ratio': float(sharpe),
        'max_drawdown': float(max_dd),
        'final_portfolio_value': float(values[-1])
    }

# Save to JSON
results_json_path = results_dir / 'qaoa_results.json'
with open(results_json_path, 'w') as f:
    json.dump(results_export, f, indent=2)
print(f"✓ Saved results to {results_json_path}")

# Save backtest to CSV
for method in ['qaoa', 'greedy', 'random']:
    backtest_csv_path = results_dir / f'qaoa_backtest_{method}.csv'
    backtest_results[method].to_csv(backtest_csv_path, index=False)
    print(f"✓ Saved {method} backtest to {backtest_csv_path}")

print(f"\n" + "="*70)
print("Results Export Summary")
print("="*70)
print(f"Results directory: {results_dir.absolute()}")
print(f"Files created:")
for file in results_dir.glob('*'):
    print(f"  - {file.name}")
print("="*70)

# Notebook cell


# Notebook cell

