# Extracted first-party notebook code; execution outputs intentionally omitted.

# Notebook cell
# IPython: !pip install scipy numpy pandas matplotlib seaborn dwave-ocean-sdk

# Notebook cell
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os
import time
from itertools import combinations
from scipy.optimize import milp, LinearConstraint, Bounds
from scipy.sparse import csr_matrix
import warnings
warnings.filterwarnings('ignore')

# Try to import D-Wave Ocean SDK, with fallback
try:
 from dwave.samplers import SimulatedAnnealingSampler
 DWAVE_AVAILABLE = True
except ImportError:
 DWAVE_AVAILABLE = False
 print("D-Wave Ocean SDK not available. Will use numpy-based simulated annealing.")

# Set random seed for reproducibility
np.random.seed(42)

print("Setup complete. D-Wave available:", DWAVE_AVAILABLE)

# Notebook cell
# Define data path relative to notebook location
data_dir = os.path.join(os.getcwd, '..', 'data')
cointegrated_pairs_path = os.path.join(data_dir, 'cointegrated_pairs.csv')

# Try to load data from notebook 01
if os.path.exists(cointegrated_pairs_path):
 print(f"Loading cointegrated pairs from {cointegrated_pairs_path}")
 pairs_data = pd.read_csv(cointegrated_pairs_path)
else:
 print(f"File not found at {cointegrated_pairs_path}. Generating synthetic pair data.")
 
 # Generate synthetic cointegrated pairs
 n_pairs = np.random.randint(30, 51)
 pairs_data = pd.DataFrame({
 'pair_id': [f'pair_{i:03d}' for i in range(n_pairs)],
 'asset_1': [f'ticker_{np.random.randint(0, 500):03d}' for _ in range(n_pairs)],
 'asset_2': [f'ticker_{np.random.randint(0, 500):03d}' for _ in range(n_pairs)],
 'alpha': np.random.uniform(0.05, 0.25, n_pairs),
 'correlation': np.random.uniform(0.3, 0.95, n_pairs),
 'half_life': np.random.uniform(5, 60, n_pairs),
 'liquidity_score': np.random.uniform(0.6, 1.0, n_pairs),
 'p_value': np.random.uniform(0.001, 0.05, n_pairs)
 })
 
 # Ensure directory exists
 os.makedirs(data_dir, exist_ok=True)
 pairs_data.to_csv(cointegrated_pairs_path, index=False)
 print(f"Generated synthetic data: {n_pairs} pairs")

print(f"\nLoaded {len(pairs_data)} candidate pairs")
print(pairs_data.head(10))

# Notebook cell
# Summary statistics of pair characteristics
print("\nPair Characteristics Summary:")
print(pairs_data[['alpha', 'correlation', 'half_life', 'liquidity_score']].describe)

# Check for missing values
print("\nMissing values:")
print(pairs_data.isnull.sum)

# Notebook cell
# QUBO parameters
K = 15 # Maximum number of pairs to select
lambda_corr = 1.0 # Correlation penalty coefficient
P = 10.0 # Penalty weight for constraint violation (tuned empirically)

n = len(pairs_data)
alpha_scores = pairs_data['alpha'].values

print(f"Problem size: n = {n} pairs")
print(f"Max pairs (K) = {K}")
print(f"Lambda (correlation penalty) = {lambda_corr}")
print(f"Penalty weight (P) = {P}")

# Notebook cell
def build_qubo_matrix(pair_data, K, lambda_corr, P):
 """
 Build the QUBO matrix Q from pair characteristics.
 
 Returns Q as an upper triangular matrix (n x n).
 """
 n = len(pair_data)
 Q = np.zeros((n, n))
 
 alpha_scores = pair_data['alpha'].values
 correlations = pair_data['correlation'].values
 
 # Compute pairwise correlation matrix
 # For simplicity, use pair-wise correlation (if available) or approximate
 corr_matrix = np.outer(correlations, correlations) * 0.5
 np.fill_diagonal(corr_matrix, 0) # No self-correlation
 
 # Build QUBO: Q_ij = coefficient of x_i * x_j term
 # Diagonal: Q_ii = alpha_i - P(1 - 2K)
 for i in range(n):
 Q[i, i] = alpha_scores[i] - P * (1 - 2 * K)
 
 # Off-diagonal: Q_ij = -lambda * corr_ij + 2P (for i < j)
 for i in range(n):
 for j in range(i + 1, n):
 Q[i, j] = -lambda_corr * corr_matrix[i, j] + 2 * P
 
 return Q

Q = build_qubo_matrix(pairs_data, K, lambda_corr, P)
print(f"QUBO matrix shape: {Q.shape}")
print(f"\nDiagonal range: [{Q.diagonal.min:.4f}, {Q.diagonal.max:.4f}]")
print(f"Off-diagonal range: [{Q[np.triu_indices_from(Q, k=1)].min:.4f}, {Q[np.triu_indices_from(Q, k=1)].max:.4f}]")

# Notebook cell
# Visualize QUBO matrix as heatmap
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Full matrix
sns.heatmap(Q, cmap='RdBu_r', center=0, ax=axes[0], cbar_kws={'label': 'Q[i,j]'})
axes[0].set_title('QUBO Matrix Q (Full)')
axes[0].set_xlabel('Pair Index j')
axes[0].set_ylabel('Pair Index i')

# Upper triangular only (sparse representation)
Q_upper = np.triu(Q)
sns.heatmap(Q_upper, cmap='RdBu_r', center=0, ax=axes[1], cbar_kws={'label': 'Q[i,j]'})
axes[1].set_title('QUBO Matrix Q (Upper Triangular)')
axes[1].set_xlabel('Pair Index j')
axes[1].set_ylabel('Pair Index i')

plt.tight_layout
plt.savefig(os.path.join(data_dir, 'qubo_matrix_heatmap.png'), dpi=150, bbox_inches='tight')
plt.show

print("QUBO matrix heatmap saved.")

# Notebook cell
def evaluate_solution(x, alpha_scores, corr_matrix, lambda_corr, K, P):
 """
 Evaluate objective value and constraint satisfaction for a solution.
 """
 n_selected = x.sum
 
 # Objective: maximize alpha - lambda * correlation_penalty
 alpha_term = np.sum(alpha_scores * x)
 corr_penalty = lambda_corr * np.sum(corr_matrix[np.ix_(x == 1, x == 1)])
 obj_value = alpha_term - corr_penalty
 
 # Constraint penalty
 constraint_violation = max(0, n_selected - K)
 penalty = P * constraint_violation ** 2
 
 qubo_value = obj_value - penalty
 
 return {
 'n_selected': n_selected,
 'alpha_term': alpha_term,
 'correlation_penalty': corr_penalty,
 'objective': obj_value,
 'constraint_violation': constraint_violation,
 'qubo_value': qubo_value,
 'is_feasible': constraint_violation == 0
 }

# Compute pairwise correlation matrix for evaluation
corr_matrix = np.outer(pairs_data['correlation'].values, pairs_data['correlation'].values) * 0.5
np.fill_diagonal(corr_matrix, 0)

# Notebook cell
# Solver 1: Exhaustive Search (if problem is small enough)
print("=" * 80)
print("SOLVER 1: EXHAUSTIVE SEARCH")
print("=" * 80)

exhaustive_result = None

if n <= 20:
 print(f"Problem size {n} <= 20. Running exhaustive search...")
 start_time = time.time
 
 best_solution = None
 best_qubo_value = -np.inf
 
 # Try all possible selections up to K pairs
 for k in range(K + 1):
 for pair_indices in combinations(range(n), k):
 x = np.zeros(n, dtype=int)
 x[list(pair_indices)] = 1
 
 metrics = evaluate_solution(x, alpha_scores, corr_matrix, lambda_corr, K, P)
 
 if metrics['qubo_value'] > best_qubo_value:
 best_qubo_value = metrics['qubo_value']
 best_solution = x.copy
 best_metrics = metrics
 
 elapsed = time.time - start_time
 exhaustive_result = {
 'solution': best_solution,
 'metrics': best_metrics,
 'runtime': elapsed,
 'solver': 'Exhaustive Search'
 }
 
 print(f"Best solution found in {elapsed:.4f} seconds")
 print(f" Selected pairs: {best_metrics['n_selected']}")
 print(f" Alpha sum: {best_metrics['alpha_term']:.4f}")
 print(f" Correlation penalty: {best_metrics['correlation_penalty']:.4f}")
 print(f" QUBO value: {best_metrics['qubo_value']:.4f}")
 print(f" Feasible: {best_metrics['is_feasible']}")
else:
 print(f"Problem size {n} > 20. Skipping exhaustive search.")

# Notebook cell
# Solver 2: Simulated Annealing
print("\n" + "=" * 80)
print("SOLVER 2: SIMULATED ANNEALING")
print("=" * 80)

if DWAVE_AVAILABLE:
 print("Using D-Wave Ocean SDK SimulatedAnnealingSampler...")
 start_time = time.time
 
 # Create QUBO dictionary for D-Wave
 qubo_dict = {}
 for i in range(n):
 qubo_dict[(i, i)] = Q[i, i]
 for i in range(n):
 for j in range(i + 1, n):
 if Q[i, j] != 0:
 qubo_dict[(i, j)] = Q[i, j]
 
 sampler = SimulatedAnnealingSampler
 sampleset = sampler.sample_qubo(qubo_dict, num_reads=100)
 
 # Get best solution
 best_sample = sampleset.first.sample
 sa_solution = np.array([best_sample[i] for i in range(n)], dtype=int)
 
 elapsed = time.time - start_time
 sa_metrics = evaluate_solution(sa_solution, alpha_scores, corr_matrix, lambda_corr, K, P)
 
 sa_result = {
 'solution': sa_solution,
 'metrics': sa_metrics,
 'runtime': elapsed,
 'solver': 'D-Wave Simulated Annealing'
 }
 
 print(f"Solution found in {elapsed:.4f} seconds")
 print(f" Selected pairs: {sa_metrics['n_selected']}")
 print(f" Alpha sum: {sa_metrics['alpha_term']:.4f}")
 print(f" Correlation penalty: {sa_metrics['correlation_penalty']:.4f}")
 print(f" QUBO value: {sa_metrics['qubo_value']:.4f}")
 print(f" Feasible: {sa_metrics['is_feasible']}")

else:
 print("D-Wave not available. Implementing numpy-based simulated annealing...")
 
 def numpy_simulated_annealing(Q, alpha_scores, n, K, lambda_corr, P, 
 T_initial=10.0, T_final=0.1, iterations=1000):
 """
 Pure numpy implementation of simulated annealing for QUBO.
 """
 # Start with random feasible solution
 x = np.zeros(n, dtype=int)
 initial_pairs = np.random.choice(n, min(K, n // 2), replace=False)
 x[initial_pairs] = 1
 
 # Compute initial QUBO value
 def qubo_value(x_state):
 value = np.dot(x_state, np.diag(Q))
 for i in range(n):
 for j in range(i + 1, n):
 value += Q[i, j] * x_state[i] * x_state[j]
 return value
 
 current_value = qubo_value(x)
 best_x = x.copy
 best_value = current_value
 
 # Annealing schedule
 for iteration in range(iterations):
 T = T_initial * (T_final / T_initial) ** (iteration / iterations)
 
 # Random flip of one bit
 flip_idx = np.random.randint(0, n)
 x_new = x.copy
 x_new[flip_idx] = 1 - x_new[flip_idx]
 
 # Only accept if feasible and improves or accepted by Metropolis criterion
 if x_new.sum <= K:
 new_value = qubo_value(x_new)
 delta = new_value - current_value
 
 if delta > 0 or np.exp(delta / max(T, 0.01)) > np.random.rand:
 x = x_new
 current_value = new_value
 
 if new_value > best_value:
 best_x = x_new.copy
 best_value = new_value
 
 return best_x
 
 start_time = time.time
 sa_solution = numpy_simulated_annealing(Q, alpha_scores, n, K, lambda_corr, P)
 elapsed = time.time - start_time
 
 sa_metrics = evaluate_solution(sa_solution, alpha_scores, corr_matrix, lambda_corr, K, P)
 
 sa_result = {
 'solution': sa_solution,
 'metrics': sa_metrics,
 'runtime': elapsed,
 'solver': 'Numpy Simulated Annealing'
 }
 
 print(f"Solution found in {elapsed:.4f} seconds")
 print(f" Selected pairs: {sa_metrics['n_selected']}")
 print(f" Alpha sum: {sa_metrics['alpha_term']:.4f}")
 print(f" Correlation penalty: {sa_metrics['correlation_penalty']:.4f}")
 print(f" QUBO value: {sa_metrics['qubo_value']:.4f}")
 print(f" Feasible: {sa_metrics['is_feasible']}")

# Notebook cell
# Solver 3: Integer Linear Programming (MILP)
print("\n" + "=" * 80)
print("SOLVER 3: INTEGER LINEAR PROGRAMMING (MILP)")
print("=" * 80)
print("Using scipy.optimize.milp...")

start_time = time.time

# MILP formulation: minimize -QUBO (since scipy minimizes)
# Objective: minimize sum of linear + quadratic terms
# We need to linearize the quadratic terms using auxiliary variables

# For simplicity, use linear approximation: just alpha scores as objective
c = -alpha_scores # Negative because we want to maximize

# Constraint: sum(x) <= K
A = csr_matrix(np.ones((1, n)))
constraints = LinearConstraint(A, -np.inf, K)

# Variable bounds: 0 <= x_i <= 1 (integer)
bounds = Bounds(lb=0, ub=1)

# Solve
from scipy.optimize import LinearConstraint, milp, Bounds
from scipy.sparse import csr_matrix

result = milp(c=c, constraints=constraints, bounds=bounds, integrality=np.ones(n))

elapsed = time.time - start_time
milp_solution = result.x.astype(int)
milp_metrics = evaluate_solution(milp_solution, alpha_scores, corr_matrix, lambda_corr, K, P)

milp_result = {
 'solution': milp_solution,
 'metrics': milp_metrics,
 'runtime': elapsed,
 'solver': 'Scipy MILP'
}

print(f"Solution found in {elapsed:.4f} seconds")
print(f" Selected pairs: {milp_metrics['n_selected']}")
print(f" Alpha sum: {milp_metrics['alpha_term']:.4f}")
print(f" Correlation penalty: {milp_metrics['correlation_penalty']:.4f}")
print(f" QUBO value: {milp_metrics['qubo_value']:.4f}")
print(f" Feasible: {milp_metrics['is_feasible']}")

# Notebook cell
# Compile all results
all_results = [sa_result, milp_result]
if exhaustive_result is not None:
 all_results.insert(0, exhaustive_result)

# Create comparison table
print("\n" + "=" * 80)
print("SOLVER COMPARISON")
print("=" * 80)

comparison_rows = []
for result in all_results:
 comparison_rows.append({
 'Solver': result['solver'],
 'Selected Pairs': result['metrics']['n_selected'],
 'Alpha Sum': f"{result['metrics']['alpha_term']:.4f}",
 'Corr Penalty': f"{result['metrics']['correlation_penalty']:.4f}",
 'QUBO Value': f"{result['metrics']['qubo_value']:.4f}",
 'Feasible': result['metrics']['is_feasible'],
 'Runtime (s)': f"{result['runtime']:.4f}"
 })

comparison_df = pd.DataFrame(comparison_rows)
print(comparison_df.to_string(index=False))

# Notebook cell
# Sensitivity: Vary lambda (correlation penalty)
print("\nSensitivity Analysis: Varying Lambda (Correlation Penalty)")
print("=" * 80)

lambda_values = np.linspace(0.1, 2.0, 10)
lambda_sensitivity = []

for lambda_val in lambda_values:
 Q_lambda = build_qubo_matrix(pairs_data, K, lambda_val, P)
 
 # Quick SA solution
 x_solution = numpy_simulated_annealing(Q_lambda, alpha_scores, n, K, lambda_val, P, 
 iterations=500)
 metrics = evaluate_solution(x_solution, alpha_scores, corr_matrix, lambda_val, K, P)
 
 lambda_sensitivity.append({
 'lambda': lambda_val,
 'n_selected': metrics['n_selected'],
 'alpha_sum': metrics['alpha_term'],
 'corr_penalty': metrics['correlation_penalty'],
 'qubo_value': metrics['qubo_value']
 })

lambda_sensitivity_df = pd.DataFrame(lambda_sensitivity)
print(lambda_sensitivity_df.to_string(index=False))

# Notebook cell
# Sensitivity: Vary K (max pairs)
print("\nSensitivity Analysis: Varying K (Max Pairs to Select)")
print("=" * 80)

K_values = range(5, 21)
K_sensitivity = []

for K_val in K_values:
 Q_K = build_qubo_matrix(pairs_data, K_val, lambda_corr, P)
 
 # Quick SA solution
 x_solution = numpy_simulated_annealing(Q_K, alpha_scores, n, K_val, lambda_corr, P, 
 iterations=500)
 metrics = evaluate_solution(x_solution, alpha_scores, corr_matrix, lambda_corr, K_val, P)
 
 K_sensitivity.append({
 'K': K_val,
 'n_selected': metrics['n_selected'],
 'alpha_sum': metrics['alpha_term'],
 'corr_penalty': metrics['correlation_penalty'],
 'qubo_value': metrics['qubo_value']
 })

K_sensitivity_df = pd.DataFrame(K_sensitivity)
print(K_sensitivity_df.to_string(index=False))

# Notebook cell
# Plot sensitivity results
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Lambda sensitivity
axes[0, 0].plot(lambda_sensitivity_df['lambda'], lambda_sensitivity_df['n_selected'], 
 'o-', linewidth=2, markersize=8)
axes[0, 0].set_xlabel('Lambda (Correlation Penalty)', fontsize=11)
axes[0, 0].set_ylabel('Number of Selected Pairs', fontsize=11)
axes[0, 0].set_title('Lambda Sensitivity: Portfolio Size', fontsize=12, fontweight='bold')
axes[0, 0].grid(True, alpha=0.3)

axes[0, 1].plot(lambda_sensitivity_df['lambda'], lambda_sensitivity_df['alpha_sum'], 
 'o-', color='green', linewidth=2, markersize=8, label='Alpha Sum')
axes[0, 1].plot(lambda_sensitivity_df['lambda'], lambda_sensitivity_df['corr_penalty'], 
 's--', color='red', linewidth=2, markersize=8, label='Corr Penalty')
axes[0, 1].set_xlabel('Lambda (Correlation Penalty)', fontsize=11)
axes[0, 1].set_ylabel('Value', fontsize=11)
axes[0, 1].set_title('Lambda Sensitivity: Objectives', fontsize=12, fontweight='bold')
axes[0, 1].legend
axes[0, 1].grid(True, alpha=0.3)

# K sensitivity
axes[1, 0].plot(K_sensitivity_df['K'], K_sensitivity_df['n_selected'], 
 'o-', color='purple', linewidth=2, markersize=8)
axes[1, 0].set_xlabel('K (Max Pairs)', fontsize=11)
axes[1, 0].set_ylabel('Number of Selected Pairs', fontsize=11)
axes[1, 0].set_title('K Sensitivity: Portfolio Size', fontsize=12, fontweight='bold')
axes[1, 0].grid(True, alpha=0.3)

axes[1, 1].plot(K_sensitivity_df['K'], K_sensitivity_df['qubo_value'], 
 'o-', color='darkblue', linewidth=2, markersize=8)
axes[1, 1].set_xlabel('K (Max Pairs)', fontsize=11)
axes[1, 1].set_ylabel('QUBO Value', fontsize=11)
axes[1, 1].set_title('K Sensitivity: Objective Value', fontsize=12, fontweight='bold')
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout
plt.savefig(os.path.join(data_dir, 'sensitivity_analysis.png'), dpi=150, bbox_inches='tight')
plt.show

print("Sensitivity analysis plots saved.")

# Notebook cell
# Use best solution (from SA) for visualization
best_solution = sa_result['solution']
selected_indices = np.where(best_solution == 1)[0]
selected_pairs = pairs_data.iloc[selected_indices]

print(f"\nBest Solution from {sa_result['solver']}:")
print(f"Selected {len(selected_indices)} pairs out of {n}")
print("\nSelected Pairs:")
print(selected_pairs[['pair_id', 'asset_1', 'asset_2', 'alpha', 'correlation', 'half_life']])

# Notebook cell
# Visualization 1: Alpha scores - selected vs rejected
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Bar chart of alpha scores
colors = ['green' if i in selected_indices else 'lightgray' for i in range(n)]
axes[0].bar(range(n), pairs_data['alpha'].values, color=colors, edgecolor='black', linewidth=0.5)
axes[0].set_xlabel('Pair Index', fontsize=11)
axes[0].set_ylabel('Alpha Score', fontsize=11)
axes[0].set_title('Alpha Scores: Selected (green) vs Rejected (gray)', fontsize=12, fontweight='bold')
axes[0].axhline(y=pairs_data['alpha'].mean, color='blue', linestyle='--', linewidth=2, label='Mean')
axes[0].legend
axes[0].grid(True, alpha=0.3, axis='y')

# Box plot comparison
selected_alpha = pairs_data.iloc[selected_indices]['alpha'].values
rejected_alpha = pairs_data.iloc[~pairs_data.index.isin(selected_indices)]['alpha'].values

bp = axes[1].boxplot([selected_alpha, rejected_alpha], labels=['Selected', 'Rejected'], patch_artist=True)
for patch, color in zip(bp['boxes'], ['green', 'lightgray']):
 patch.set_facecolor(color)
 patch.set_edgecolor('black')
axes[1].set_ylabel('Alpha Score', fontsize=11)
axes[1].set_title('Alpha Distribution: Selected vs Rejected', fontsize=12, fontweight='bold')
axes[1].grid(True, alpha=0.3, axis='y')

plt.tight_layout
plt.savefig(os.path.join(data_dir, 'alpha_analysis.png'), dpi=150, bbox_inches='tight')
plt.show

print(f"Selected pairs - Mean alpha: {selected_alpha.mean:.4f}")
print(f"Rejected pairs - Mean alpha: {rejected_alpha.mean:.4f}")

# Notebook cell
# Visualization 2: Network graph of selected pairs and correlations
try:
 import networkx as nx
 
 # Create network graph
 G = nx.Graph
 
 # Add nodes for selected pairs
 for idx in selected_indices:
 G.add_node(idx, label=pairs_data.iloc[idx]['pair_id'])
 
 # Add edges for correlations between selected pairs
 for i, idx_i in enumerate(selected_indices):
 for j, idx_j in enumerate(selected_indices):
 if i < j:
 corr_val = corr_matrix[idx_i, idx_j]
 if corr_val > 0.2: # Only show significant correlations
 G.add_edge(idx_i, idx_j, weight=corr_val)
 
 # Draw network
 fig, ax = plt.subplots(figsize=(12, 10))
 
 pos = nx.spring_layout(G, k=2, iterations=50, seed=42)
 
 # Node sizes based on alpha
 node_sizes = [pairs_data.iloc[idx]['alpha'] * 3000 for idx in G.nodes]
 
 # Draw nodes
 nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color='lightgreen', 
 edgecolors='darkgreen', linewidths=2, ax=ax)
 
 # Draw edges with width proportional to correlation
 edges = G.edges
 edge_widths = [G[u][v]['weight'] * 3 for u, v in edges]
 nx.draw_networkx_edges(G, pos, width=edge_widths, alpha=0.6, edge_color='gray', ax=ax)
 
 # Draw labels
 labels = {idx: pairs_data.iloc[idx]['pair_id'] for idx in G.nodes}
 nx.draw_networkx_labels(G, pos, labels, font_size=9, font_weight='bold', ax=ax)
 
 ax.set_title('Network of Selected Pairs\n(Node size = alpha score, Edge width = correlation)', 
 fontsize=12, fontweight='bold')
 ax.axis('off')
 
 plt.tight_layout
 plt.savefig(os.path.join(data_dir, 'pair_network_graph.png'), dpi=150, bbox_inches='tight')
 plt.show
 
 print("Network graph visualization complete.")
 
except ImportError:
 print("networkx not installed. Skipping network graph visualization.")

# Notebook cell
# Visualization 3: Solver comparison table with visual elements
fig, ax = plt.subplots(figsize=(12, 4))
ax.axis('tight')
ax.axis('off')

# Prepare table data
table_data = []
headers = ['Solver', 'Pairs', 'Alpha Sum', 'Corr Penalty', 'QUBO Value', 'Feasible', 'Time (s)']

for result in all_results:
 table_data.append([
 result['solver'],
 f"{result['metrics']['n_selected']}",
 f"{result['metrics']['alpha_term']:.3f}",
 f"{result['metrics']['correlation_penalty']:.3f}",
 f"{result['metrics']['qubo_value']:.3f}",
 '✓' if result['metrics']['is_feasible'] else '✗',
 f"{result['runtime']:.4f}"
 ])

table = ax.table(cellText=table_data, colLabels=headers, cellLoc='center', loc='center',
 colWidths=[0.2, 0.08, 0.12, 0.15, 0.15, 0.08, 0.12])

table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1, 2)

# Color header
for i in range(len(headers)):
 table[(0, i)].set_facecolor('#4CAF50')
 table[(0, i)].set_text_props(weight='bold', color='white')

# Alternate row colors
for i in range(1, len(table_data) + 1):
 for j in range(len(headers)):
 if i % 2 == 0:
 table[(i, j)].set_facecolor('#f0f0f0')
 else:
 table[(i, j)].set_facecolor('#ffffff')
 table[(i, j)].set_edgecolor('#cccccc')

plt.title('Solver Performance Comparison', fontsize=13, fontweight='bold', pad=20)
plt.tight_layout
plt.savefig(os.path.join(data_dir, 'solver_comparison_table.png'), dpi=150, bbox_inches='tight')
plt.show

print("Solver comparison visualization saved.")

# Notebook cell
# Ensure results directory exists
results_dir = os.path.join(os.getcwd, '..', 'results')
os.makedirs(results_dir, exist_ok=True)
os.makedirs(data_dir, exist_ok=True)

print(f"Exporting results to {results_dir}")
print(f"Exporting data to {data_dir}")

# Notebook cell
# Export QUBO matrix
qubo_path = os.path.join(data_dir, 'qubo_matrix.npy')
np.save(qubo_path, Q)
print(f"QUBO matrix saved: {qubo_path}")

# Export QUBO metadata
qubo_metadata = {
 'shape': Q.shape,
 'K': K,
 'lambda_corr': lambda_corr,
 'P': P,
 'n_pairs': n,
 'diagonal_mean': float(np.mean(np.diag(Q))),
 'diagonal_std': float(np.std(np.diag(Q)))
}

metadata_path = os.path.join(data_dir, 'qubo_metadata.json')
with open(metadata_path, 'w') as f:
 json.dump(qubo_metadata, f, indent=2)
print(f"QUBO metadata saved: {metadata_path}")

# Notebook cell
# Export all solutions to JSON
solutions_export = []

for result in all_results:
 solution_data = {
 'solver': result['solver'],
 'solution_vector': result['solution'].tolist,
 'selected_pair_indices': np.where(result['solution'] == 1)[0].tolist,
 'metrics': {
 'n_selected': int(result['metrics']['n_selected']),
 'alpha_sum': float(result['metrics']['alpha_term']),
 'correlation_penalty': float(result['metrics']['correlation_penalty']),
 'objective_value': float(result['metrics']['objective']),
 'qubo_value': float(result['metrics']['qubo_value']),
 'is_feasible': bool(result['metrics']['is_feasible']),
 'constraint_violation': int(result['metrics']['constraint_violation'])
 },
 'runtime_seconds': float(result['runtime'])
 }
 solutions_export.append(solution_data)

solutions_path = os.path.join(results_dir, 'qubo_solutions.json')
with open(solutions_path, 'w') as f:
 json.dump(solutions_export, f, indent=2)
print(f"Solutions exported: {solutions_path}")

# Notebook cell
# Export selected pairs details
selected_pairs_export = selected_pairs.copy
selected_pairs_export['selected'] = True

# Add all pairs with selection status
all_pairs_with_status = pairs_data.copy
all_pairs_with_status['selected'] = all_pairs_with_status.index.isin(selected_indices)

pairs_status_path = os.path.join(results_dir, 'pair_selection_status.csv')
all_pairs_with_status.to_csv(pairs_status_path, index=False)
print(f"Pair selection status saved: {pairs_status_path}")

# Summary statistics
summary = {
 'total_pairs': int(n),
 'selected_pairs': int(len(selected_indices)),
 'selection_ratio': float(len(selected_indices) / n),
 'best_solver': all_results[np.argmax([r['metrics']['qubo_value'] for r in all_results])]['solver'],
 'best_qubo_value': float(max([r['metrics']['qubo_value'] for r in all_results])),
 'avg_selected_alpha': float(selected_alpha.mean),
 'avg_rejected_alpha': float(rejected_alpha.mean),
 'max_selected_alpha': float(selected_alpha.max),
 'min_selected_alpha': float(selected_alpha.min)
}

summary_path = os.path.join(results_dir, 'qubo_summary.json')
with open(summary_path, 'w') as f:
 json.dump(summary, f, indent=2)
print(f"Summary saved: {summary_path}")

# Notebook cell
# Final report
print("\n" + "="*80)
print("QUBO FORMULATION PHASE - COMPLETE")
print("="*80)
print(f"\nProject: Asymmetrica Investments AG - Quantum Signal Generation")
print(f"Phase: Phase 0 - QUBO Formulation for Pair Selection")
print(f"\nProblem Summary:")
print(f" Candidate pairs: {n}")
print(f" Max portfolio size (K): {K}")
 print(f" Lambda (correlation penalty): {lambda_corr}")
print(f" Penalty weight (P): {P}")
print(f"\nOptimization Results:")
print(f" Best solver: {summary['best_solver']}")
print(f" Selected pairs: {summary['selected_pairs']}")
print(f" Selection ratio: {summary['selection_ratio']:.1%}")
print(f" Best QUBO value: {summary['best_qubo_value']:.4f}")
print(f" Avg alpha (selected): {summary['avg_selected_alpha']:.4f}")
print(f" Avg alpha (rejected): {summary['avg_rejected_alpha']:.4f}")
print(f"\nExported Files:")
print(f" - QUBO matrix: {qubo_path}")
print(f" - Solutions: {solutions_path}")
print(f" - Pair status: {pairs_status_path}")
print(f" - Summary: {summary_path}")
print(f"\nVisualizations:")
print(f" - QUBO matrix heatmap")
print(f" - Sensitivity analysis plots")
print(f" - Alpha score analysis")
print(f" - Pair network graph")
print(f" - Solver comparison table")
print("\n" + "="*80)

# Notebook cell


# Notebook cell


# Notebook cell

