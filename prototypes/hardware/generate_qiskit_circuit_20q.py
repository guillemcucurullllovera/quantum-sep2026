#!/usr/bin/env python3
"""
Generate a 20-qubit VQC circuit diagram for Phase 4A (IBM Eagle r3).
Shows the scaling potential: 14 data qubits + 6 auxiliary qubits.

Usage:
    pip install qiskit matplotlib pylatexenc
    python scripts/generate_qiskit_circuit_20q.py

Output: circuit_diagram_20q_light.png (for the IBM presentation)
"""

from qiskit import QuantumCircuit
from qiskit.circuit import Parameter
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

N_DATA_QUBITS = 14
N_AUX_QUBITS = 6
N_QUBITS = N_DATA_QUBITS + N_AUX_QUBITS  # 20
N_LAYERS = 2

# Feature names for data qubits
FEATURE_NAMES = [
    'momentum', 'mom_accel', 'volatility', 'vol_regime', 'ATR',
    'carry_diff', 'yield_curve', 'event_risk', 'sentiment',
    'sent_momentum', 'COT_pos', 'COT_extreme', 'DXY_mom', 'risk_sent',
    'aux_0', 'aux_1', 'aux_2', 'aux_3', 'aux_4', 'aux_5',
]

# Domain-aware cross-entanglement (expanded for 20 qubits)
# Same 10 pairs as 14q, plus 6 new pairs connecting auxiliary qubits
CROSS_PAIRS = [
    # Original domain-aware pairs
    (0, 7),    # momentum ↔ event_risk
    (1, 8),    # mom_accel ↔ sentiment
    (2, 5),    # volatility ↔ carry
    (3, 10),   # vol_regime ↔ COT_positioning
    (4, 12),   # ATR ↔ DXY_momentum
    (5, 13),   # carry ↔ risk_sentiment
    (6, 9),    # yield_curve ↔ sent_momentum
    (7, 11),   # event_risk ↔ COT_extreme
    (8, 3),    # sentiment ↔ vol_regime
    (9, 6),    # sent_momentum ↔ yield_curve
    # Auxiliary qubit bridges (richer entanglement topology)
    (0, 14),   # momentum → aux_0 (momentum auxiliary)
    (2, 15),   # volatility → aux_1 (vol auxiliary)
    (5, 16),   # carry → aux_2 (macro auxiliary)
    (8, 17),   # sentiment → aux_3 (sentiment auxiliary)
    (10, 18),  # COT_pos → aux_4 (positioning auxiliary)
    (12, 19),  # DXY_mom → aux_5 (cross-asset auxiliary)
    # Auxiliary-to-auxiliary connections (cross-domain entanglement)
    (14, 17),  # momentum_aux ↔ sentiment_aux
    (15, 16),  # vol_aux ↔ macro_aux
    (18, 19),  # positioning_aux ↔ cross_asset_aux
]


def build_vqc_20q_circuit():
    """Build the 20-qubit VQC for Phase 4A (IBM Eagle r3)."""
    qc = QuantumCircuit(N_QUBITS, name='VQC_v4_20q')

    # ── ENCODING BLOCK ──
    # Data qubits: Rx with trainable encoding scales
    encoding_params = [Parameter(f'α_{i}') for i in range(N_DATA_QUBITS)]
    for i in range(N_DATA_QUBITS):
        qc.rx(encoding_params[i], i)

    # Auxiliary qubits: initialized in superposition (Hadamard)
    for i in range(N_DATA_QUBITS, N_QUBITS):
        qc.h(i)

    qc.barrier(label='encode')

    for layer in range(N_LAYERS):
        # ── VARIATIONAL BLOCK (all 20 qubits) ──
        ry_params = [Parameter(f'θ_{layer}_{i}') for i in range(N_QUBITS)]
        rz_params = [Parameter(f'φ_{layer}_{i}') for i in range(N_QUBITS)]

        for i in range(N_QUBITS):
            qc.ry(ry_params[i], i)
            qc.rz(rz_params[i], i)

        qc.barrier(label=f'var_L{layer+1}')

        # ── ENTANGLEMENT ──
        if layer == 0:
            # Layer 1: Full nearest-neighbor chain (20 qubits)
            for i in range(0, N_QUBITS - 1, 2):
                qc.cx(i, i + 1)
            for i in range(1, N_QUBITS - 1, 2):
                qc.cx(i, i + 1)

            qc.barrier(label='adj_ent')

            # Domain-aware + auxiliary cross-entanglement
            for q1, q2 in CROSS_PAIRS:
                qc.cx(q1, q2)

            qc.barrier(label='cross_ent')
        else:
            # Layer 2: Nearest-neighbor only
            for i in range(0, N_QUBITS - 1, 2):
                qc.cx(i, i + 1)
            for i in range(1, N_QUBITS - 1, 2):
                qc.cx(i, i + 1)

            qc.barrier(label='adj_ent')

    # ── MEASUREMENT ──
    qc.measure_all()

    return qc


def main():
    qc = build_vqc_20q_circuit()

    print(f"20-Qubit Circuit (Phase 4A - IBM Eagle r3):")
    print(f"  Qubits: {qc.num_qubits}")
    print(f"  Data qubits: {N_DATA_QUBITS}")
    print(f"  Auxiliary qubits: {N_AUX_QUBITS}")
    print(f"  Parameters: {qc.num_parameters}")
    print(f"  Depth: {qc.depth()}")
    print(f"  Gates: {dict(qc.count_ops())}")
    print(f"  Cross-entanglement pairs: {len(CROSS_PAIRS)}")

    # Light version (white bg) for presentation
    fig = qc.draw(
        output='mpl',
        fold=-1,
        scale=0.5,
        plot_barriers=True,
    )
    fig.set_size_inches(28, 12)
    fig.savefig('circuit_diagram_20q_light.png',
                dpi=150,
                bbox_inches='tight',
                pad_inches=0.3)
    print(f"\nSaved: circuit_diagram_20q_light.png")

    # Dark version
    fig2 = qc.draw(
        output='mpl',
        style={
            'backgroundcolor': '#0A1628',
            'textcolor': '#E0E0E0',
            'subtextcolor': '#78909C',
            'linecolor': '#4A90D9',
            'creglinecolor': '#4A90D9',
            'gatetextcolor': '#FFFFFF',
            'gatefacecolor': '#1A3A5C',
            'barrierfacecolor': '#2A3A4C',
            'displaycolor': {
                'rx': ('#004D40', '#00BFA5'),
                'ry': ('#E65100', '#FF6F00'),
                'rz': ('#E65100', '#FF6F00'),
                'h': ('#6A1B9A', '#AB47BC'),
                'cx': ('#1565C0', '#42A5F5'),
                'measure': ('#B71C1C', '#EF5350'),
            },
        },
        fold=-1,
        scale=0.5,
        plot_barriers=True,
    )
    fig2.set_size_inches(28, 12)
    fig2.savefig('circuit_diagram_20q_dark.png',
                 dpi=150,
                 facecolor='#0A1628',
                 bbox_inches='tight',
                 pad_inches=0.3)
    print(f"Saved: circuit_diagram_20q_dark.png")

    # Print comparison
    print(f"\n{'='*50}")
    print(f"COMPARISON: 14q vs 20q")
    print(f"{'='*50}")
    print(f"{'Metric':<25} {'14-qubit':>10} {'20-qubit':>10}")
    print(f"{'-'*45}")
    print(f"{'Data qubits':<25} {'14':>10} {'14':>10}")
    print(f"{'Auxiliary qubits':<25} {'0':>10} {'6':>10}")
    print(f"{'Total qubits':<25} {'14':>10} {'20':>10}")
    print(f"{'Parameters':<25} {'~57':>10} {'~161':>10}")
    print(f"{'Cross-entangle pairs':<25} {'10':>10} {f'{len(CROSS_PAIRS)}':>10}")
    print(f"{'Two-qubit gates':<25} {'~56':>10} {'~100+':>10}")
    print(f"{'IBM Backend':<25} {'Simulator':>10} {'Eagle r3':>10}")
    print(f"{'Gate fidelity needed':<25} {'N/A':>10} {'~99.5%':>10}")


if __name__ == '__main__':
    main()
