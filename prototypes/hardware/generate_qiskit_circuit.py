#!/usr/bin/env python3
"""
Generate a real Qiskit circuit diagram for the 14-qubit VQC v4.
This produces a publication-quality circuit image for the IBM presentation.

Usage:
    pip install qiskit qiskit-aer matplotlib pylatexenc
    python scripts/generate_qiskit_circuit.py

Output: circuit_diagram_qiskit.png (in the phase0 directory)
"""

from qiskit import QuantumCircuit
from qiskit.circuit import Parameter
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

# ── Our actual VQC v4 architecture ──
# 14 qubits, 2 variational layers, domain-aware cross-entanglement
# 57 params: 14 encoding scales + 28 variational (2 layers × 14 qubits) + 14 output weights + 1 bias

N_QUBITS = 14
N_LAYERS = 2

# Domain-aware cross-entanglement pairs (from circuit_v4.py)
# These connect related financial features across the circuit
CROSS_PAIRS = [
    (0, 7),   # momentum ↔ event_risk
    (1, 8),   # mom_accel ↔ sentiment
    (2, 5),   # volatility ↔ carry
    (3, 10),  # vol_regime ↔ COT_positioning
    (4, 12),  # ATR ↔ DXY_momentum
    (5, 13),  # carry ↔ risk_sentiment
    (6, 9),   # yield_curve ↔ sent_momentum
    (7, 11),  # event_risk ↔ COT_extreme
    (8, 3),   # sentiment ↔ vol_regime
    (9, 6),   # sent_momentum ↔ yield_curve
]

# Feature names for qubit labels
FEATURE_NAMES = [
    'momentum', 'mom_accel', 'volatility', 'vol_regime', 'ATR',
    'carry_diff', 'yield_curve', 'event_risk', 'sentiment',
    'sent_momentum', 'COT_pos', 'COT_extreme', 'DXY_mom', 'risk_sent'
]

def build_vqc_v4_circuit():
    """Build the actual VQC v4 circuit in Qiskit."""
    qc = QuantumCircuit(N_QUBITS, name='VQC_v4')

    # ── ENCODING BLOCK ──
    # Rx gates with trainable encoding scales: Rx(scale_i * x_i)
    encoding_params = [Parameter(f'α_{i}') for i in range(N_QUBITS)]
    for i in range(N_QUBITS):
        qc.rx(encoding_params[i], i)

    qc.barrier(label='encode')

    for layer in range(N_LAYERS):
        # ── VARIATIONAL BLOCK ──
        # Ry and Rz rotations on each qubit
        ry_params = [Parameter(f'θ_{layer}_{i}') for i in range(N_QUBITS)]
        rz_params = [Parameter(f'φ_{layer}_{i}') for i in range(N_QUBITS)]

        for i in range(N_QUBITS):
            qc.ry(ry_params[i], i)
            qc.rz(rz_params[i], i)

        qc.barrier(label=f'var_L{layer+1}')

        # ── ENTANGLEMENT ──
        if layer == 0:
            # Layer 1: Adjacent CNOT chain (nearest-neighbor)
            for i in range(0, N_QUBITS - 1, 2):
                qc.cx(i, i + 1)
            for i in range(1, N_QUBITS - 1, 2):
                qc.cx(i, i + 1)

            qc.barrier(label='adj_ent')

            # Domain-aware cross-entanglement
            for q1, q2 in CROSS_PAIRS:
                qc.cx(q1, q2)

            qc.barrier(label='cross_ent')
        else:
            # Layer 2: Adjacent CNOT chain only
            for i in range(0, N_QUBITS - 1, 2):
                qc.cx(i, i + 1)
            for i in range(1, N_QUBITS - 1, 2):
                qc.cx(i, i + 1)

            qc.barrier(label='adj_ent')

    # ── MEASUREMENT ──
    qc.measure_all()

    return qc


def main():
    qc = build_vqc_v4_circuit()

    print(f"Circuit built:")
    print(f"  Qubits: {qc.num_qubits}")
    print(f"  Parameters: {qc.num_parameters}")
    print(f"  Depth: {qc.depth()}")
    print(f"  Gates: {dict(qc.count_ops())}")

    # Draw with dark style for the presentation
    fig = qc.draw(
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
                'cx': ('#1565C0', '#42A5F5'),
                'measure': ('#B71C1C', '#EF5350'),
            },
            'fontsize': 10,
            'subfontsize': 8,
        },
        fold=-1,  # Don't fold the circuit
        scale=0.6,
        plot_barriers=True,
    )

    # Resize for presentation
    fig.set_size_inches(22, 10)
    fig.savefig('circuit_diagram_qiskit.png',
                dpi=150,
                facecolor='#0A1628',
                bbox_inches='tight',
                pad_inches=0.3)

    print(f"\nSaved: circuit_diagram_qiskit.png")
    print(f"Use this image in slide 3 of the IBM presentation.")

    # Also save a light version (white background) as backup
    fig2 = qc.draw(
        output='mpl',
        fold=-1,
        scale=0.6,
        plot_barriers=True,
    )
    fig2.set_size_inches(22, 10)
    fig2.savefig('circuit_diagram_qiskit_light.png',
                 dpi=150,
                 bbox_inches='tight',
                 pad_inches=0.3)

    print(f"Saved (light version): circuit_diagram_qiskit_light.png")


if __name__ == '__main__':
    main()
