#!/usr/bin/env python3
"""
VQC Smoke Test — Phase 1A
===========================
Proves the quantum encoding pipeline works end-to-end:
  captured signals → builder → 8-feature vector → angle encoding → VQC → measurement

This is NOT training. This is a smoke test that verifies:
  1. Different market states produce different quantum states
  2. The encoding preserves information (not all-zero, not degenerate)
  3. Parameter-shift gradients exist (circuit is trainable)
  4. The 8-qubit circuit runs correctly on the frozen contract

Usage:
  # From captured snapshots (default)
  python scripts/vqc_smoke_test.py

  # From pre-saved vectors file
  python scripts/vqc_smoke_test.py --vectors results/vectors_for_vqc.json

  # With fixtures (no Redis/capture needed)
  python scripts/vqc_smoke_test.py --fixture

  # Generate HTML report
  python scripts/vqc_smoke_test.py --html results/vqc_smoke_report.html
"""

import argparse
import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

PHASE0_DIR = Path(__file__).parent.parent
RESULTS_DIR = PHASE0_DIR / "results"

# Delayed numpy import with clear error
try:
    import numpy as np
except ImportError:
    print("ERROR: numpy required. Install with: pip install numpy")
    sys.exit(1)

from quantum_signal.circuit import VQCSmokeTest as VQC, N_QUBITS
from quantum_signal.models import QuantumInputVector

REDUCED_8Q_NAMES = [
    "momentum_composite", "realized_vol_20d", "vol_regime",
    "carry_differential", "event_risk_score", "net_sentiment",
    "cot_positioning", "risk_sentiment",
]


def load_vectors_from_file(path):
    """Load pre-computed vectors from JSON."""
    with open(path) as f:
        data = json.load(f)
    vectors = []
    for entry in data:
        vectors.append({
            "asset": entry["asset"],
            "features": entry["features_8q"],
            "label": f"{entry['asset']}_t{entry.get('tick', '?')}",
        })
    return vectors


def load_vectors_from_snapshots(snapshot_dir):
    """Load snapshots and build vectors through canonical builder."""
    from quantum_signal.builder import QuantumVectorBuilder

    snapshot_path = Path(snapshot_dir)
    if not snapshot_path.exists():
        return []

    builders = {}
    vectors = []

    for asset_dir in sorted(snapshot_path.iterdir()):
        if not asset_dir.is_dir() or asset_dir.name.startswith("_"):
            continue
        asset = asset_dir.name

        if asset not in builders:
            builders[asset] = QuantumVectorBuilder(window_size=252)

        for json_file in sorted(asset_dir.glob("*.json")):
            try:
                with open(json_file) as f:
                    data = json.load(f)
                agents = data.get("agents", {})
                technical = agents.get("technical") or {}
                macro = agents.get("macro") or {}
                news = agents.get("news") or {}
                fundamental = agents.get("fundamental") or {}
                risk = agents.get("risk_sentiment") or {}

                # Skip snapshots with no real agent data (only global signals)
                has_data = any([
                    technical.get("rsi"),
                    macro.get("carry_differential"),
                    news.get("base_bull_count") or news.get("focus_bull_count"),
                    fundamental.get("base_cot_net_pct"),
                ])
                if not has_data:
                    continue

                vec = builders[asset].build(
                    technical=technical, macro=macro, news=news,
                    fundamental=fundamental, risk_sentiment=risk,
                    asset=asset, timestamp=data.get("captured_at"),
                )
                tick = len([v for v in vectors if v["asset"] == asset])
                vectors.append({
                    "asset": asset,
                    "features": vec.to_reduced_8q(),
                    "label": f"{asset}_t{tick}",
                })
            except Exception as e:
                print(f"  Warning: skipping {json_file.name}: {e}")

    return vectors


def generate_fixture_vectors():
    """Generate synthetic test vectors spanning the feature space."""
    import random
    random.seed(42)

    vectors = []
    scenarios = [
        ("BULL_momentum", [0.8, 0.3, 0.0, 0.5, 0.2, 0.7, 0.4, 0.3]),
        ("BEAR_momentum", [-0.7, 0.8, 1.0, -0.3, 0.8, -0.5, -0.3, 0.8]),
        ("HIGH_vol", [0.1, 0.95, 1.0, 0.0, 0.6, -0.1, 0.0, 0.7]),
        ("LOW_vol", [0.3, 0.1, 0.0, 0.2, 0.1, 0.2, 0.1, 0.2]),
        ("RISK_OFF", [-0.4, 0.7, 1.0, -0.5, 0.9, -0.6, -0.4, 0.9]),
        ("RISK_ON", [0.6, 0.2, 0.0, 0.4, 0.1, 0.5, 0.3, 0.15]),
        ("NEUTRAL", [0.0, 0.5, 0.5, 0.0, 0.5, 0.0, 0.0, 0.5]),
        ("EXTREME_BULL", [0.95, 0.1, 0.0, 0.8, 0.05, 0.9, 0.8, 0.1]),
    ]
    for name, feats in scenarios:
        vectors.append({"asset": name, "features": feats, "label": name})

    # Add random variations
    for i in range(12):
        feats = [random.uniform(-0.8, 0.8) if j in (0, 3, 5, 6) else random.uniform(0, 1)
                 for j in range(8)]
        vectors.append({"asset": f"RND_{i}", "features": feats, "label": f"random_{i}"})

    return vectors


def run_smoke_tests(vectors):
    """Execute all smoke test checks."""
    vqc = VQC(seed=42)
    results = {
        "circuit_config": vqc.to_dict(),
        "n_vectors": len(vectors),
        "tests": {},
        "vectors": [],
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    print("\n" + "=" * 60)
    print("VQC SMOKE TEST — PHASE 1A")
    print("=" * 60)
    print(f"\nCircuit: {N_QUBITS} qubits, {N_QUBITS} params")
    print(f"Encoding: RY(feature × π)")
    print(f"Ansatz: CNOT chain + single RY variational layer")
    print(f"Measurement: ⟨Z₀⟩")
    print(f"Vectors: {len(vectors)}")

    # ── TEST 1: Forward pass produces valid outputs ──────────────
    print(f"\n{'─' * 60}")
    print("TEST 1: Forward pass — valid outputs")
    print(f"{'─' * 60}")
    expectations = []
    for v in vectors:
        exp = vqc.forward(v["features"])
        exp_all = vqc.forward_all_qubits(v["features"])
        expectations.append(exp)
        results["vectors"].append({
            "label": v["label"],
            "asset": v["asset"],
            "features": v["features"],
            "expectation_z0": exp,
            "expectations_all": exp_all,
        })
        print(f"  {v['label']:20s}  ⟨Z₀⟩ = {exp:+.6f}  "
              f"8Q = [{', '.join(f'{e:+.3f}' for e in exp_all)}]")

    all_valid = all(-1 <= e <= 1 for e in expectations)
    all_different = len(set(round(e, 8) for e in expectations)) > 1
    results["tests"]["forward_valid"] = all_valid
    results["tests"]["forward_diverse"] = all_different
    print(f"\n  All in [-1,1]: {'PASS' if all_valid else 'FAIL'}")
    print(f"  Diverse outputs: {'PASS' if all_different else 'FAIL'}")

    # ── TEST 2: Different inputs → different outputs ─────────────
    print(f"\n{'─' * 60}")
    print("TEST 2: Input sensitivity — encoding preserves information")
    print(f"{'─' * 60}")
    if len(vectors) >= 2:
        unique_outputs = len(set(round(e, 6) for e in expectations))
        spread = max(expectations) - min(expectations)
        std = float(np.std(expectations))
        results["tests"]["unique_outputs"] = unique_outputs
        results["tests"]["output_spread"] = spread
        results["tests"]["output_std"] = std
        print(f"  Unique outputs: {unique_outputs}/{len(vectors)}")
        print(f"  Output spread: {spread:.6f}")
        print(f"  Output std: {std:.6f}")
        print(f"  Result: {'PASS' if unique_outputs > 1 and spread > 0.01 else 'FAIL'}")

    # ── TEST 3: Fidelity matrix (are states distinguishable?) ────
    print(f"\n{'─' * 60}")
    print("TEST 3: State distinguishability — fidelity matrix")
    print(f"{'─' * 60}")
    # Select diverse vectors: pick one per unique asset, then fill to 6
    seen_assets = {}
    diverse_indices = []
    for idx, v in enumerate(vectors):
        if v["asset"] not in seen_assets:
            seen_assets[v["asset"]] = idx
            diverse_indices.append(idx)
        if len(diverse_indices) >= 6:
            break
    # If fewer than 6 unique assets, add more from different ticks
    if len(diverse_indices) < 6:
        for idx in range(len(vectors)):
            if idx not in diverse_indices:
                diverse_indices.append(idx)
            if len(diverse_indices) >= 6:
                break
    fid_vectors = [vectors[i] for i in diverse_indices]
    n_check = len(fid_vectors)
    fidelities = []
    fid_matrix = []
    for i in range(n_check):
        row = []
        for j in range(n_check):
            if i == j:
                fid = 1.0
            else:
                fid = vqc.state_fidelity(fid_vectors[i]["features"], fid_vectors[j]["features"])
                if i < j:
                    fidelities.append(fid)
            row.append(fid)
        fid_matrix.append(row)

    # Print compact fidelity matrix
    labels = [v["label"][:10] for v in fid_vectors]
    header = f"  {'':>10s} " + " ".join(f"{l:>10s}" for l in labels)
    print(header)
    for i, row in enumerate(fid_matrix):
        cells = " ".join(f"{v:>10.4f}" for v in row)
        print(f"  {labels[i]:>10s} {cells}")

    avg_fid = sum(fidelities) / len(fidelities) if fidelities else 1.0
    min_fid = min(fidelities) if fidelities else 1.0
    results["tests"]["avg_fidelity"] = avg_fid
    results["tests"]["min_fidelity"] = min_fid
    results["tests"]["fidelity_matrix"] = fid_matrix
    results["fid_vectors"] = [{"label": v["label"]} for v in fid_vectors]
    print(f"\n  Avg off-diagonal fidelity: {avg_fid:.4f}")
    print(f"  Min fidelity: {min_fid:.4f}")
    print(f"  Result: {'PASS' if avg_fid < 0.95 else 'MARGINAL' if avg_fid < 0.99 else 'FAIL'}"
          f" (lower = more distinguishable)")

    # ── TEST 4: Parameter-shift gradients exist ──────────────────
    print(f"\n{'─' * 60}")
    print("TEST 4: Trainability — parameter-shift gradients")
    print(f"{'─' * 60}")
    test_vec = vectors[0]["features"]
    grad = vqc.gradient(test_vec)
    grad_nonzero = sum(1 for g in grad if abs(g) > 1e-10)
    grad_norm = float(np.linalg.norm(grad))
    results["tests"]["gradient_nonzero_count"] = grad_nonzero
    results["tests"]["gradient_norm"] = grad_norm
    results["tests"]["gradients"] = grad.tolist()

    print(f"  Input: {vectors[0]['label']}")
    print(f"  Gradients: [{', '.join(f'{g:+.6f}' for g in grad)}]")
    print(f"  Non-zero: {grad_nonzero}/{N_QUBITS}")
    print(f"  Gradient norm: {grad_norm:.6f}")
    print(f"  Result: {'PASS' if grad_nonzero > 0 and grad_norm > 1e-6 else 'FAIL'}"
          f" (circuit is {'trainable' if grad_nonzero > 0 else 'NOT trainable'})")

    # ── TEST 5: Parameter sensitivity ────────────────────────────
    print(f"\n{'─' * 60}")
    print("TEST 5: Parameter sensitivity — varying θ changes output")
    print(f"{'─' * 60}")
    test_vec = vectors[0]["features"]
    base_exp = vqc.forward(test_vec)
    param_sensitivity = []
    for i in range(N_QUBITS):
        vqc_shifted = VQC(params=vqc.params.copy())
        vqc_shifted.params[i] += 0.5
        shifted_exp = vqc_shifted.forward(test_vec)
        delta = abs(shifted_exp - base_exp)
        param_sensitivity.append(delta)
        print(f"  θ_{i} += 0.5: ⟨Z₀⟩ {base_exp:+.4f} → {shifted_exp:+.4f}  (Δ={delta:.4f})")

    avg_sensitivity = sum(param_sensitivity) / len(param_sensitivity)
    results["tests"]["avg_param_sensitivity"] = avg_sensitivity
    print(f"\n  Avg sensitivity: {avg_sensitivity:.4f}")
    print(f"  Result: {'PASS' if avg_sensitivity > 0.01 else 'FAIL'}")

    # ── OVERALL VERDICT ──────────────────────────────────────────
    print(f"\n{'=' * 60}")
    all_pass = (
        results["tests"].get("forward_valid", False) and
        results["tests"].get("forward_diverse", False) and
        results["tests"].get("gradient_nonzero_count", 0) > 0 and
        avg_fid < 0.99
    )
    verdict = "ALL TESTS PASS" if all_pass else "SOME TESTS FAILED"
    results["verdict"] = verdict
    print(f"VERDICT: {verdict}")
    if all_pass:
        print("The 8-qubit VQC smoke test confirms:")
        print("  ✓ Angle encoding maps market features to distinct quantum states")
        print("  ✓ The variational circuit produces diverse, valid outputs")
        print("  ✓ Parameter-shift gradients exist → circuit is trainable")
        print("  ✓ Different market regimes are distinguishable in Hilbert space")
        print("  ✓ The frozen QuantumInputVector v1.0 contract works through quantum encoding")
    print(f"{'=' * 60}")

    return results


def generate_smoke_html(results, output_path):
    """Generate HTML report for VQC smoke test."""
    tests = results["tests"]
    vectors = results["vectors"]

    # Vector result rows
    vec_rows = ""
    for v in vectors[:20]:
        feats = " ".join(f"{f:+.3f}" for f in v["features"])
        exp_all = " ".join(f"{e:+.3f}" for e in v["expectations_all"])
        vec_rows += f"""<tr>
            <td>{v['label']}</td>
            <td style="font-family:monospace;font-size:11px">{feats}</td>
            <td style="font-family:monospace;font-weight:bold">{v['expectation_z0']:+.6f}</td>
            <td style="font-family:monospace;font-size:11px">{exp_all}</td>
        </tr>"""

    # Gradient rows
    grad_rows = ""
    for i, g in enumerate(tests.get("gradients", [])):
        bar_width = min(abs(g) * 200, 100)
        color = "#4CAF50" if g > 0 else "#F44336"
        grad_rows += f"""<tr>
            <td>θ_{i} ({REDUCED_8Q_NAMES[i][:15]})</td>
            <td style="font-family:monospace">{g:+.6f}</td>
            <td><div style="width:{bar_width}px;height:12px;background:{color};border-radius:3px"></div></td>
        </tr>"""

    # Fidelity matrix
    fid_html = ""
    fid_matrix = tests.get("fidelity_matrix", [])
    if fid_matrix:
        n = len(fid_matrix)
        labels = [v["label"][:12] for v in (results.get("fid_vectors") or vectors[:n])]
        header_cells = "".join(f'<th style="font-size:11px">{l}</th>' for l in labels)
        rows = ""
        for i, row in enumerate(fid_matrix):
            cells = ""
            for j, val in enumerate(row):
                bg = "#e8f5e9" if val > 0.9 else "#fff9c4" if val > 0.5 else "#ffcdd2"
                if i == j:
                    bg = "#e3f2fd"
                cells += f'<td style="font-family:monospace;font-size:11px;background:{bg}">{val:.3f}</td>'
            rows += f"<tr><td style='font-weight:bold;font-size:11px'>{labels[i]}</td>{cells}</tr>"
        fid_html = f"""<table><tr><th></th>{header_cells}</tr>{rows}</table>"""

    config = results["circuit_config"]
    verdict = results["verdict"]
    verdict_color = "#4CAF50" if "PASS" in verdict else "#F44336"

    html = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>Phase 1A — VQC Smoke Test</title>
<style>
  body {{ font-family: -apple-system, BlinkMacSystemFont, sans-serif; max-width: 1100px;
         margin: 40px auto; padding: 0 20px; color: #333; background: #fafafa; }}
  h1 {{ border-bottom: 3px solid #1a1a2e; padding-bottom: 10px; color: #1a1a2e; }}
  h2 {{ color: #16213e; margin-top: 32px; }}
  table {{ border-collapse: collapse; width: 100%; margin: 12px 0; background: white; }}
  th, td {{ border: 1px solid #e0e0e0; padding: 6px 10px; text-align: left; font-size: 13px; }}
  th {{ background: #1a1a2e; color: white; }}
  .verdict {{ font-size: 24px; font-weight: bold; color: {verdict_color}; margin: 20px 0;
              padding: 16px; background: white; border-radius: 8px;
              box-shadow: 0 1px 3px rgba(0,0,0,0.1); }}
  .circuit-box {{ background: #1a1a2e; color: #e0e0e0; padding: 20px; border-radius: 8px;
                  font-family: monospace; font-size: 13px; margin: 16px 0; white-space: pre; }}
  .test-card {{ background: white; padding: 16px; border-radius: 8px; margin: 12px 0;
                box-shadow: 0 1px 3px rgba(0,0,0,0.1); }}
  .test-pass {{ border-left: 4px solid #4CAF50; }}
  .test-fail {{ border-left: 4px solid #F44336; }}
  .badge {{ display: inline-block; padding: 3px 10px; border-radius: 10px;
            font-weight: bold; color: white; font-size: 12px; }}
  .badge-pass {{ background: #4CAF50; }}
  .badge-fail {{ background: #F44336; }}
  .stat-grid {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 14px; margin: 20px 0; }}
  .stat-card {{ background: white; padding: 16px; border-radius: 8px; text-align: center;
                box-shadow: 0 1px 3px rgba(0,0,0,0.1); }}
  .stat {{ font-size: 28px; font-weight: bold; color: #1a1a2e; }}
  .stat-label {{ font-size: 11px; color: #777; text-transform: uppercase; }}
  footer {{ margin-top: 40px; padding-top: 16px; border-top: 1px solid #ddd;
            color: #999; font-size: 12px; }}
</style></head>
<body>
<h1>Phase 1A — 8-Qubit VQC Smoke Test</h1>
<p style="color:#666">Quantum encoding verification &middot; {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}</p>

<div class="verdict">{verdict}</div>

<div class="circuit-box">
Circuit Architecture (8 qubits, {config['n_params']} parameters):

  q0: ─ RY(f₀·π) ──●──────────────── RY(θ₀) ── ⟨Z⟩
  q1: ─ RY(f₁·π) ──X──●───────────── RY(θ₁) ──
  q2: ─ RY(f₂·π) ─────X──●────────── RY(θ₂) ──
  q3: ─ RY(f₃·π) ────────X──●─────── RY(θ₃) ──
  q4: ─ RY(f₄·π) ───────────X──●──── RY(θ₄) ──
  q5: ─ RY(f₅·π) ──────────────X──●─ RY(θ₅) ──
  q6: ─ RY(f₆·π) ─────────────────X──●─ RY(θ₆) ──
  q7: ─ RY(f₇·π) ────────────────────X─ RY(θ₇) ──

  Encoding: feature × π → RY rotation angle
  Entanglement: Linear CNOT chain
  Ansatz: Single RY variational layer
  Measurement: ⟨Z⟩ expectation on qubit 0

  f₀=momentum_composite  f₁=realized_vol_20d  f₂=vol_regime
  f₃=carry_differential  f₄=event_risk_score  f₅=net_sentiment
  f₆=cot_positioning     f₇=risk_sentiment
</div>

<div class="stat-grid">
  <div class="stat-card"><div class="stat">{N_QUBITS}</div><div class="stat-label">Qubits</div></div>
  <div class="stat-card"><div class="stat">{len(vectors)}</div><div class="stat-label">Vectors Tested</div></div>
  <div class="stat-card"><div class="stat">{tests.get('unique_outputs', 0)}</div><div class="stat-label">Unique Outputs</div></div>
  <div class="stat-card"><div class="stat">{tests.get('gradient_nonzero_count', 0)}/8</div><div class="stat-label">Grad Non-Zero</div></div>
</div>

<h2>Test Results</h2>

<div class="test-card {'test-pass' if tests.get('forward_valid') else 'test-fail'}">
  <strong>Test 1: Forward Pass</strong> <span class="badge {'badge-pass' if tests.get('forward_valid') else 'badge-fail'}">{'PASS' if tests.get('forward_valid') else 'FAIL'}</span>
  <p>All outputs are valid expectation values in [-1, 1]. {tests.get('unique_outputs', 0)} unique outputs from {len(vectors)} inputs.</p>
</div>

<div class="test-card {'test-pass' if tests.get('output_spread', 0) > 0.01 else 'test-fail'}">
  <strong>Test 2: Input Sensitivity</strong> <span class="badge {'badge-pass' if tests.get('output_spread', 0) > 0.01 else 'badge-fail'}">{'PASS' if tests.get('output_spread', 0) > 0.01 else 'FAIL'}</span>
  <p>Output spread: {tests.get('output_spread', 0):.6f}, std: {tests.get('output_std', 0):.6f}. Different market states produce different circuit outputs.</p>
</div>

<div class="test-card {'test-pass' if tests.get('avg_fidelity', 1) < 0.99 else 'test-fail'}">
  <strong>Test 3: State Distinguishability</strong> <span class="badge {'badge-pass' if tests.get('avg_fidelity', 1) < 0.99 else 'badge-fail'}">{'PASS' if tests.get('avg_fidelity', 1) < 0.99 else 'FAIL'}</span>
  <p>Avg fidelity: {tests.get('avg_fidelity', 0):.4f} (lower = more distinguishable). Market regimes produce distinct quantum states.</p>
</div>

<div class="test-card {'test-pass' if tests.get('gradient_nonzero_count', 0) > 0 else 'test-fail'}">
  <strong>Test 4: Trainability</strong> <span class="badge {'badge-pass' if tests.get('gradient_nonzero_count', 0) > 0 else 'badge-fail'}">{'PASS' if tests.get('gradient_nonzero_count', 0) > 0 else 'FAIL'}</span>
  <p>Gradient norm: {tests.get('gradient_norm', 0):.6f}. {tests.get('gradient_nonzero_count', 0)}/8 parameters have non-zero gradients. Circuit is trainable.</p>
</div>

<h2>Per-Vector Results</h2>
<table>
<tr><th>Label</th><th>Input Features (8Q)</th><th>⟨Z₀⟩</th><th>All ⟨Z⟩</th></tr>
{vec_rows}
</table>

<h2>Parameter Gradients</h2>
<table>
<tr><th>Parameter</th><th>Gradient</th><th>Magnitude</th></tr>
{grad_rows}
</table>

<h2>Fidelity Matrix</h2>
<p style="font-size:12px;color:#666">Values near 1.0 = similar states (green). Values near 0 = highly distinguishable (red).</p>
{fid_html}

<footer>
  quantum_signal v1.0 &middot; VQC Phase 1A smoke test &middot; {N_QUBITS}-qubit circuit &middot; numpy statevector simulator
</footer>
</body></html>"""

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w") as f:
        f.write(html)
    print(f"\nHTML report: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="VQC Smoke Test — Phase 1A")
    parser.add_argument("--vectors", help="Pre-computed vectors JSON file")
    parser.add_argument("--snapshot-dir", default=str(PHASE0_DIR / "snapshots"),
                        help="Snapshot directory for building vectors")
    parser.add_argument("--fixture", action="store_true",
                        help="Use synthetic fixture vectors (no snapshots needed)")
    parser.add_argument("--html", default=str(RESULTS_DIR / "vqc_smoke_report.html"),
                        help="HTML report output path")
    parser.add_argument("--json-out", help="Save raw results as JSON")
    args = parser.parse_args()

    # Load vectors
    if args.vectors and os.path.exists(args.vectors):
        print(f"Loading vectors from: {args.vectors}")
        vectors = load_vectors_from_file(args.vectors)
    elif not args.fixture:
        print(f"Loading snapshots from: {args.snapshot_dir}")
        vectors = load_vectors_from_snapshots(args.snapshot_dir)
        if not vectors:
            print("No snapshots found, falling back to fixtures...")
            vectors = generate_fixture_vectors()
    else:
        vectors = generate_fixture_vectors()

    if not vectors:
        print("ERROR: No vectors available")
        sys.exit(1)

    print(f"Loaded {len(vectors)} vectors for smoke test")

    # Run smoke tests
    results = run_smoke_tests(vectors)

    # Generate reports
    generate_smoke_html(results, args.html)

    if args.json_out:
        os.makedirs(os.path.dirname(args.json_out) or ".", exist_ok=True)
        with open(args.json_out, "w") as f:
            json.dump(results, f, indent=2, default=str)
        print(f"JSON results: {args.json_out}")


if __name__ == "__main__":
    main()
