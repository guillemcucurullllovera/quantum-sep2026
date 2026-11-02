"""Thin markdown report generator over experiments.analysis's summaries."""
from __future__ import annotations

from experiments.analysis import MetricSummary


def render_markdown_summary(title: str, summaries: list[MetricSummary]) -> str:
    lines = [f"# {title}", ""]
    if not summaries:
        lines.append("_No results._")
        return "\n".join(lines)
    lines.append("| metric | n | mean | stdev | min | max |")
    lines.append("|---|---|---|---|---|---|")
    for s in summaries:
        lines.append(
            f"| {s.metric} | {s.n} | {s.mean:.6g} | {s.stdev:.6g} | {s.minimum:.6g} | {s.maximum:.6g} |"
        )
    return "\n".join(lines) + "\n"
