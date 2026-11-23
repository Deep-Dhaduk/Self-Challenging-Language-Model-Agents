"""Portable JSON/CSV artifacts and static SVG plots; no plotting dependency required."""

import csv
import html
import json
from pathlib import Path


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def write_jsonl(path: Path, values) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for value in values:
            handle.write(json.dumps(value, ensure_ascii=False) + "\n")


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def bar_chart(path: Path, title: str, subtitle: str, rows: list[tuple[str, float, str]]) -> None:
    height = 130 + 68 * len(rows)
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="1000" height="{height}" viewBox="0 0 1000 {height}">',
             '<rect width="100%" height="100%" fill="#f8fafc"/>',
             '<g font-family="Arial, sans-serif" fill="#0f172a">',
             f'<text x="28" y="35" font-size="23" font-weight="bold">{html.escape(title)}</text>',
             f'<text x="28" y="63" font-size="14">{html.escape(subtitle)}</text>']
    for i, (label, value, color) in enumerate(rows):
        y = 93 + i * 68
        width = 620 * max(0, min(1, value))
        parts.extend([f'<text x="28" y="{y + 23}" font-size="15">{html.escape(label)}</text>',
                      f'<rect x="250" y="{y}" width="620" height="34" rx="5" fill="#e2e8f0"/>',
                      f'<rect x="250" y="{y}" width="{width:.2f}" height="34" rx="5" fill="{color}"/>',
                      f'<text x="885" y="{y + 23}" font-size="16">{value:.1%}</text>'])
    parts.append('</g></svg>')
    path.write_text("\n".join(parts) + "\n", encoding="utf-8")


def write_reports(output: Path, summary: dict, ablations: list[dict]) -> None:
    write_json(output / "summary.json", summary)
    write_json(output / "filter_ablation.json", ablations)
    write_csv(output / "filter_ablation.csv", ablations)
    metric_rows, chart_rows = [], []
    for label in ("before", "after"):
        scores = summary.get(label)
        if scores is None:
            continue
        row = {"stage": label, **{key: value for key, value in scores.items() if key != "by_operation"}}
        metric_rows.append(row)
        chart_rows.append((f"{label.capitalize()}: Pass@1", scores["pass_at_1"], "#2563eb" if label == "before" else "#0f766e"))
        k = scores["attempts_per_task"]
        if k > 1:
            chart_rows.append((f"{label.capitalize()}: Pass@{k}", scores[f"pass_at_{k}"], "#60a5fa" if label == "before" else "#5eead4"))
    write_csv(output / "metrics.csv", metric_rows)
    label = "CPU strategy-policy demonstration" if summary["backend"] == "tabular" else "LLM experiment: " + summary["model"]
    bar_chart(output / "evaluation.svg", label, "Held-out synthetic retail tasks. Measured locally; these are not the paper's benchmark results.", chart_rows)
    bar_chart(output / "filter_ablation.svg", "Task-filter ablation: controlled fixture defects",
              "Acceptance rate on an intentionally corrupted fixture set; not an LLM task-generation measurement.",
              [(row["filter"], row["acceptance_rate"], color) for row, color in
               zip(ablations, ("#94a3b8", "#60a5fa", "#0f766e"))])
