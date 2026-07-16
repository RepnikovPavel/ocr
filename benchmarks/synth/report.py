"""Turn one or more synth_edge report JSONs into a Markdown edge summary."""

import argparse
import json
from pathlib import Path


def _fmt(value):
    return "n/a" if value is None else f"{value:.2f}"


def render_markdown(reports):
    """reports: list of (label, report_dict). Returns markdown string."""
    lines = ["# dots.mocr — synthetic edge report", ""]
    for label, report in reports:
        cfg = report.get("config", {})
        summary = report.get("summary", {})
        lines.append(f"## {label}")
        lines.append("")
        lines.append(f"- config: dpi={cfg.get('dpi')}, max_pixels={cfg.get('max_pixels')}, "
                     f"max_new_tokens={cfg.get('max_new_tokens')}")
        by_kind = summary.get("by_kind", {})
        if by_kind:
            lines.append("- mean primary metric by kind: "
                         + ", ".join(f"{k}={_fmt(v)}" for k, v in by_kind.items()))
        lines.append(f"- cases: {summary.get('n_cases')}")
        lines.append("")
        for family, points in summary.get("families", {}).items():
            lines.append(f"### {family}")
            lines.append("")
            lines.append("| difficulty | case | score |")
            lines.append("|---|---|---|")
            edge_note = None
            prev_ok = True
            for p in points:
                score = p["score"]
                mark = ""
                if score >= 0.9:
                    mark = "✅"
                elif score >= 0.6:
                    mark = "⚠️"
                    if prev_ok and edge_note is None:
                        edge_note = p
                else:
                    mark = "❌"
                    if prev_ok and edge_note is None:
                        edge_note = p
                prev_ok = score >= 0.9
                lines.append(f"| {p['knob']} | {p['case_id']} | {_fmt(score)} {mark} |")
            lines.append("")
            if edge_note:
                lines.append(f"**edge**: degrades starting at difficulty "
                             f"`{edge_note['knob']}` ({edge_note['case_id']}, "
                             f"score {_fmt(edge_note['score'])}).")
            else:
                lines.append("**edge**: no degradation observed across the tested range.")
            lines.append("")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("reports", nargs="+", help="one or more report JSON files")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    loaded = []
    for path in args.reports:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        loaded.append((Path(path).stem, data))
    md = render_markdown(loaded)
    Path(args.out).write_text(md, encoding="utf-8")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
