"""Write one LaTeX document per group + a JSON manifest of cases.

Pure Python (no LaTeX, no torch): run it anywhere. A separate texlive step
compiles the .tex files; run_edge.py consumes the PDFs plus the manifest.
"""

import argparse
import json
from pathlib import Path

from benchmarks.synth import docs


def write_tex_files(outdir):
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    cases = docs.all_cases()
    groups = docs.cases_by_group(cases)
    manifest = {"groups": {}}
    for group, group_cases in groups.items():
        tex = docs.build_document(group, group_cases)
        (outdir / f"{group}.tex").write_text(tex, encoding="utf-8")
        manifest["groups"][group] = [
            {
                "case_id": c.case_id,
                "kind": c.kind,
                "params": c.params,
                "ground_truth": c.ground_truth,
            }
            for c in group_cases
        ]
    (outdir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True, help="output dir for .tex + manifest.json")
    args = parser.parse_args()
    manifest = write_tex_files(args.out)
    total = sum(len(v) for v in manifest["groups"].values())
    print(f"wrote {len(manifest['groups'])} documents, {total} cases to {args.out}")


if __name__ == "__main__":
    main()
