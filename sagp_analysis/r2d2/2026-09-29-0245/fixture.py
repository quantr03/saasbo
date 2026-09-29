"""A synthetic `runs_r2d2/`-shaped tree with a known injected Delta, for testing `analyze.py`.

Each R2-D2 run is its half-Cauchy twin's stored run relabelled: the regret column is multiplied by
10**delta (so y(R2-D2) - y(twin) at t = 199 is delta exactly, the regrets being far above the 1e-8
floor), best_f is recomputed as f_star - regret, f is capped at best_f (so best_f still bounds the
running maximum of f), and the manifest's method, creation commit and every resume commit are set to
an R2-D2 launch commit. coords.csv and samples/ are symlinks to the twin's, so the stored runs are
only ever read. t = 20's y_mean/y_std, f_star and S are the twin's, so the pairing check passes.

    python fixture.py --hc-runs runs --out <scratch>/runs_r2d2 --scenario d_only
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import analyze  # noqa: E402

PILOT_COMMIT = "97eda925b"  # the pilot's launch commit (plan Task 13)


def _relabel_iterations(src: Path, dst: Path, delta: float, f_star: float, method: str) -> None:
    with src.open(newline="") as fh:
        rows = list(csv.reader(fh))
    head, body = rows[0], rows[1:]
    col = {c: i for i, c in enumerate(head)}
    scale = 10.0 ** delta
    for r in body:
        regret = float(r[col["regret"]]) * scale
        best_f = f_star - regret
        r[col["regret"]] = repr(regret)
        r[col["best_f"]] = repr(best_f)
        r[col["f"]] = repr(min(float(r[col["f"]]), best_f))
        r[col["method"]] = method
    with dst.open("w", newline="") as fh:
        csv.writer(fh).writerows([head] + body)


def build(hc_root: Path, out_root: Path, deltas: dict, families, seeds, commit: str = PILOT_COMMIT) -> None:
    """deltas: {R2-D2 run dir name: array over units u = family_index * len(seeds) + seed_index}."""
    seeds = list(seeds)
    for fi, fam in enumerate(families):
        for si, seed in enumerate(seeds):
            u = fi * len(seeds) + si
            for d, delta in deltas.items():
                m = analyze.BY_DIR[d]
                src = Path(hc_root) / fam / m.twin / f"seed{seed:02d}"
                dst = Path(out_root) / fam / d / f"seed{seed:02d}"
                dst.mkdir(parents=True)
                man = json.loads((src / "manifest.json").read_text())
                twin_method = man["method"]
                man["method"] = m.method
                man["git"] = {"commit": commit, "dirty": False}
                for r in man.get("resumed", []):
                    r["commit"] = commit
                (dst / "manifest.json").write_text(json.dumps(man, indent=2, sort_keys=True) + "\n")
                _relabel_iterations(src / "iterations.csv", dst / "iterations.csv", float(delta[u]),
                                    man["objective"]["f_star"], m.method)
                (dst / "log.txt").write_text((src / "log.txt").read_text().replace(
                    f"method={twin_method} ", f"method={m.method} "))
                (dst / "coords.csv").symlink_to((src / "coords.csv").resolve())
                (dst / "samples").symlink_to((src / "samples").resolve(), target_is_directory=True)


def main():
    import test_analyze  # the scenarios live with the tests
    ap = argparse.ArgumentParser()
    ap.add_argument("--hc-runs", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--scenario", choices=list(test_analyze.SCENARIOS), required=True)
    a = ap.parse_args()
    if a.out.exists():
        raise SystemExit(f"{a.out} exists; the fixture is only ever written to a new directory")
    deltas, expected = test_analyze.SCENARIOS[a.scenario]
    build(a.hc_runs, a.out, {k: np.asarray(v, float) for k, v in deltas.items()},
          families=list(analyze.PILOT_FAMILIES), seeds=range(5))
    print(f"built {a.out} ({a.scenario}); expected criteria {expected}")


if __name__ == "__main__":
    main()
