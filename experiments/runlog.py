"""experiments.runlog: one run's directory -- its configuration, its provenance, its files.

`RunLogger` owns every file a run writes and the order they are written in: the samples and the
checkpoint first, then the diagnostics, then the `iterations.csv` row that describes them, so a
complete row is always backed by the artifacts it names and a killed run can be resumed from the
last one. `manifest.json` is the run's provenance -- the resolved `RunConfig` and its hash, the git
commit and its cleanliness, the package versions, the digests of the vendored reference files and
the objective's ground-truth labels -- written once at creation and appended to on every resume.
`experiments.run_bo` is what drives all of it; nothing in `sagp/` may import this module.
"""
from __future__ import annotations

import csv
import dataclasses
import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from sagp.diagnostics import DiagThresholds
from sagp.gp import ELL_PRIOR, FittedGP, NUTSConfig


# --- configuration ---


@dataclass(frozen=True)
class RunConfig:
    """Every setting of one run, resolved once and written verbatim into `manifest.json`.

    Frozen and hashed (`config_hash`) because it is the identity of a run directory: resume
    refuses to continue a checkpoint under a different configuration, which is the only automatic
    protection there is against silently splicing two half-runs of different methods together.
    Two fields are excluded from that hash, because they say where the run lives and how far it
    goes rather than what it is: `out_dir`, so a moved or copied run directory still resumes, and
    `T`, so a run killed by a wall-clock limit can be resumed with a longer budget -- the very
    situation resume exists for. Extending `T` changes nothing already computed: iteration `t`
    depends on `(seed, t)` and the points before it, and `T` reaches the loop only through the
    end-of-run Sobol readout (`t == T - 1`), so the shortened run's last row carries a
    `sobol_hat` the longer run would have computed later.
    """

    family: str
    seed: int
    D: int
    method: str
    T: int
    n_init: int
    acq: str = "logei"
    alpha: float | None = None
    fixed_noise: float | None = None
    noiseless: bool = False
    nuts: NUTSConfig = NUTSConfig()
    thresholds: DiagThresholds = DiagThresholds()
    ell_prior: tuple[float, float] = ELL_PRIOR
    num_init_candidates: int = 5000
    num_restarts_ei: int = 5
    sobol_every: int = 25
    sobol_n: int = 2048
    out_dir: str = "runs"


def config_hash(cfg: RunConfig) -> str:
    """sha256 over the settings, less `out_dir` and `T`: the identity resume checks against."""
    fields = dataclasses.asdict(cfg)
    for excluded in ("out_dir", "T"):
        fields.pop(excluded)
    return hashlib.sha256(
        json.dumps(fields, sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()


def run_dir_for(out_dir: str | os.PathLike[str], cfg: RunConfig) -> Path:
    """`<out_dir>/<family>/<method>/seed{seed:02d}`, with the method's "/" flattened to "-"."""
    return Path(out_dir) / cfg.family / cfg.method.replace("/", "-") / f"seed{cfg.seed:02d}"


# --- provenance ---

_REPO_ROOT = Path(__file__).resolve().parent.parent
# The vendored reference this module is a generalization of. The plan requires all three stay
# byte-identical, so their hashes are what lets a later reader check that claim against a run.
_REFERENCE_FILES: tuple[str, ...] = ("saasgp.py", "saasbo.py", "util.py")
# The packages whose version can move a run's numbers; recorded, and compared again on resume.
_VERSIONED: tuple[str, ...] = ("jax", "jaxlib", "numpyro", "numpy", "scipy")
# What the copied `optimize_ei` keeps hard-coded (plan section 2): not `RunConfig` fields, because
# no run may vary them, but written into the manifest so it states them rather than implying them.
_REFERENCE_CONSTANTS: dict[str, float] = {"maxfun": 100, "jitter_sd": 1e-3, "xi": 0.0}


def _now() -> str:
    """The current time, ISO 8601 in UTC: what `created` and every `resumed` entry are stamped."""
    return datetime.now(timezone.utc).isoformat()


def _git_provenance() -> dict[str, object]:
    """The commit this run was launched from and whether the tree was dirty, or an unknown pair.

    Every failure -- no git, no checkout (the study can be run from an unpacked archive), a
    detached or broken repository -- collapses to `commit="unknown"`, `dirty=None` rather than to
    an exception: provenance is bookkeeping, and it must not be able to end a 16-hour job in its
    first second. `dirty` is `git status --porcelain` being non-empty, which counts untracked
    files too, since a run against an uncommitted script is exactly the case worth flagging.
    """

    def git(*args: str) -> str | None:
        try:
            done = subprocess.run(["git", *args], capture_output=True, text=True, cwd=_REPO_ROOT)
        except OSError:
            return None
        return done.stdout if done.returncode == 0 else None

    commit, status = git("rev-parse", "HEAD"), git("status", "--porcelain")
    return {
        "commit": "unknown" if commit is None else commit.strip(),
        "dirty": None if status is None else bool(status.strip()),
    }


def _versions() -> dict[str, str]:
    """The interpreter and the five packages a run's numbers depend on, by installed version.

    `PackageNotFoundError` is caught for the same reason `_git_provenance` catches everything: a
    conda environment can carry a working `jaxlib` whose distribution metadata is not where
    `importlib.metadata` looks, and that is not a reason to refuse to start the run.
    """
    versions = {"python": platform.python_version()}
    for name in _VERSIONED:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = "unknown"
    return versions


def _environment() -> dict[str, object]:
    """The thread settings and host, because bit-identity is only claimed *within* them.

    XLA's CPU backend is deterministic at a fixed thread count on a fixed CPU and not guaranteed
    across two of either (plan section 4, step 7), so a resume on a differently configured machine
    is reproducible in the sense the study needs -- same seeds, same design -- but not necessarily
    bit for bit, and these numbers are what tells a later reader which of the two they are looking
    at. `machine` and `processor` are here because `platform.platform()` names the OS build and
    not the instruction set: a cluster's nodes can share it and still differ in the vector width
    the kernels are compiled to, which is exactly the difference this block has to be able to show.
    """
    return {
        "XLA_FLAGS": os.environ.get("XLA_FLAGS"),
        "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS"),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "cpu_count": os.cpu_count(),
    }


def _reference_sha256() -> dict[str, str]:
    """sha256 of each vendored reference file, hashed off disk at the moment the run starts."""
    return {
        name: hashlib.sha256((_REPO_ROOT / name).read_bytes()).hexdigest()
        for name in _REFERENCE_FILES
    }


def _objective_labels(objective: object) -> dict[str, object]:
    """The problem's identity, so a run directory names it without the objective file beside it.

    `f_star` above all: the regret column is `f_star - best_f`, so a reader who wants to recompute
    a regret from `iterations.csv` alone needs the constant it was taken against, not a promise
    that regenerating the family would produce the same one.
    """
    labels = objective.labels
    return {
        "family": labels.family,
        "seed": int(labels.seed),
        "D": int(objective.D),
        "S": [int(i) for i in labels.S],
        "f_star": float(objective.f_star),
        "gamma": float(labels.gamma),
        "noise_sd": float(labels.noise_sd),
    }


# --- the run directory ---

# The plan's row schema, less the trailing x_0 ... x_{D-1} which depend on D. `reason` is the
# fit's own status_reason, or the exception that replaced the fit.
_ROW_FIELDS: tuple[str, ...] = (
    "t", "method", "family", "seed", "y", "f", "best_obs", "best_f", "regret", "acq_value",
    "fit_wall_s", "acq_wall_s", "fit_calls", "nuts_attempts", "status", "reason",
    "r_hat_max", "r_hat_median", "frac_r_hat_below_1_05", "n_eff_min", "divergences",
    "num_steps_mean",
    "r_hat_max_native", "n_eff_min_native", "r_hat_max_ell", "n_eff_min_ell",
    "r_hat_max_global", "n_eff_min_global",
    "a0_r_hat_max", "a0_n_eff_min", "a0_divergences",
    "y_mean", "y_std", "sobol_computed",
)
# The numbers of the retained attempt's `Diagnostics` that the row carries (its `wall_s`,
# `passed` and `reason` are already in `fit_wall_s`, `status` and `reason`): the six pooled ones
# and the six per-group ones the trigger is attributed with (`gp._DIAG_GROUPS`).
_DIAG_FIELDS: tuple[str, ...] = (
    "r_hat_max", "r_hat_median", "frac_r_hat_below_1_05", "n_eff_min", "divergences",
    "num_steps_mean",
    "r_hat_max_native", "n_eff_min_native", "r_hat_max_ell", "n_eff_min_ell",
    "r_hat_max_global", "n_eff_min_global",
)
# What the row keeps of *attempt 0* when a refit replaced it (ruling R43). Without these the
# trigger is unrecoverable from `iterations.csv`: the unprefixed columns are the retained
# attempt's, which on a refit row is the attempt that passed. `save_samples` keeps the whole
# attempt-0 `Diagnostics` beside the draws; these three are the ones the gate reads.
_ATTEMPT0_FIELDS: tuple[str, ...] = ("r_hat_max", "n_eff_min", "divergences")
_COORD_FIELDS: tuple[str, ...] = ("t", "i", "native_median", "p_active", "sobol_hat")


class RunLogger:
    """One run directory: the manifest, the checkpoint, the two CSVs, the samples and `log.txt`.

    Every write goes through here so that the *order* of writes is stated in one place. The
    checkpoint is written first, by tmp-file and `os.replace`, and the appends follow: a kill
    between the two then leaves the logs one iteration behind the checkpoint, which `truncate_to`
    rolls back, rather than leaving a duplicated or half-written row that no reader could detect.
    Within the appends the `iterations.csv` row is last (ruling R34), which makes it the run's
    completion marker: `last_logged_t` reads it as "this iteration's artifacts are all there".
    """

    def __init__(self, run_dir: Path, cfg: RunConfig) -> None:
        self.dir = run_dir
        self.cfg = cfg
        self.samples_dir = run_dir / "samples"
        self.iterations = run_dir / "iterations.csv"
        self.coords = run_dir / "coords.csv"
        self.checkpoint_path = run_dir / "checkpoint.npz"
        self.manifest = run_dir / "manifest.json"
        self.log_path = run_dir / "log.txt"
        self.row_fields = _ROW_FIELDS + tuple(f"x_{i}" for i in range(cfg.D))
        self.samples_dir.mkdir(parents=True, exist_ok=True)

    def log(self, message: str) -> None:
        """Append one line to `log.txt` (timings, statuses, tracebacks): the run's narrative."""
        with self.log_path.open("a") as handle:
            handle.write(message.rstrip("\n") + "\n")

    def write_manifest(self, objective: object) -> None:
        """The run's full provenance (plan section 4, step 7), written once when the run starts.

        Every block answers one question: could this directory have been produced by a different
        experiment than the one it claims to be? The `RunConfig` fields and `config_hash` fix the
        settings and resume compares the hash; `reference_constants` records what the copied
        `optimize_ei` holds fixed *underneath* those settings, which no flag can reach; and the
        rest -- the commit, the package versions, the thread environment, the vendored files'
        hashes and the objective's own labels -- fixes the code and the problem the settings were
        applied to. `resumed` starts empty and only `note_resume` ever adds to it: nothing here is
        recomputed in place, since a manifest that quietly followed its environment would answer
        that question with today's environment rather than with the run's.
        """
        fields = dataclasses.asdict(self.cfg)
        fields["config_hash"] = config_hash(self.cfg)
        fields["created"] = _now()
        fields["git"] = _git_provenance()
        fields["versions"] = _versions()
        fields["env"] = _environment()
        fields["reference_sha256"] = _reference_sha256()
        fields["objective"] = _objective_labels(objective)
        # The two acquisition-optimizer sizes are `RunConfig` fields *and* belong here: this block
        # is the optimizer's whole operating point, and splitting it across two places to avoid
        # repeating two numbers would make it readable only next to the config it came from.
        fields["reference_constants"] = {
            **_REFERENCE_CONSTANTS,
            "num_init_candidates": self.cfg.num_init_candidates,
            "num_restarts_ei": self.cfg.num_restarts_ei,
        }
        fields["resumed"] = []
        self.manifest.write_text(json.dumps(fields, indent=2, sort_keys=True, default=str) + "\n")

    def write_environment_lock(self) -> None:
        """`environment.lock.txt`: every installed distribution as `name==version`, sorted.

        The manifest's `versions` names the six packages we believe can move the numbers; this
        file names all of them, because those six are a hypothesis and this is the record that
        has to outlive it. Written when the run starts and never on resume: it describes the
        environment the run's first iterations were computed in, and a resume's own environment
        is what the manifest's `resumed` entry is for.
        """
        lines = sorted(
            f"{dist.metadata['Name']}=={dist.version}"
            for dist in importlib.metadata.distributions()
        )
        (self.dir / "environment.lock.txt").write_text("\n".join(lines) + "\n")

    def note_resume(self) -> None:
        """Append this resume to the manifest's `resumed` list, warning about anything that moved.

        Plan section 4, step 7 draws the line here: the config hash is *asserted* (by `run_bo`,
        which refuses a checkpoint written under different settings), while the commit and the
        package versions are only warned about. They have to be -- resuming days later into a
        rebuilt environment is the case this machinery exists for, and refusing it would throw
        away the run -- so what the study gets instead is a dated entry naming what changed, in
        `log.txt` for the person watching and in the manifest for the analysis. The manifest is
        re-read rather than rebuilt, so `created`, the original versions and every earlier resume
        survive; only this run's own `resumed` entry is new. That read-modify-write is the one
        place the manifest can be destroyed by a kill, so it lands by tmp-file and `os.replace`
        like the checkpoint (ruling R36): a truncated manifest would take the run's whole
        provenance with it, and resume reads `config_hash` back out of it.

        The entry carries this resume's `T` and its own `env` as well as the commit and the moved
        versions. `T` because it is excluded from `config_hash` on purpose -- extending the budget
        is the one setting a resume may change -- so the manifest's own `T` is the last resume's
        and the earlier ones would otherwise be unrecoverable; `env` because bit-identity is only
        claimed within one CPU and one thread setting, and a run's later iterations can have been
        computed on a different machine from its first.
        """
        recorded = json.loads(self.manifest.read_text())
        was = recorded.get("versions", {})
        now = _versions()
        changed = sorted(name for name, version in now.items() if was.get(name) != version)
        git = _git_provenance()
        recorded.setdefault("resumed", []).append(
            {
                "time": _now(),
                "commit": git["commit"],
                "versions_changed": changed,
                "T": self.cfg.T,
                "env": _environment(),
            }
        )
        tmp = self.manifest.with_name(self.manifest.name + ".tmp")
        tmp.write_text(json.dumps(recorded, indent=2, sort_keys=True, default=str) + "\n")
        os.replace(tmp, self.manifest)
        for name in changed:
            self.log(f"warning: {name} is {now[name]}, this run started under {was.get(name)}")
        started_at = recorded.get("git", {}).get("commit")
        if started_at != git["commit"]:
            self.log(f"warning: git commit is {git['commit']}, this run started at {started_at}")

    def manifest_hash(self) -> str | None:
        """The `config_hash` recorded in `manifest.json`, or None if there is no manifest."""
        if not self.manifest.exists():
            return None
        return json.loads(self.manifest.read_text()).get("config_hash")

    def reset(self) -> None:
        """Empty the run directory of everything a previous run wrote (a fresh, non-resumed run)."""
        for path in (self.iterations, self.coords, self.checkpoint_path, self.log_path):
            path.unlink(missing_ok=True)
        for path in self.samples_dir.glob("t*.npz"):
            path.unlink()

    def checkpoint(
        self, X: np.ndarray, y: np.ndarray, f: np.ndarray, t_done: int, statuses: list[str]
    ) -> None:
        """The state a resume needs, written atomically: everything else is derivable from it.

        `statuses` holds one entry per *iteration* (t from `n_init` to `t_done`), not per point:
        the initial design has no fit and therefore no status. The config hash rides along so a
        checkpoint can be told apart from another run's by looking at the file alone.
        """
        tmp = self.checkpoint_path.with_name(self.checkpoint_path.name + ".tmp")
        with tmp.open("wb") as handle:
            np.savez(
                handle,
                X=X,
                y=y,
                f=f,
                t_done=t_done,
                statuses=np.array(statuses, dtype="<U16"),
                config_hash=config_hash(self.cfg),
            )
        os.replace(tmp, self.checkpoint_path)

    def load_checkpoint(self) -> dict[str, np.ndarray] | None:
        """`checkpoint.npz` as a dict, or None if this run has not completed an iteration yet."""
        if not self.checkpoint_path.exists():
            return None
        with np.load(self.checkpoint_path, allow_pickle=False) as data:
            return {name: data[name] for name in data.files}

    def append_row(self, row: dict[str, object]) -> None:
        """Append one iteration to `iterations.csv`, writing the header if the file is new."""
        new = not self.iterations.exists() or self.iterations.stat().st_size == 0
        with self.iterations.open("a", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=self.row_fields)
            if new:
                writer.writeheader()
            writer.writerow(row)

    def append_coords(self, t: int, readout: dict[str, object]) -> None:
        """Append this iteration's D per-coordinate readouts to `coords.csv` (long format).

        Long rather than wide because D differs between studies and `sobol_hat` is only computed
        on some iterations; a long table needs no schema change when either moves.
        """
        new = not self.coords.exists() or self.coords.stat().st_size == 0
        native_median = np.asarray(readout["native_median"], dtype=float)
        p_active = np.asarray(readout["p_active"], dtype=float)
        sobol_hat = np.asarray(readout["sobol_hat"], dtype=float)
        with self.coords.open("a", newline="") as handle:
            writer = csv.writer(handle)
            if new:
                writer.writerow(_COORD_FIELDS)
            for i in range(native_median.shape[0]):
                writer.writerow(
                    [t, i, float(native_median[i]), float(p_active[i]), float(sobol_hat[i])]
                )

    def save_samples(self, t: int, fitted: FittedGP) -> None:
        """The retained draws of iteration `t`, with the status and attempts that produced them.

        This is the only place `bo.py` touches `fitted.samples`, and it does not look inside: what
        a cell's sites mean is `gp.py`'s business, and every later analysis re-reads them from here.
        Attempt 0's whole `Diagnostics` rides along as `a0_<field>` scalars (ruling R43), because
        the draws in this file are the *retained* attempt's and the row's unprefixed diagnostics
        are that attempt's too: without these, a refit's trigger is nowhere on disk. A MAP
        reference has no attempts and writes none of them.
        """
        arrays = {site: np.asarray(draws) for site, draws in fitted.samples.items()}
        attempt0 = (
            {f"a0_{name}": value for name, value in dataclasses.asdict(fitted.attempts[0]).items()}
            if fitted.attempts
            else {}
        )
        with (self.samples_dir / f"t{t:03d}.npz").open("wb") as handle:
            np.savez(
                handle,
                status=fitted.status,
                nuts_attempts=len(fitted.attempts),
                **attempt0,
                **arrays,
            )

    def last_logged_t(self) -> int | None:
        """The largest `t` with a *complete* row in `iterations.csv`, or None if there is none.

        A row shorter than the header is a row a kill interrupted mid-write; counting it as
        logged would leave the file with a torn line no reader could parse. Under ruling R34 a
        complete row is also the run's completion marker for the iteration -- the coordinates and
        the samples are written before it -- so this is what resume rolls the checkpoint back to.
        """
        rows = _complete_rows(self.iterations)
        if not rows:
            return None
        width = len(rows[0])
        complete = [int(row[0]) for row in rows[1:] if len(row) == width]
        return max(complete) if complete else None

    def truncate_to(self, t_done: int) -> None:
        """Drop every logged iteration after `t_done`, so the logs agree with the checkpoint."""
        for path in (self.iterations, self.coords):
            _truncate_csv(path, t_done)
        for path in self.samples_dir.glob("t*.npz"):
            if int(path.stem[1:]) > t_done:
                path.unlink()


def _complete_rows(path: Path) -> list[list[str]]:
    """`path`'s CSV rows, less a final line the kill left without its terminator (ruling R35).

    `csv.writer` ends every row it finishes with a line terminator, so a last line without one is
    a write that was interrupted -- and it can still *parse*, at the header's full width, when the
    kill happened to land on a field boundary. Nothing in the parsed rows distinguishes the two,
    so the raw text is what is inspected. Missing and empty files read as no rows at all.
    """
    if not path.exists():
        return []
    with path.open(newline="") as handle:  # newline="" so the terminators survive the read
        lines = handle.read().splitlines(keepends=True)
    if lines and not lines[-1].endswith("\n"):
        lines.pop()
    return list(csv.reader(lines))


def _truncate_csv(path: Path, t_done: int) -> None:
    """Rewrite `path` keeping its header and the rows whose first column (`t`) is <= `t_done`.

    A file holding no complete row at all is *removed* rather than left alone (ruling R37): the
    kill landed inside the header itself, and `append_row` treats any non-empty file as one that
    already has a header, so leaving the fragment there would append every later row underneath a
    torn first line. Unlinking makes the next append rewrite the header. A missing file -- a Sobol
    run's `coords.csv`, which never existed -- takes the same branch and is a no-op.
    """
    rows = _complete_rows(path)
    if not rows:
        path.unlink(missing_ok=True)
        return
    width = len(rows[0])
    kept = [row for row in rows[1:] if len(row) == width and int(row[0]) <= t_done]
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(rows[0])
        writer.writerows(kept)
