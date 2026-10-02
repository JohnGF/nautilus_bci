"""Run provenance for analysis scripts.

Every analysis pipeline should record *what data it ran on* next to its
outputs, so a `results/` folder is self-describing months later. Call::

    write_provenance(out_dir, script_file=__file__,
                     params={"bids_root": bids_root, "sub": sub, "ses": ses})

which writes `RUN_INFO.txt` (human-readable) into `out_dir` capturing:
timestamp, script path, full CLI argv, git commit/branch/dirty state,
and the given params (path-like values get an existence check).

Stdlib only. Git lookup is best-effort (never raises). Importing this
module has no side effects.
"""

import datetime
import json
import os
import subprocess
import sys


def _repo_root_for(script_file):
    """Repo root assumed two levels above the calling script (scripts/*/<file>)."""
    return os.path.abspath(
        os.path.join(os.path.dirname(os.path.abspath(script_file)), "..", "..")
    )


def _git_info(repo_root):
    """Best-effort git provenance; never raises."""
    info = {"commit": "unavailable", "branch": "unavailable", "dirty": "unknown"}

    def _run(*args):
        try:
            proc = subprocess.run(
                ["git", "-C", repo_root, *args],
                capture_output=True, text=True, timeout=10,
            )
            if proc.returncode == 0:
                return proc.stdout.strip()
        except Exception:
            pass
        return ""

    commit = _run("rev-parse", "--short", "HEAD")
    if commit:
        info["commit"] = commit
    branch = _run("rev-parse", "--abbrev-ref", "HEAD")
    if branch:
        info["branch"] = branch
    status = _run("status", "--porcelain")
    if commit or branch or status is not None:
        dirty_paths = [ln for ln in status.splitlines() if ln.strip()] if status else []
        info["dirty"] = f"yes ({len(dirty_paths)} uncommitted paths)" if dirty_paths else "no"
    return info


def _describe_param(value):
    """Make a param value provenance-friendly (paths get an existence check)."""
    if isinstance(value, dict):
        return {str(k): _describe_param(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_describe_param(v) for v in value]
    if isinstance(value, os.PathLike):
        value = str(value)
    if isinstance(value, str) and (
        "/" in value or "\\" in value or value.endswith((".json", ".tsv", ".vhdr", ".edf"))
    ):
        return {"given": value, "exists_from_cwd": os.path.exists(value)}
    return value


def collect_provenance(script_file=None, params=None):
    """Return an ordered provenance dict (also used to render RUN_INFO.txt)."""
    if script_file is None:
        script_file = sys.argv[0] if sys.argv else "<unknown>"
    script_abs = os.path.abspath(script_file)
    try:
        repo_root = _repo_root_for(script_file)
        script_rel = os.path.relpath(script_abs, repo_root)
    except Exception:
        repo_root = os.path.dirname(script_abs)
        script_rel = os.path.basename(script_abs)
    git = _git_info(repo_root)
    return {
        "timestamp": datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
        "script": script_rel,
        "command": " ".join(sys.argv) if sys.argv else "<unknown>",
        "git_commit": git["commit"],
        "git_branch": git["branch"],
        "git_dirty": git["dirty"],
        "params": _describe_param(dict(params)) if params else "(no params recorded)",
    }


def render_txt(prov, output_dir):
    """Render the provenance dict as human-readable text."""
    lines = [
        "# RUN_INFO.txt — auto-generated analysis provenance (do not hand-edit)",
        f"timestamp:  {prov['timestamp']}",
        f"script:     {prov['script']}",
        f"command:    {prov['command']}",
        f"git_commit: {prov['git_commit']}",
        f"git_branch: {prov['git_branch']}",
        f"git_dirty:  {prov['git_dirty']}",
        f"output_dir: {os.path.abspath(output_dir)}",
        "",
        "inputs:",
    ]
    params = prov["params"]
    if isinstance(params, dict) and params:
        for key, val in params.items():
            if isinstance(val, dict) and set(val) == {"given", "exists_from_cwd"}:
                lines.append(f"  {key}: {val['given']} (exists: {val['exists_from_cwd']})")
            else:
                rendered = json.dumps(val, default=str)
                lines.append(f"  {key}: {rendered}")
    else:
        lines.append(f"  {params}")
    return "\n".join(lines) + "\n"


def write_provenance(out_dir, script_file=None, params=None, filename="RUN_INFO.txt"):
    """Write RUN_INFO.txt into out_dir (creating it). Returns the file path."""
    os.makedirs(out_dir, exist_ok=True)
    prov = collect_provenance(script_file=script_file, params=params)
    path = os.path.join(out_dir, filename)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(render_txt(prov, out_dir))
    return path
