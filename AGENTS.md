# AGENTS.md — Nautilus BCI contributor guide (human + AI agents)

## Repo layout (post-2026-10 reorg)

| Path | What lives here |
|---|---|
| `scripts/` | **Python project root** (`pyproject.toml`, `uv.lock`, `.venv/`). All code: `tasks/`, `analysis/`, `bridges/`, `recorders/`, `training/`, `training_fnf/`, `utils/`, `visualizers/`, `bids*/` data |
| `assets/` | Runtime media: `music_tracks/`, `sounds/`, `videos/`. Read-mostly, ~400 MB |
| `results/` | Generated analysis outputs (figures, JSON reports). Never hand-edit; regenerable |
| `docs/` | `guides/` (onboarding), `science-notes/` (findings), `vendor-manuals/` (g.tec PDFs, reference only) |
| `games/` | Tower Defense + FNF game tasks. `tower-defense-bci/` and `fnf_prot/` are **git submodules** (separate repos); `tower-defense/` is built Godot binaries |
| `integrations/` | Bridges to external systems (`external_game_engines/`, `galaxy_watch_lsl` submodule) |
| `models/` | Pretrained weights + `assistive_inferencer.py` |

## Running things

- Python ≥ 3.12. All commands run from `scripts/`:
  ```bash
  cd scripts
  uv run python run_bci_suite.py            # master control panel
  uv run python tasks/task_launcher.py      # task selector
  uv run python analysis/analyze_bids_dataset.py --sub 01 --ses 02
  ```
  or use `scripts/.venv/bin/python` directly.
- `.bat` launchers are Windows entry points (`start_*.bat`); `.sh` for Linux. They `cd` to their own dir — keep them cwd-independent.
- Many `scripts/analysis/*.py` scripts are standalone (run from repo root OR `scripts/`). Never assume cwd.

## Path conventions (must follow)

- Python code: `from utils.paths import ASSETS_MUSIC, ASSETS_SOUNDS, ASSETS_VIDEOS, RESULTS_DIR, REPO_ROOT` (works whenever `scripts/` is on `sys.path`, which every `tasks/*` module bootstraps itself).
- Standalone analysis scripts: anchor to file location, e.g.
  `os.path.join(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")), "results", "<name>")`.
- Analysis outputs go to `<repo>/results/...` only. Never write to cwd-relative `analysis_results/` / `analyzes_results/` (pre-reorg names — treat any you find as a bug).
- Runtime media reads come from `<repo>/assets/...` only. Config/logs (`music_offset_config.json`, `battery_log.json`) stay in `scripts/`.
- BIDS datasets (`scripts/bids*/`) are large and stable — do not move or restructure them. `*_scans.tsv` churn in git status is pre-existing noise.
- Tower Defense game code resolves via `find_tower_defense_dirs()` in `scripts/training/{train_and_run,gui}.py` — canonical location is `games/tower-defense-bci/python/`, legacy root location kept as fallback.

## Submodule rules

- `games/tower-defense-bci`, `games/fnf_prot`, `integrations/galaxy_watch_lsl` are submodules. **Do not edit files inside them** — changes belong in their own repos.
- Moving a submodule requires updating `.gitmodules`, `.git/config` (`git config --rename-section`), the `.git` gitdir pointer, and `core.worktree` in `.git/modules/<name>/config`. If in doubt, don't move them.

## Verification before finishing

- `python3 -m py_compile <edited files>` (LSP "cannot resolve import" noise for venv packages is pre-existing — ignore it).
- CLI smoke: `<venv-python> <script> --help` must load.
- `git submodule status` must show no `-`/`+` prefixes; `git ls-files --stage games/ integrations/` must show `160000` gitlinks, never submodule file content.

## Known follow-ups

- `assets/` (~400 MB) and large `results/` binaries exceed GitHub's 50 MB/file recommendation — migrate to Git LFS.
- `models/__pycache__/*.pyc` is staged but should be ignored; `fnf_decoding_metrics.json` still embeds old `scripts/analysis/...` path strings (regenerable artifact).
