"""
train_and_run.py
================
Unified Command-Line & Graphical Entry Point for BCI Tower Defense Rhythm Training.

Usage:
  # Launch the Graphical User Interface (Default):
  python train_and_run.py
  # or explicitly:
  python train_and_run.py --gui

  # Command-Line headless training:
  python train_and_run.py --dataset bids_tower_defense --sub 02 --ses 05 --alg riemann_logreg

  # Train across multiple sessions and auto-select best algorithm:
  python train_and_run.py --dataset bids_tower_defense --sub 02 --ses 01,02,03,04,05 --alg all

  # Train and immediately launch the real-time pipeline:
  python train_and_run.py --dataset bids_tower_defense --sub 02 --ses 05 --alg riemann_logreg --run-pipeline --source simulator
"""

import os
import sys
import subprocess
from pathlib import Path

# Auto-detect virtual environment if launched with system Python lacking dependencies
def _ensure_environment():
    try:
        import numpy
        import scipy
        import sklearn
    except ImportError:
        _here = Path(__file__).resolve().parent
        cand_venvs = [
            _here.parent.parent.parent / "tower-defense-bci" / "python" / ".venv" / "bin" / "python",
            _here.parent.parent / "tower-defense-bci" / "python" / ".venv" / "bin" / "python",
            Path("/home/guilhermecoto/Documentos/Lasige/tower-defense-bci/python/.venv/bin/python"),
            _here.parent.parent.parent / "tower-defense-bci" / "python" / ".venv" / "Scripts" / "python.exe",
            _here.parent.parent / "tower-defense-bci" / "python" / ".venv" / "Scripts" / "python.exe",
            Path(r"c:\Users\guilh\Desktop\Lasige\tower-defense-bci\python\.venv\Scripts\python.exe")
        ]
        for venv_py in cand_venvs:
            if venv_py.exists():
                # Re-exec under virtual environment python
                cmd = [str(venv_py), str(Path(__file__).resolve())] + sys.argv[1:]
                env = os.environ.copy()
                env["PYTHONPATH"] = str(_here) + os.pathsep + env.get("PYTHONPATH", "")
                sys.exit(subprocess.call(cmd, env=env))
        print("[Error] Required dependencies (numpy, scipy, scikit-learn) not found.")
        print(f"Please run using the virtual environment at: tower-defense-bci/python/.venv/bin/python or Scripts/python.exe")
        sys.exit(1)

_ensure_environment()

import json
import argparse
import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, cohen_kappa_score, confusion_matrix

# Ensure local directories are in sys.path
_current_dir = Path(__file__).resolve().parent

def find_tower_defense_dirs():
    """Finds all candidate tower-defense-bci/python directories across environments."""
    candidates = []

    # 1. Environment variable override
    for env_k in ["TOWER_DEFENSE_PYTHON_DIR", "TOWER_DEFENSE_DIR", "TD_PYTHON_DIR"]:
        env_val = os.environ.get(env_k)
        if env_val:
            p = Path(env_val).resolve()
            if (p / "main.py").exists():
                candidates.append(p)
            elif (p / "python" / "main.py").exists():
                candidates.append(p / "python")

    # 2. Submodule inside nautilus_bci
    submod = _current_dir.parent.parent / "tower-defense-bci" / "python"
    if (submod / "main.py").exists():
        candidates.append(submod.resolve())

    # 3. Sibling next to nautilus_bci
    sibling = _current_dir.parent.parent.parent / "tower-defense-bci" / "python"
    if (sibling / "main.py").exists():
        candidates.append(sibling.resolve())

    # 4. Working directory / relative to cwd
    cwd = Path.cwd().resolve()
    for c in [
        cwd / "tower-defense-bci" / "python",
        cwd / "python",
        cwd.parent / "tower-defense-bci" / "python",
    ]:
        if (c / "main.py").exists():
            candidates.append(c.resolve())

    # 5. Search upwards from _current_dir
    p = _current_dir.resolve()
    while p != p.parent:
        c1 = p / "tower-defense-bci" / "python"
        if (c1 / "main.py").exists():
            candidates.append(c1.resolve())
        p = p.parent

    # 6. Fallback known standard locations
    for fixed in [
        Path("/home/guilhermecoto/Documentos/Lasige/tower-defense-bci/python"),
        Path("/home/guilhermecoto/Documentos/Lasige/nautilus_bci/tower-defense-bci/python"),
        Path.home() / "Documentos" / "Lasige" / "tower-defense-bci" / "python",
        Path.home() / "Documents" / "Lasige" / "tower-defense-bci" / "python",
        Path(r"C:\Users\guilh\Desktop\Lasige\tower-defense-bci\python"),
    ]:
        if fixed.exists() and (fixed / "main.py").exists():
            candidates.append(fixed.resolve())

    unique = []
    for c in candidates:
        if c not in unique:
            unique.append(c)
    return unique


def get_primary_tower_defense_dir():
    """Returns the primary tower-defense-bci/python directory, prioritizing existing virtualenvs."""
    dirs = find_tower_defense_dirs()
    if not dirs:
        submod = _current_dir.parent.parent / "tower-defense-bci" / "python"
        return submod if submod.exists() else (_current_dir.parent.parent.parent / "tower-defense-bci" / "python")

    for d in dirs:
        if (d / ".venv" / "bin" / "python").exists() or (d / ".venv" / "Scripts" / "python.exe").exists():
            return d
    return dirs[0]


def resolve_pipeline_python(td_dir: Path) -> Path:
    """Finds the best python interpreter equipped to run the tower-defense pipeline."""
    for venv_sub in [
        td_dir / ".venv" / "bin" / "python",
        td_dir / ".venv" / "Scripts" / "python.exe",
    ]:
        if venv_sub.exists():
            return venv_sub

    for other_td in find_tower_defense_dirs():
        for venv_sub in [
            other_td / ".venv" / "bin" / "python",
            other_td / ".venv" / "Scripts" / "python.exe",
        ]:
            if venv_sub.exists():
                return venv_sub

    for root_cand in [
        _current_dir.parent.parent / ".venv" / "bin" / "python",
        _current_dir.parent.parent / ".venv" / "Scripts" / "python.exe",
        _current_dir.parent / ".venv" / "bin" / "python",
        _current_dir.parent / ".venv" / "Scripts" / "python.exe",
        _current_dir / ".venv" / "bin" / "python",
        _current_dir / ".venv" / "Scripts" / "python.exe",
    ]:
        if root_cand.exists():
            return root_cand

    if "VIRTUAL_ENV" in os.environ:
        v_py = Path(os.environ["VIRTUAL_ENV"]) / ("Scripts/python.exe" if sys.platform == "win32" else "bin/python")
        if v_py.exists():
            return v_py

    return Path(sys.executable)


_td_python_dir = get_primary_tower_defense_dir()
if str(_current_dir) not in sys.path:
    sys.path.insert(0, str(_current_dir))
for td_d in find_tower_defense_dirs():
    if str(td_d) not in sys.path:
        sys.path.insert(0, str(td_d))

import dataset
import algorithms


def train_model(
    dataset_name="bids_tower_defense",
    sub_id="02",
    session_ids=None,
    include_listening=False,
    listening_items=None,
    alg_key="riemann_logreg",
    n_splits=5,
    C_val=0.1,
    output_path=None,
    report_path=None,
    custom_tag=""
):
    """
    Trains a 4-class mental rhythm decoder across chosen dataset, subject, and sessions.
    Evaluates under n-fold Stratified Cross-Validation, exports .joblib and .json artifacts.
    """
    if (session_ids is None or len(session_ids) == 0) and not include_listening:
        available = dataset.get_available_sessions(dataset_name, sub_id)
        if not available:
            raise ValueError(f"No sessions found for sub-{sub_id} in {dataset_name}")
        session_ids = available

    sub_clean = sub_id.replace("sub-", "")
    print("=" * 80)
    print(" BCI TOWER DEFENSE: RHYTHM MODEL TRAINING STUDIO ".center(80, "="))
    print("=" * 80)
    print(f"[*] Dataset         : {dataset_name}")
    print(f"[*] Subject         : sub-{sub_clean}")
    print(f"[*] Game Sessions   : {session_ids if session_ids else 'None'}")
    if include_listening:
        print(f"[*] Music Listening : bids_listening (items: {listening_items if listening_items else 'all'})")
    else:
        print(f"[*] Music Listening : Disabled")
    print(f"[*] Algorithm       : {alg_key.upper()}")
    print(f"[*] CV Folds        : {n_splits}")
    print("=" * 80)

    # 1. Load Sessions
    print("\n---> Loading and Preprocessing EEG Sessions...")
    X_im, y_im, stats, meta_df = dataset.load_dataset_sessions(
        dataset_name,
        sub_clean,
        session_ids if session_ids else [],
        include_listening=include_listening,
        listening_item_ids=listening_items,
        sfreq=250.0,
        win_len_s=3.0,
        spatial_mode="robust_car",
        progress_callback=lambda msg, _: print(f"    {msg}")
    )

    print(f"\n[+] Total Pooled Dataset: {len(y_im)} trials | Epoch Shape: {X_im.shape}")
    class_counts = dict(pd.Series(y_im).value_counts())
    print(f"    Class Breakdown (0:FIRE, 1:WATER, 2:WIND, 3:ELECTRICITY): {class_counts}")

    # 2. Benchmark or Single Algorithm Training
    candidates = list(algorithms.ALGORITHMS.keys()) if alg_key == "all" else [alg_key]
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    results = {}

    print("\n" + "-" * 80)
    print(f" {n_splits}-FOLD STRATIFIED CROSS-VALIDATION EVALUATION ".center(80, "-"))
    print("-" * 80)

    for c_key in candidates:
        c_info = algorithms.ALGORITHMS[c_key]
        print(f"\n[*] Evaluating [{c_info['name']}]...")

        accs, f1s, kappas = [], [], []
        oof_preds = np.zeros_like(y_im)

        for fold, (tr_idx, te_idx) in enumerate(cv.split(X_im, y_im)):
            if "riemann" in c_key or "fbcsp" in c_key:
                clf = algorithms.create_classifier(c_key, C=C_val) if hasattr(algorithms.ALGORITHMS[c_key]['class'], 'C') else algorithms.create_classifier(c_key)
            else:
                clf = algorithms.create_classifier(c_key)

            clf.fit(X_im[tr_idx], y_im[tr_idx])
            p_te = clf.predict(X_im[te_idx])
            oof_preds[te_idx] = p_te

            accs.append(accuracy_score(y_im[te_idx], p_te))
            f1s.append(f1_score(y_im[te_idx], p_te, average="macro"))
            kappas.append(cohen_kappa_score(y_im[te_idx], p_te))

        mean_acc = float(np.mean(accs))
        std_acc = float(np.std(accs))
        mean_f1 = float(np.mean(f1s))
        mean_kap = float(np.mean(kappas))
        cm = confusion_matrix(y_im, oof_preds).tolist()

        print(f"    • Mean Accuracy : {mean_acc * 100:.2f}% ± {std_acc * 100:.2f}% (Chance: 25.00%)")
        print(f"    • Macro F1      : {mean_f1:.3f}")
        print(f"    • Cohen's Kappa : {mean_kap:.3f}")

        results[c_key] = {
            'name': c_info['name'],
            'mean_acc': mean_acc,
            'std_acc': std_acc,
            'mean_f1': mean_f1,
            'mean_kappa': mean_kap,
            'cm': cm
        }

    # Select best model
    best_key = max(results.keys(), key=lambda k: results[k]['mean_acc'])
    best_res = results[best_key]

    print("\n" + "=" * 80)
    print(f" SELECTED BEST ALGORITHM: {best_res['name']} ({best_res['mean_acc']*100:.2f}%) ".center(80, "="))
    print("=" * 80)

    # 3. Fit Final Production Model
    print(f"\n[*] Fitting final model on full dataset ({len(y_im)} trials)...")
    if "riemann" in best_key or "fbcsp" in best_key:
        final_model = algorithms.create_classifier(best_key, C=C_val) if hasattr(algorithms.ALGORITHMS[best_key]['class'], 'C') else algorithms.create_classifier(best_key)
    else:
        final_model = algorithms.create_classifier(best_key)

    final_model.fit(X_im, y_im)
    self_acc = float(accuracy_score(y_im, final_model.predict(X_im)))
    print(f"[+] Final Model Fit Complete. Self-Accuracy: {self_acc*100:.2f}%")

    # 4. Save Artifacts
    models_dir = _current_dir / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    if custom_tag.strip():
        tag = custom_tag.strip()
    else:
        lis_suffix = f"_{len(listening_items)}lis" if include_listening and listening_items else ("_lis" if include_listening else "")
        n_ses = len(session_ids) if session_ids else 0
        tag = f"sub{sub_clean}_{best_key}_{n_ses}ses{lis_suffix}"

    if output_path is None:
        output_path = models_dir / f"rhythm_model_{tag}.joblib"
    else:
        output_path = Path(output_path)

    if report_path is None:
        report_path = models_dir / f"rhythm_report_{tag}.json"
    else:
        report_path = Path(report_path)

    export_dict = {
        'model': final_model,
        'model_name': f"{best_key.upper()}_4Class_TowerDefense",
        'algorithm_key': best_key,
        'algorithm_name': best_res['name'],
        'dataset_folder': dataset_name,
        'subject': sub_clean,
        'sessions': session_ids,
        'include_listening': include_listening,
        'listening_items': listening_items if include_listening else [],
        'classes': ['FIRE', 'WATER', 'WIND', 'ELECTRICITY'],
        'element_mapping': algorithms.ELEMENT_NAMES,
        'sfreq': 250.0,
        'window_size_sec': 3.0,
        'n_trials': len(y_im),
        'metrics': {
            'cv_accuracy_mean': best_res['mean_acc'],
            'cv_accuracy_std': best_res['std_acc'],
            'cv_f1_macro': best_res['mean_f1'],
            'cv_cohen_kappa': best_res['mean_kappa'],
            'self_accuracy': self_acc
        },
        'confusion_matrix': best_res['cm']
    }

    joblib.dump(export_dict, output_path)
    print(f"\n[+] Exported joblib model: {output_path}")

    # Sync to all found tower-defense-bci models directories
    for td_d in find_tower_defense_dirs():
        td_models = td_d / "models"
        if td_models.exists():
            try:
                joblib.dump(export_dict, td_models / output_path.name)
                print(f"[+] Synced model to {td_d.parent.name}/python/models/{output_path.name}")
            except Exception as e:
                print(f"[!] Note: Could not sync to {td_models}: {e}")

    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(export_dict, f, indent=2, default=str)
    print(f"[+] Exported JSON report: {report_path}")

    return final_model, output_path, export_dict


def launch_pipeline(
    model_path,
    source="simulator",
    mode="bids_replay",
    sub="02",
    ses="05",
    threshold=0.40,
    auto_send=True,
    interactive=False
):
    """Launches tower-defense-bci/python/main.py with the specified model."""
    td_dir = get_primary_tower_defense_dir()
    main_py = td_dir / "main.py"
    if not main_py.exists():
        for cand in find_tower_defense_dirs():
            if (cand / "main.py").exists():
                td_dir = cand
                main_py = cand / "main.py"
                break

    if not main_py.exists():
        raise FileNotFoundError(
            f"Could not locate main.py at {main_py}. Checked candidates: {find_tower_defense_dirs()}"
        )

    print("\n" + "=" * 80)
    print(" INITIATING REAL-TIME BCI PIPELINE (GODOT BRIDGE) ".center(80, "="))
    print(f"[*] Tower Defense Dir : {td_dir}")
    print(f"[*] Script Path       : {main_py}")
    print(f"[*] Active Model      : {Path(model_path).name}")
    print("=" * 80)

    # Ensure current directory and td_dir are in sys.path and PYTHONPATH for unpickling custom classifiers
    if str(_current_dir) not in sys.path:
        sys.path.insert(0, str(_current_dir))
    if str(td_dir) not in sys.path:
        sys.path.insert(0, str(td_dir))
    os.environ["PYTHONPATH"] = str(_current_dir) + os.pathsep + str(td_dir) + os.pathsep + os.environ.get("PYTHONPATH", "")

    # Resolve python interpreter
    venv_py = resolve_pipeline_python(td_dir)

    # Import and run directly in-process or via pipeline
    try:
        from main import run_pipeline
        run_pipeline(
            source=source,
            mode=mode,
            auto_send=auto_send,
            threshold=threshold,
            interactive=interactive,
            model_path=str(model_path),
            sub=sub,
            ses=ses
        )
    except Exception as e:
        print(f"[*] Starting pipeline in subprocess due to: {e}")
        import subprocess
        cmd = [
            str(venv_py), str(main_py),
            "--source", source,
            "--mode", mode,
            "--threshold", str(threshold),
            "--sub", sub,
            "--ses", ses,
            "--model", str(model_path)
        ]
        if auto_send:
            cmd.append("--auto-send")
        if interactive:
            cmd.append("--interactive")

        run_env = os.environ.copy()
        run_env["PYTHONPATH"] = str(_current_dir) + os.pathsep + str(td_dir) + os.pathsep + run_env.get("PYTHONPATH", "")
        subprocess.run(cmd, cwd=str(td_dir), env=run_env)


def main():
    parser = argparse.ArgumentParser(description="BCI Tower Defense Rhythm Training Studio & Real-Time Launcher")
    parser.add_argument("--gui", action="store_true", help="Launch Graphical User Interface (Default if no arguments provided)")
    parser.add_argument("--dataset", type=str, default="bids_tower_defense",
                        help="Dataset variant: bids_tower_defense, bids_tower_defense(old), bids_tower_defense_6_3_27")
    parser.add_argument("--sub", type=str, default="02", help="Subject ID (e.g. 01, 02)")
    parser.add_argument("--ses", type=str, default="all",
                        help="Sessions to pool: 'all', single session '05', or comma-separated '01,02,05'")
    parser.add_argument("--alg", type=str, default="riemann_logreg",
                        choices=["all"] + list(algorithms.ALGORITHMS.keys()),
                        help="Algorithm or 'all' to benchmark and select best")
    parser.add_argument("--cv", type=int, default=5, help="Number of cross-validation folds")
    parser.add_argument("--C", type=float, default=0.1, help="Regularization parameter C")
    parser.add_argument("--tag", type=str, default="", help="Custom model filename tag")
    parser.add_argument("--output", type=str, default=None, help="Output path for joblib model")
    parser.add_argument("--report", type=str, default=None, help="Output path for JSON report")
    parser.add_argument("--include-listening", action="store_true", default=False,
                        help="Include pure music listening data from bids_listening")
    parser.add_argument("--listening-items", type=str, default=None,
                        help="Optional comma-separated list of listening tracks/sessions to include")

    # Pre-Trained Model & Real-Time Pipeline flags
    parser.add_argument("--model", type=str, default=None,
                        help="Path to an existing pre-trained .joblib/.pkl model (skips training if launching pipeline)")
    parser.add_argument("--run-pipeline", action="store_true", help="Immediately run real-time pipeline")
    parser.add_argument("--source", type=str, default="simulator", choices=["simulator", "lsl"],
                        help="EEG source for real-time pipeline")
    parser.add_argument("--mode", type=str, default="bids_replay", choices=["bids_replay", "synthetic"],
                        help="Simulator mode")
    parser.add_argument("--threshold", type=float, default=0.40, help="Confidence threshold")
    parser.add_argument("--no-auto-send", dest="auto_send", action="store_false", default=True,
                        help="Disable auto-send to Godot")
    parser.add_argument("--interactive", action="store_true", default=False,
                        help="Interactive rhythm switching in simulator")

    # If user ran without arguments, open GUI
    if len(sys.argv) == 1:
        from gui import launch_gui
        launch_gui()
        return

    args = parser.parse_args()

    if args.gui:
        from gui import launch_gui
        launch_gui()
        return

    # If user provided an existing model and wants to run the pipeline directly
    if args.model and args.run_pipeline:
        launch_pipeline(
            model_path=args.model,
            source=args.source,
            mode=args.mode,
            sub=args.sub,
            ses=args.ses if args.ses != "all" else "05",
            threshold=args.threshold,
            auto_send=args.auto_send,
            interactive=args.interactive
        )
        return

    # Parse sessions
    if args.ses.lower() == "all":
        sessions = dataset.get_available_sessions(args.dataset, args.sub)
    else:
        sessions = [s.strip().replace("ses-", "") for s in args.ses.split(",") if s.strip()]

    # Parse listening items
    listening_items = [s.strip() for s in args.listening_items.split(",") if s.strip()] if args.listening_items else None

    # Train model
    final_model, model_path, report = train_model(
        dataset_name=args.dataset,
        sub_id=args.sub,
        session_ids=sessions,
        include_listening=args.include_listening,
        listening_items=listening_items,
        alg_key=args.alg,
        n_splits=args.cv,
        C_val=args.C,
        output_path=args.output,
        report_path=args.report,
        custom_tag=args.tag
    )

    # Launch pipeline if requested
    if args.run_pipeline:
        launch_pipeline(
            model_path=model_path,
            source=args.source,
            mode=args.mode,
            sub=args.sub,
            ses=sessions[-1] if sessions else "05",
            threshold=args.threshold,
            auto_send=args.auto_send,
            interactive=args.interactive
        )


if __name__ == "__main__":
    main()
