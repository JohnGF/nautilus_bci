import os
import sys
import json
import time
import queue
import threading
import subprocess
from pathlib import Path
import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext, filedialog

def _ensure_environment():
    try:
        import pandas
    except ImportError:
        _here = Path(__file__).resolve().parent
        cand_venvs = [
            _here.parent / ".venv" / "bin" / "python",
            _here.parent / ".venv" / "Scripts" / "python.exe",
        ]
        for venv_py in cand_venvs:
            if venv_py.exists():
                cmd = [str(venv_py), str(Path(__file__).resolve())] + sys.argv[1:]
                env = os.environ.copy()
                env["PYTHONPATH"] = str(_here.parent / "analysis") + os.pathsep + env.get("PYTHONPATH", "")
                sys.exit(subprocess.call(cmd, env=env))
        print("[Error] Required dependencies (pandas, etc) not found and no local .venv found.")
        sys.exit(1)

_ensure_environment()

import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, cohen_kappa_score, confusion_matrix

_current_dir = Path(__file__).resolve().parent
_scripts_dir = _current_dir.parent
_ws_root = _current_dir.parent.parent.parent

if str(_current_dir) not in sys.path:
    sys.path.insert(0, str(_current_dir))
    
# Import dataset and algorithms from FNF context
import dataset
from analyze_fnf_bci import get_pipelines, DIRECTIONS

class FNFTrainingStudioGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Friday Night Funkin' (FNF) BCI Training Studio")
        self.root.geometry("1180x820")
        self.root.minsize(980, 700)

        # Apply high-contrast modern theme colors
        self.bg_color = "#0d1117"
        self.panel_bg = "#161b22"
        self.card_bg = "#21262d"
        self.card_border = "#30363d"
        self.accent_color = "#e34c26" # FNF reddish-orange accent
        self.fg_color = "#ffffff"
        self.muted_fg = "#c9d1d9"
        self.success_color = "#3fb950"
        self.metric_title_color = "#ff7b72"
        self.input_bg = "#0d1117"

        self.root.configure(bg=self.bg_color)
        self._setup_styles()

        # State
        self.selected_subject = tk.StringVar()
        self.session_vars = {}
        
        self.selected_algorithm = tk.StringVar()
        self.cv_folds = tk.IntVar(value=5)
        self.model_name_var = tk.StringVar()

        # Real-time state
        self.rt_source = tk.StringVar(value="simulator")
        self.rt_mode = tk.StringVar(value="bids_replay")
        self.rt_threshold = tk.DoubleVar(value=0.40)
        self.rt_autosend = tk.BooleanVar(value=True)
        self.rt_interactive = tk.BooleanVar(value=True)
        self.last_trained_model_path = None
        self.realtime_process = None

        # Pre-trained models catalog
        self.available_models = {}
        self.selected_model_var = tk.StringVar()

        self.log_queue = queue.Queue()

        self._build_ui()
        self._init_defaults()
        self._refresh_available_models()
        self._start_log_consumer()

    def _setup_styles(self):
        self.style = ttk.Style(self.root)
        try:
            self.style.theme_use("clam")
        except Exception:
            pass

        self.style.configure(".", background=self.panel_bg, foreground=self.fg_color, font=("Segoe UI", 10))
        self.style.configure("TLabel", background=self.panel_bg, foreground=self.fg_color)
        self.style.configure("Muted.TLabel", background=self.panel_bg, foreground=self.muted_fg, font=("Segoe UI", 9))
        self.style.configure("Header.TLabel", font=("Segoe UI", 14, "bold"), foreground="#ffffff")
        self.style.configure("SubHeader.TLabel", font=("Segoe UI", 11, "bold"), foreground=self.accent_color)

        self.style.configure("TFrame", background=self.panel_bg)
        self.style.configure("TLabelframe", background=self.panel_bg, foreground=self.accent_color, bordercolor=self.card_border)
        self.style.configure("TLabelframe.Label", background=self.panel_bg, foreground=self.accent_color, font=("Segoe UI", 10, "bold"))

        self.style.configure("Accent.TButton", font=("Segoe UI", 10, "bold"), background="#d73a49", foreground="#ffffff", bordercolor="#cb2431")
        self.style.map("Accent.TButton", background=[("active", "#cb2431"), ("disabled", "#30363d")], foreground=[("disabled", "#8b949e")])

        self.style.configure("Success.TButton", font=("Segoe UI", 10, "bold"), background="#1f6feb", foreground="#ffffff", bordercolor="#388bfd")
        self.style.map("Success.TButton", background=[("active", "#388bfd"), ("disabled", "#30363d")], foreground=[("disabled", "#8b949e")])

        self.style.configure("Danger.TButton", font=("Segoe UI", 9, "bold"), background="#da3633", foreground="#ffffff")
        self.style.map("Danger.TButton", background=[("active", "#f85149"), ("disabled", "#30363d")], foreground=[("disabled", "#8b949e")])

        self.style.configure("TCheckbutton", background=self.panel_bg, foreground=self.fg_color)
        self.style.map("TCheckbutton", background=[("active", self.panel_bg)])

        self.style.configure("TCombobox", fieldbackground=self.input_bg, background=self.panel_bg, foreground="#ffffff", arrowcolor="#ffffff")
        self.style.map("TCombobox", fieldbackground=[("readonly", self.input_bg)], foreground=[("readonly", "#ffffff")])
        self.style.configure("TSpinbox", fieldbackground=self.input_bg, foreground="#ffffff", arrowcolor="#ffffff")
        self.style.configure("TEntry", fieldbackground=self.input_bg, foreground="#ffffff")

    def _build_ui(self):
        header_frame = tk.Frame(self.root, bg=self.card_bg, height=60)
        header_frame.pack(side=tk.TOP, fill=tk.X, padx=12, pady=(10, 6))

        title_lbl = tk.Label(
            header_frame,
            text="FNF BCI ARROW LOCK-ON TRAINING STUDIO",
            font=("Segoe UI", 15, "bold"),
            bg=self.card_bg,
            fg="#ffffff"
        )
        title_lbl.pack(anchor="w", padx=16, pady=(8, 2))

        sub_lbl = tk.Label(
            header_frame,
            text="Train decoders for directional arrows across FNF sessions & bridge live into Godot",
            font=("Segoe UI", 9),
            bg=self.card_bg,
            fg=self.muted_fg
        )
        sub_lbl.pack(anchor="w", padx=16, pady=(0, 8))

        main_container = tk.Frame(self.root, bg=self.bg_color)
        main_container.pack(fill=tk.BOTH, expand=True, padx=12, pady=6)

        left_frame = tk.Frame(main_container, bg=self.panel_bg, bd=1, relief=tk.SOLID)
        left_frame.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 6), pady=0)
        self._build_left_controls(left_frame)

        right_frame = tk.Frame(main_container, bg=self.panel_bg, bd=1, relief=tk.SOLID)
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(6, 0), pady=0)
        self._build_right_dashboard(right_frame)

    def _build_left_controls(self, parent):
        canvas = tk.Canvas(parent, bg=self.panel_bg, highlightthickness=0, width=470)
        scrollbar = ttk.Scrollbar(parent, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)

        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=4, pady=4)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        sub_group = ttk.LabelFrame(scrollable_frame, text=" 1. Subject & Session Selection ", padding=10)
        sub_group.pack(fill=tk.X, padx=8, pady=6)

        sub_row = ttk.Frame(sub_group)
        sub_row.pack(fill=tk.X, pady=(0, 6))
        ttk.Label(sub_row, text="Subject:").pack(side=tk.LEFT, padx=(0, 8))
        self.sub_combo = ttk.Combobox(
            sub_row,
            textvariable=self.selected_subject,
            state="readonly",
            width=15,
            font=("Segoe UI", 9)
        )
        self.sub_combo.pack(side=tk.LEFT)
        self.sub_combo.bind("<<ComboboxSelected>>", self._on_subject_change)

        ttk.Label(sub_group, text="Game Sessions (bids_fnf):").pack(anchor="w", pady=(4, 2))
        self.sessions_box = tk.Frame(sub_group, bg=self.card_bg, bd=1, relief=tk.GROOVE)
        self.sessions_box.pack(fill=tk.X, pady=4)

        btn_row = ttk.Frame(sub_group)
        btn_row.pack(fill=tk.X, pady=(4, 6))
        ttk.Button(btn_row, text="Select All", command=self._select_all_sessions, width=14).pack(side=tk.LEFT, padx=(0, 4))
        ttk.Button(btn_row, text="Clear All", command=self._clear_all_sessions, width=14).pack(side=tk.LEFT)

        alg_group = ttk.LabelFrame(scrollable_frame, text=" 2. Decoding Algorithm ", padding=10)
        alg_group.pack(fill=tk.X, padx=8, pady=6)

        # Get pipelines from analyze_fnf_bci
        self.pipelines = get_pipelines()
        alg_options = ["all"] + list(self.pipelines.keys())
        self.alg_display_map = {
            "all": "⚡ Benchmark All (Auto-Select Best)",
            **{k: f"{k}" for k in self.pipelines.keys()}
        }
        self.display_to_key = {v: k for k, v in self.alg_display_map.items()}

        ttk.Label(alg_group, text="Algorithm:").pack(anchor="w")
        self.alg_combo = ttk.Combobox(
            alg_group,
            values=list(self.alg_display_map.values()),
            state="readonly",
            font=("Segoe UI", 9)
        )
        self.alg_combo.pack(fill=tk.X, pady=(2, 6))

        param_row = ttk.Frame(alg_group)
        param_row.pack(fill=tk.X, pady=(4, 2))
        ttk.Label(param_row, text="CV Folds:").pack(side=tk.LEFT, padx=(0, 4))
        ttk.Spinbox(param_row, from_=2, to=10, textvariable=self.cv_folds, width=4).pack(side=tk.LEFT, padx=(0, 16))

        ttk.Label(alg_group, text="Custom Model Tag (optional):").pack(anchor="w", pady=(6, 2))
        ttk.Entry(alg_group, textvariable=self.model_name_var).pack(fill=tk.X)

        train_box = ttk.Frame(scrollable_frame)
        train_box.pack(fill=tk.X, padx=8, pady=(12, 10))
        self.train_btn = ttk.Button(
            train_box,
            text="▶  TRAIN MODEL & EVALUATE",
            style="Accent.TButton",
            command=self._start_training_thread
        )
        self.train_btn.pack(fill=tk.X, ipady=6)

    def _build_right_dashboard(self, parent):
        metrics_group = ttk.LabelFrame(parent, text=" Evaluation Metrics (Stratified CV) ", padding=10)
        metrics_group.pack(fill=tk.X, padx=10, pady=8)

        m_cards_frame = ttk.Frame(metrics_group)
        m_cards_frame.pack(fill=tk.X, pady=4)

        self.card_acc = self._create_metric_card(m_cards_frame, "CV ACCURACY", "--")
        self.card_f1 = self._create_metric_card(m_cards_frame, "MACRO F1", "--")
        self.card_trials = self._create_metric_card(m_cards_frame, "TOTAL TRIALS", "--")

        self.progress_bar = ttk.Progressbar(metrics_group, orient="horizontal", mode="determinate")
        self.progress_bar.pack(fill=tk.X, pady=(8, 2))
        self.progress_lbl = ttk.Label(metrics_group, text="Ready to train.", style="Muted.TLabel")
        self.progress_lbl.pack(anchor="w")

        log_group = ttk.LabelFrame(parent, text=" Live Execution Console & Test Results ", padding=8)
        log_group.pack(fill=tk.BOTH, expand=True, padx=10, pady=4)

        self.log_text = scrolledtext.ScrolledText(
            log_group,
            wrap=tk.WORD,
            bg="#0d1117",
            fg="#f0f6fc",
            insertbackground="#ff7b72",
            selectbackground="#1f6feb",
            selectforeground="#ffffff",
            font=("Consolas", 10),
            height=12
        )
        self.log_text.pack(fill=tk.BOTH, expand=True)

        self.log_text.tag_config("success", foreground="#3fb950", font=("Consolas", 10, "bold"))
        self.log_text.tag_config("info", foreground="#ff7b72")
        self.log_text.tag_config("header", foreground="#e3b341", font=("Consolas", 10, "bold"))
        self.log_text.tag_config("error", foreground="#f85149", font=("Consolas", 10, "bold"))
        self.log_text.tag_config("metric", foreground="#bc8cff", font=("Consolas", 10, "bold"))

        rt_group = ttk.LabelFrame(parent, text=" Real-Time Pipeline Bridge (Godot FNF) ", padding=10)
        rt_group.pack(fill=tk.X, padx=10, pady=(4, 10))

        model_row = ttk.Frame(rt_group)
        model_row.pack(fill=tk.X, pady=(0, 4))

        ttk.Label(model_row, text="Active Model:", font=("Segoe UI", 9, "bold")).pack(side=tk.LEFT, padx=(0, 6))
        self.model_combo = ttk.Combobox(
            model_row,
            textvariable=self.selected_model_var,
            state="readonly",
            font=("Segoe UI", 9)
        )
        self.model_combo.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 6))
        self.model_combo.bind("<<ComboboxSelected>>", self._on_model_selection_change)

        ttk.Button(model_row, text="🔄", width=3, command=self._refresh_available_models).pack(side=tk.LEFT, padx=(0, 4))

        self.model_info_lbl = tk.Label(
            rt_group,
            text="",
            bg=self.panel_bg,
            fg="#ff7b72",
            font=("Segoe UI", 9, "italic"),
            anchor="w",
            justify=tk.LEFT
        )
        self.model_info_lbl.pack(fill=tk.X, pady=(0, 6))

        rt_opts_row = ttk.Frame(rt_group)
        rt_opts_row.pack(fill=tk.X, pady=(0, 6))

        ttk.Label(rt_opts_row, text="Source:").pack(side=tk.LEFT, padx=(0, 4))
        self.rt_source_combo = ttk.Combobox(
            rt_opts_row,
            textvariable=self.rt_source,
            values=["simulator", "lsl"],
            state="readonly",
            width=11
        )
        self.rt_source_combo.pack(side=tk.LEFT, padx=(0, 14))
        
        ttk.Label(rt_opts_row, text="Threshold:").pack(side=tk.LEFT, padx=(0, 4))
        ttk.Spinbox(
            rt_opts_row,
            from_=0.25,
            to=0.90,
            increment=0.05,
            textvariable=self.rt_threshold,
            width=5
        ).pack(side=tk.LEFT, padx=(0, 14))

        ttk.Checkbutton(rt_opts_row, text="Auto-Send Godot (UDP 5005)", variable=self.rt_autosend).pack(side=tk.LEFT)

        rt_btn_row = ttk.Frame(rt_group)
        rt_btn_row.pack(fill=tk.X, pady=(6, 2))

        self.launch_rt_btn = ttk.Button(
            rt_btn_row,
            text="🚀  LAUNCH REAL-TIME FNF PIPELINE",
            style="Success.TButton",
            command=self._launch_realtime_pipeline
        )
        self.launch_rt_btn.pack(side=tk.LEFT, fill=tk.X, expand=True, ipady=5, padx=(0, 6))

        self.stop_rt_btn = ttk.Button(
            rt_btn_row,
            text="⏹ Stop Pipeline",
            style="Danger.TButton",
            command=self._stop_realtime_pipeline,
            state="disabled"
        )
        self.stop_rt_btn.pack(side=tk.RIGHT, ipady=5)

    def _create_metric_card(self, parent, title, initial_val):
        card = tk.Frame(
            parent,
            bg=self.card_bg,
            highlightbackground="#30363d",
            highlightthickness=1,
            bd=0,
            padx=14,
            pady=8
        )
        card.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=4)

        t_lbl = tk.Label(
            card,
            text=title,
            font=("Segoe UI", 9, "bold"),
            bg=self.card_bg,
            fg=self.metric_title_color
        )
        t_lbl.pack(anchor="w")

        v_lbl = tk.Label(
            card,
            text=initial_val,
            font=("Segoe UI", 16, "bold"),
            bg=self.card_bg,
            fg=self.success_color
        )
        v_lbl.pack(anchor="w", pady=(3, 0))
        return v_lbl

    def _init_defaults(self):
        subjects = dataset.get_available_subjects()
        self.sub_combo['values'] = subjects
        if subjects:
            self.selected_subject.set(subjects[0])
            self._on_subject_change()

        self.alg_combo.current(0)

    def _on_subject_change(self, event=None):
        sub_id = self.selected_subject.get()
        if not sub_id:
            return

        sessions = dataset.get_available_sessions(sub_id)
        self._refresh_session_checkboxes(sessions)

    def _refresh_session_checkboxes(self, sessions):
        for widget in self.sessions_box.winfo_children():
            widget.destroy()

        self.session_vars.clear()
        
        if not sessions:
            lbl = tk.Label(self.sessions_box, text="No sessions found.", bg=self.card_bg, fg=self.muted_fg, font=("Segoe UI", 9))
            lbl.pack(padx=8, pady=6)
            return

        for ses in sessions:
            var = tk.BooleanVar(value=True)
            self.session_vars[ses] = var

            cb_frame = tk.Frame(self.sessions_box, bg=self.card_bg, pady=3, padx=4)
            cb_frame.pack(fill=tk.X, padx=4, pady=2)

            cb = tk.Checkbutton(
                cb_frame,
                text=f"ses-{ses}",
                variable=var,
                bg=self.card_bg,
                fg="#ffffff",
                selectcolor="#1f6feb",
                activebackground=self.card_bg,
                activeforeground="#ff7b72",
                font=("Segoe UI", 10, "bold")
            )
            cb.pack(side=tk.LEFT)

    def _select_all_sessions(self):
        for v in self.session_vars.values():
            v.set(True)

    def _clear_all_sessions(self):
        for v in self.session_vars.values():
            v.set(False)

    def _log(self, text, end="\n"):
        self.log_queue.put(text + end)

    def _start_log_consumer(self):
        def check_logs():
            while not self.log_queue.empty():
                msg = self.log_queue.get_nowait()
                tag = None
                s_strip = msg.strip()
                if s_strip.startswith("[+]"):
                    tag = "success"
                elif s_strip.startswith("[*]") or s_strip.startswith("•"):
                    tag = "info"
                elif s_strip.startswith("===") or s_strip.startswith("---") or "SELECTED" in s_strip:
                    tag = "header"
                elif "[ERROR]" in s_strip or "error" in s_strip.lower():
                    tag = "error"
                elif "accuracy" in s_strip.lower() or "macro f1" in s_strip.lower():
                    tag = "metric"

                if tag:
                    self.log_text.insert(tk.END, msg, tag)
                else:
                    self.log_text.insert(tk.END, msg)
                self.log_text.see(tk.END)
            self.root.after(80, check_logs)
        self.root.after(80, check_logs)

    def _start_training_thread(self):
        selected_sessions = [ses for ses, var in self.session_vars.items() if var.get()]

        if not selected_sessions:
            messagebox.showwarning(
                "No Data Selected",
                "Please select at least one session to train."
            )
            return

        self.train_btn.config(state="disabled")
        self.progress_bar['value'] = 0
        self.progress_lbl.config(text="Starting training...")

        worker = threading.Thread(
            target=self._run_training_worker,
            args=(
                self.selected_subject.get(),
                selected_sessions,
                self.display_to_key.get(self.alg_combo.get(), "CSP + LDA"),
                self.cv_folds.get(),
                self.model_name_var.get()
            ),
            daemon=True
        )
        worker.start()

    def _run_training_worker(
        self, sub_id, session_ids, alg_key, n_splits, custom_tag
    ):
        try:
            self._log("=" * 72)
            self._log(f"[*] STARTING MODEL TRAINING STUDIO")
            self._log(f" • Subject        : sub-{sub_id}")
            self._log(f" • Game Sessions  : {session_ids}")
            self._log(f" • Algorithm      : {alg_key.upper()}")
            self._log("=" * 72)

            def progress_cb(msg, frac):
                self._log(f"[*] {msg}")
                self.root.after(0, lambda: self._update_progress(frac * 100, msg))

            # Load data
            X_im, y_im = dataset.load_dataset_sessions(
                sub_id,
                session_ids,
                progress_callback=progress_cb
            )

            self._log(f"\n[+] Total Pooled Trials: {len(y_im)} | Epoch Shape: {X_im.shape}")
            class_counts = dict(pd.Series(y_im).value_counts())
            self._log(f"    Class Distribution (0:Left, 1:Right, 2:Up, 3:Down, 4:Rest): {class_counts}")

            self.root.after(0, lambda: self.card_trials.config(text=str(len(y_im))))

            if alg_key == "all":
                candidates = list(self.pipelines.keys())
            else:
                candidates = [alg_key]

            cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
            results = {}

            total_algs = len(candidates)
            for a_idx, c_key in enumerate(candidates):
                self._log(f"\n---> Evaluating [{c_key}] via {n_splits}-Fold CV...")

                accs, f1s = [], []
                oof_preds = np.zeros_like(y_im)

                # Fetch fresh pipeline instance per fold using get_pipelines
                
                for fold, (tr_idx, te_idx) in enumerate(cv.split(X_im, y_im)):
                    clf = get_pipelines()[c_key]
                    
                    clf.fit(X_im[tr_idx], y_im[tr_idx])
                    p_te = clf.predict(X_im[te_idx])
                    oof_preds[te_idx] = p_te

                    acc = accuracy_score(y_im[te_idx], p_te)
                    accs.append(acc)
                    f1s.append(f1_score(y_im[te_idx], p_te, average="macro"))

                mean_acc = float(np.mean(accs))
                std_acc = float(np.std(accs))
                mean_f1 = float(np.mean(f1s))
                cm = confusion_matrix(y_im, oof_preds).tolist()

                self._log(f"     [+] Accuracy : {mean_acc * 100:.2f}% ± {std_acc * 100:.2f}%")
                self._log(f"     [+] Macro F1 : {mean_f1:.3f}")

                results[c_key] = {
                    'name': c_key,
                    'mean_acc': mean_acc,
                    'std_acc': std_acc,
                    'mean_f1': mean_f1,
                    'cm': cm
                }

                progress = 0.75 + ((a_idx + 1) / total_algs) * 0.20
                self.root.after(0, lambda p=progress, k=c_key: self._update_progress(p * 100, f"Evaluated {k}"))

            # Pick best model
            best_key = max(results.keys(), key=lambda k: results[k]['mean_acc'])
            best_res = results[best_key]

            self._log("\n" + "=" * 72)
            self._log(f"[*] BEST MODEL SELECTED: {best_res['name']} ({best_res['mean_acc']*100:.2f}%)")
            self._log("=" * 72)

            # Update metrics cards
            self.root.after(0, lambda: self._update_metrics(best_res))

            # Train Final Model on All Data
            self._log(f"[*] Training final production model on full {len(y_im)} trials...")
            final_model = get_pipelines()[best_key]

            final_model.fit(X_im, y_im)
            self_acc = float(accuracy_score(y_im, final_model.predict(X_im)))
            self._log(f"[+] Final Model Fit Complete. Self-Accuracy: {self_acc*100:.2f}%")

            # Save Model Artifacts
            models_dir = _current_dir / "models"
            models_dir.mkdir(parents=True, exist_ok=True)

            if custom_tag.strip():
                tag = custom_tag.strip()
            else:
                tag = f"sub{sub_id}_{best_key.replace(' ', '_')}_{len(session_ids)}ses"

            model_file = models_dir / f"fnf_model_{tag}.joblib"
            report_file = models_dir / f"fnf_report_{tag}.json"

            export_dict = {
                'model': final_model,
                'model_name': f"{best_key.upper()}_FNF",
                'algorithm_name': best_res['name'],
                'subject': sub_id,
                'sessions': session_ids,
                'classes': ['Left', 'Right', 'Up', 'Down', 'Rest'],
                'n_trials': len(y_im),
                'metrics': {
                    'cv_accuracy_mean': best_res['mean_acc'],
                    'cv_accuracy_std': best_res['std_acc'],
                    'cv_f1_macro': best_res['mean_f1'],
                    'self_accuracy': self_acc
                },
                'confusion_matrix': best_res['cm']
            }

            joblib.dump(export_dict, model_file)
            self._log(f"[+] Exported joblib artifact: {model_file}")

            with open(report_file, 'w', encoding='utf-8') as f:
                json.dump(export_dict, f, indent=2, default=str)
            self._log(f"[+] Exported JSON report: {report_file}")

            self.last_trained_model_path = model_file
            self.root.after(0, lambda: self._on_training_complete(model_file))

        except Exception as e:
            self._log(f"\n[ERROR] Training failed: {e}")
            import traceback
            self._log(traceback.format_exc())
            self.root.after(0, lambda: messagebox.showerror("Training Error", str(e)))
            self.root.after(0, lambda: self._update_progress(0, "Error during training."))
        finally:
            self.root.after(0, lambda: self.train_btn.config(state="normal"))

    def _update_progress(self, val, text):
        self.progress_bar['value'] = val
        self.progress_lbl.config(text=text)

    def _update_metrics(self, res):
        self.card_acc.config(text=f"{res['mean_acc']*100:.1f}%")
        self.card_f1.config(text=f"{res['mean_f1']:.3f}")

    def _on_training_complete(self, model_file):
        self._update_progress(100, f"Training complete! Saved to {model_file.name}")
        self._refresh_available_models()
        for lbl, path in self.available_models.items():
            if path.name == model_file.name:
                self.selected_model_var.set(lbl)
                self._on_model_selection_change()
                break
        messagebox.showinfo(
            "Training Complete",
            f"Successfully trained and exported model!\n\nFile: {model_file.name}\n\nModel is now selected as the Active Model."
        )

    def _refresh_available_models(self):
        self.available_models.clear()
        search_dirs = [
            _current_dir / "models"
        ]

        found = []
        for sdir in search_dirs:
            if sdir.exists():
                for ext in ["*.joblib", "*.pkl"]:
                    for f in sorted(sdir.glob(ext)):
                        label = f"{f.stem}  ({f.parent.name})"
                        if label not in self.available_models:
                            self.available_models[label] = f
                            found.append(label)

        self.model_combo['values'] = found
        if found:
            target_label = found[0]
            if self.last_trained_model_path:
                for lbl, path in self.available_models.items():
                    if path.name == self.last_trained_model_path.name:
                        target_label = lbl
                        break
            self.selected_model_var.set(target_label)
            self._on_model_selection_change()
        else:
            self.selected_model_var.set("")
            self.model_info_lbl.config(text="No pre-trained models found. Train a model above.")

    def _on_model_selection_change(self, event=None):
        label = self.selected_model_var.get()
        model_path = self.available_models.get(label)
        if not model_path or not Path(model_path).exists():
            return

        try:
            data = joblib.load(model_path)
            if isinstance(data, dict):
                sub = data.get('subject', '?')
                ses = data.get('sessions', '?')
                alg = data.get('algorithm_name') or data.get('model_name', '?')
                metrics = data.get('metrics', {})
                cv_acc = metrics.get('cv_accuracy_mean')
                n_trials = data.get('n_trials', '?')
                acc_str = f"{cv_acc*100:.1f}%" if cv_acc is not None else "N/A"

                self.model_info_lbl.config(
                    text=f"[Active Model] {model_path.name}  |  sub-{sub}  |  ses: {ses}  |  {alg}  |  CV Acc: {acc_str} ({n_trials} trials)"
                )
                if cv_acc is not None:
                    self.card_acc.config(text=acc_str)
                if 'cv_f1_macro' in metrics:
                    self.card_f1.config(text=f"{metrics['cv_f1_macro']:.3f}")
                if n_trials != '?':
                    self.card_trials.config(text=str(n_trials))
                return
        except Exception:
            pass

        self.model_info_lbl.config(text=f"[Active Model] {model_path.name} ({model_path.parent.name})")

    def _launch_realtime_pipeline(self):
        selected_display = self.selected_model_var.get()
        model_path = self.available_models.get(selected_display)
        if not model_path or not Path(model_path).exists():
            model_path = self.last_trained_model_path

        if not model_path or not Path(model_path).exists():
            messagebox.showwarning("No Model Found", "Please select or train a model first before launching the real-time pipeline.")
            return

        sub_id = self.selected_subject.get() or "01"
        active_sessions = [ses for ses, var in self.session_vars.items() if var.get()]
        ses_id = active_sessions[-1] if active_sessions else "07"

        source = self.rt_source.get()
        threshold = str(self.rt_threshold.get())
        auto_send_flag = "--auto-send" if self.rt_autosend.get() else "--no-auto-send"

        venv_py = Path(sys.executable)
        
        main_py = _scripts_dir / "bridges" / "fnf_bci_bridge.py"
        if not main_py.exists():
            messagebox.showerror("Pipeline Script Missing", f"Could not locate the FNF real-time bridge at {main_py}")
            return

        cmd = [
            str(venv_py),
            str(main_py),
            "--source", source,
            "--threshold", threshold,
            "--sub", sub_id,
            "--ses", ses_id,
            "--model", str(model_path),
            "--port", "5005",
            auto_send_flag
        ]

        self._log("\n" + "=" * 72)
        self._log(f"[*] LAUNCHING REAL-TIME FNF BCI PIPELINE...")
        self._log(f"    Command: {' '.join(cmd)}")
        self._log("=" * 72)

        try:
            run_env = os.environ.copy()
            run_env["PYTHONPATH"] = str(_current_dir) + os.pathsep + run_env.get("PYTHONPATH", "")

            creationflags = subprocess.CREATE_NEW_CONSOLE if sys.platform == "win32" else 0
            self.realtime_process = subprocess.Popen(
                cmd,
                cwd=str(_scripts_dir),
                env=run_env,
                creationflags=creationflags
            )
            self.launch_rt_btn.config(state="disabled")
            self.stop_rt_btn.config(state="normal")
            self._log(f"[+] Pipeline started (PID {self.realtime_process.pid}).")
        except Exception as e:
            self._log(f"[ERROR] Failed to start pipeline: {e}")
            messagebox.showerror("Pipeline Launch Error", str(e))

    def _stop_realtime_pipeline(self):
        if self.realtime_process:
            try:
                self.realtime_process.terminate()
                self._log("[*] Real-Time Pipeline stopped.")
            except Exception as e:
                self._log(f"[!] Warning stopping pipeline: {e}")
            self.realtime_process = None

        self.launch_rt_btn.config(state="normal")
        self.stop_rt_btn.config(state="disabled")

def launch_gui():
    root = tk.Tk()
    app = FNFTrainingStudioGUI(root)
    root.mainloop()

if __name__ == "__main__":
    launch_gui()
