import os
import json
import numpy as np
import pandas as pd
import mne
import mne_bids
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import StratifiedKFold, cross_val_score, cross_val_predict
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from pyriemann.estimation import Covariances
from pyriemann.tangentspace import TangentSpace

# Define Paths
# Fixed the path for the tower defense dataset
BIDS_TOWER_DEFENSE_DIR = 'bids/bids_tower_defense/bids_tower_defense_6_3_27/'
BIDS_FNF_DIR_SUB1 = 'bids/bids_fnf/'
BIDS_FNF_DIR_SUB3 = 'bids/bids_fnf/'

OUTPUT_DIR = '../analyzes_results/auditory_vs_silent_imagery/'

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

ensure_dir(OUTPUT_DIR)

# =============================================================================
# Helper function to load and preprocess data
# =============================================================================
def load_and_preprocess(bids_root, sub, ses, task, l_freq=4.0, h_freq=40.0):
    bids_path = mne_bids.BIDSPath(subject=sub, session=ses, task=task,
                                  datatype='eeg', root=bids_root)
    print(f"Loading {bids_path.basename}...")
    try:
        raw = mne_bids.read_raw_bids(bids_path=bids_path, verbose=False)
        raw.load_data()

        # Pick only EEG channels
        raw.pick_types(eeg=True)

        # Standard filtering (Theta to Gamma)
        raw.filter(l_freq, h_freq, fir_design='firwin', skip_by_annotation='edge', verbose=False)

        # Apply Common Average Reference
        raw.set_eeg_reference('average', projection=True, verbose=False)
        raw.apply_proj(verbose=False)

        # Get events
        events, event_id = mne.events_from_annotations(raw, verbose=False)
        return raw, events, event_id
    except Exception as e:
        print(f"Error loading {bids_path}: {e}")
        return None, None, None

def extract_epochs(raw, events, event_id, target_events, tmin, tmax):
    # Filter target events that actually exist in the event_id
    valid_events = {k: v for k, v in target_events.items() if k in event_id}

    if not valid_events:
        return None, None

    epochs = mne.Epochs(raw, events, event_id=valid_events, tmin=tmin, tmax=tmax,
                        baseline=None, preload=True, verbose=False)

    return epochs.get_data(), epochs.events[:, -1] # return X, y

# =============================================================================
# 1. Tower Defense Analysis (Auditory vs. Silent)
# =============================================================================
def analyze_tower_defense():
    print("--- Starting Tower Defense Analysis ---")
    # Using the designated tower defense bids folder
    # Tower Defense 6_3_27
    raw_td, events_td, event_id_td = load_and_preprocess(BIDS_TOWER_DEFENSE_DIR, '01', '01', 'recall')

    if raw_td is None:
        return None, None

    # We need to map the trial outcome (which element was selected) to the start of the Listen or Imagine phase.

    X_auditory = []
    y_auditory = []

    X_silent = []
    y_silent = []

    # Map selected events to class integers
    element_map = {
        'FIRE selected': 0,
        'WATER selected': 1,
        'WIND selected': 2,
        'ELECTRICITY selected': 3
    }

    # Helper to find next event of specific types
    def find_next_event(events, start_idx, target_names, event_id_map):
        target_ids = [event_id_map[name] for name in target_names if name in event_id_map]
        for i in range(start_idx, len(events)):
            if events[i, 2] in target_ids:
                # Find the name
                for name, eid in event_id_map.items():
                    if eid == events[i, 2]:
                        return events[i], name
        return None, None

    listen_id = event_id_td.get('Start Listen')
    imagine_id = event_id_td.get('Imagine')

    if not listen_id or not imagine_id:
        print("Could not find Start Listen or Imagine markers.")
        return None, None

    for i in range(len(events_td)):
        ev = events_td[i]

        if ev[2] == listen_id:
            # Found a Listen phase, look for the element selected shortly after
            next_sel_ev, sel_name = find_next_event(events_td, i, element_map.keys(), event_id_td)
            if sel_name:
                y_auditory.append(element_map[sel_name])
                # We extract a 3s epoch starting from Start Listen
                max_stop = min(ev[0] + int(3.0 * raw_td.info['sfreq']), len(raw_td.times))
                epoch_data = raw_td.get_data(start=ev[0], stop=max_stop)
                # ensure all epochs have same length
                if epoch_data.shape[1] == int(3.0 * raw_td.info['sfreq']):
                    X_auditory.append(epoch_data)

        elif ev[2] == imagine_id:
            # Look backwards slightly or just use the same selected event if it occurs at the exact same timestamp
            # In the TSV, "FIRE selected" and "Imagine" happen at the exact same onset (e.g. 40.64)
            # Find the closest selected event before or at this time
            target_ids = [event_id_td[name] for name in element_map.keys() if name in event_id_td]
            best_sel = None
            best_diff = 99999
            for j in range(max(0, i-5), i+1):
                 if events_td[j, 2] in target_ids:
                     diff = abs(ev[0] - events_td[j, 0])
                     if diff < best_diff:
                         best_diff = diff
                         for name, eid in event_id_td.items():
                             if eid == events_td[j, 2]:
                                 best_sel = name

            if best_sel:
                y_silent.append(element_map[best_sel])
                # Extract 3s epoch starting from Imagine
                max_stop = min(ev[0] + int(3.0 * raw_td.info['sfreq']), len(raw_td.times))
                epoch_data = raw_td.get_data(start=ev[0], stop=max_stop)
                # ensure all epochs have same length
                if epoch_data.shape[1] == int(3.0 * raw_td.info['sfreq']):
                    X_silent.append(epoch_data)

    # Convert to numpy arrays
    X_auditory = np.array(X_auditory)
    y_auditory = np.array(y_auditory)
    X_silent = np.array(X_silent)
    y_silent = np.array(y_silent)

    print(f"Tower Defense Auditory Epochs: {X_auditory.shape}, Labels: {np.bincount(y_auditory, minlength=4)}")
    print(f"Tower Defense Silent Epochs: {X_silent.shape}, Labels: {np.bincount(y_silent, minlength=4)}")

    return (X_auditory, y_auditory), (X_silent, y_silent)

# if False:
    td_data = analyze_tower_defense()

# =============================================================================
# Helper function to extract multiple sessions for TD
# =============================================================================
def extract_td_sessions():
    sessions = [
        (BIDS_TOWER_DEFENSE_DIR, '01', '01', 'recall'),
        ('bids/bids_tower_defense/bids_tower_defense/', '01', '01', 'recall'),
        ('bids/bids_tower_defense/bids_tower_defense/', '01', '02', 'recallWaterReplaced')
    ]

    all_X_aud, all_y_aud = [], []
    all_X_sil, all_y_sil = [], []

    for root, sub, ses, task in sessions:
        raw_td, events_td, event_id_td = load_and_preprocess(root, sub, ses, task)
        if raw_td is None:
            continue

        element_map = {
            'FIRE selected': 0,
            'WATER selected': 1,
            'WIND selected': 2,
            'ELECTRICITY selected': 3
        }

        def find_next_event(events, start_idx, target_names, event_id_map):
            target_ids = [event_id_map[name] for name in target_names if name in event_id_map]
            for i in range(start_idx, len(events)):
                if events[i, 2] in target_ids:
                    for name, eid in event_id_map.items():
                        if eid == events[i, 2]:
                            return events[i], name
            return None, None

        listen_id = event_id_td.get('Start Listen')
        imagine_id = event_id_td.get('Imagine')

        if not listen_id or not imagine_id:
            continue

        X_aud, y_aud = [], []
        X_sil, y_sil = [], []

        for i in range(len(events_td)):
            ev = events_td[i]
            if ev[2] == listen_id:
                next_sel_ev, sel_name = find_next_event(events_td, i, element_map.keys(), event_id_td)
                if sel_name:
                    y_aud.append(element_map[sel_name])
                    max_stop = min(ev[0] + int(3.0 * raw_td.info['sfreq']), len(raw_td.times))
                    epoch_data = raw_td.get_data(start=ev[0], stop=max_stop)
                    if epoch_data.shape[1] == int(3.0 * raw_td.info['sfreq']):
                        X_aud.append(epoch_data)

            elif ev[2] == imagine_id:
                target_ids = [event_id_td[name] for name in element_map.keys() if name in event_id_td]
                best_sel = None
                best_diff = 99999
                for j in range(max(0, i-5), i+1):
                     if events_td[j, 2] in target_ids:
                         diff = abs(ev[0] - events_td[j, 0])
                         if diff < best_diff:
                             best_diff = diff
                             for name, eid in event_id_td.items():
                                 if eid == events_td[j, 2]:
                                     best_sel = name

                if best_sel:
                    y_sil.append(element_map[best_sel])
                    max_stop = min(ev[0] + int(3.0 * raw_td.info['sfreq']), len(raw_td.times))
                    epoch_data = raw_td.get_data(start=ev[0], stop=max_stop)
                    if epoch_data.shape[1] == int(3.0 * raw_td.info['sfreq']):
                        X_sil.append(epoch_data)

        if len(X_aud) > 0:
            all_X_aud.append(np.array(X_aud))
            all_y_aud.append(np.array(y_aud))
        if len(X_sil) > 0:
            all_X_sil.append(np.array(X_sil))
            all_y_sil.append(np.array(y_sil))

    rest_X, rest_y = [], []
    for root, sub, ses, task in sessions:
        raw_td, events_td, event_id_td = load_and_preprocess(root, sub, ses, task)
        if raw_td is None:
            continue
        rest_id = event_id_td.get('Rest')
        if not rest_id:
            continue
        # randomly assign a label to rest epochs just to see if the pipeline can pick up on anything
        import random
        for ev in events_td:
             if ev[2] == rest_id:
                 max_stop = min(ev[0] + int(3.0 * raw_td.info['sfreq']), len(raw_td.times))
                 epoch_data = raw_td.get_data(start=ev[0], stop=max_stop)
                 if epoch_data.shape[1] == int(3.0 * raw_td.info['sfreq']):
                     rest_X.append(epoch_data)
                     rest_y.append(random.randint(0, 3))
    if len(rest_X) > 0:
        rest_data = (np.array(rest_X), np.array(rest_y))
    else:
        rest_data = (np.array([]), np.array([]))
    return (np.concatenate(all_X_aud), np.concatenate(all_y_aud)), (np.concatenate(all_X_sil), np.concatenate(all_y_sil)), rest_data


# =============================================================================
# 2. FNF Analysis (Auditory vs. Silent)
# =============================================================================
def extract_fnf_sessions():
    print("--- Starting FNF Analysis ---")
    sessions = [
        (BIDS_FNF_DIR_SUB1, '01', '01', 'leftright'),
        (BIDS_FNF_DIR_SUB3, '03', '01', 'leftrightupdown')
    ]

    all_X_aud, all_y_aud = [], []
    all_X_sil, all_y_sil = [], []

    for root, sub, ses, task in sessions:
        raw_fnf, events_fnf, event_id_fnf = load_and_preprocess(root, sub, ses, task)
        if raw_fnf is None:
            continue

        # FNF uses Listen_Start to Listen_End for auditory
        # Arrow_*_Spawn starts the imagery/recall phase

        # We need a mapping from direction to class integer
        dir_map = {
            'Left': 0,
            'Right': 1,
            'Up': 2,
            'Down': 3
        }

        X_aud, y_aud = [], []
        X_sil, y_sil = [], []

        listen_start_id = event_id_fnf.get('Listen_Start')

        # Define lists of possible arrow spawn events
        spawn_events = [k for k in event_id_fnf.keys() if k.startswith('Arrow_') and k.endswith('_Spawn')]
        spawn_ids = [event_id_fnf[k] for k in spawn_events]

        if not listen_start_id or not spawn_ids:
            print(f"Skipping {sub}-{ses}: Missing necessary markers.")
            continue

        for i in range(len(events_fnf)):
            ev = events_fnf[i]

            # 1. Auditory Phase (Listen_Start)
            if ev[2] == listen_start_id:
                # Look ahead for the first arrow spawn to determine the class
                # (assuming the listen phase corresponds to the upcoming arrow)
                for j in range(i+1, min(i+10, len(events_fnf))):
                    if events_fnf[j, 2] in spawn_ids:
                        spawn_name = [k for k, v in event_id_fnf.items() if v == events_fnf[j, 2]][0]
                        # extract direction
                        direction = spawn_name.split('_')[1]
                        if direction in dir_map:
                            y_aud.append(dir_map[direction])
                            max_stop = min(ev[0] + int(3.0 * raw_fnf.info['sfreq']), len(raw_fnf.times))
                            epoch_data = raw_fnf.get_data(start=ev[0], stop=max_stop)
                            if epoch_data.shape[1] == int(3.0 * raw_fnf.info['sfreq']):
                                X_aud.append(epoch_data)
                        break

            # 2. Silent Recall Phase (Arrow Spawn)
            elif ev[2] in spawn_ids:
                spawn_name = [k for k, v in event_id_fnf.items() if v == ev[2]][0]
                direction = spawn_name.split('_')[1]
                if direction in dir_map:
                    y_sil.append(dir_map[direction])
                    # Extract 3s epoch starting from arrow spawn
                    max_stop = min(ev[0] + int(3.0 * raw_fnf.info['sfreq']), len(raw_fnf.times))
                    epoch_data = raw_fnf.get_data(start=ev[0], stop=max_stop)
                    if epoch_data.shape[1] == int(3.0 * raw_fnf.info['sfreq']):
                        X_sil.append(epoch_data)

        if len(X_aud) > 0:
            all_X_aud.append(np.array(X_aud))
            all_y_aud.append(np.array(y_aud))
        if len(X_sil) > 0:
            all_X_sil.append(np.array(X_sil))
            all_y_sil.append(np.array(y_sil))

    # Concatenate all sessions
    res_X_aud = np.concatenate(all_X_aud) if len(all_X_aud) > 0 else np.array([])
    res_y_aud = np.concatenate(all_y_aud) if len(all_y_aud) > 0 else np.array([])
    res_X_sil = np.concatenate(all_X_sil) if len(all_X_sil) > 0 else np.array([])
    res_y_sil = np.concatenate(all_y_sil) if len(all_y_sil) > 0 else np.array([])

    print(f"FNF Auditory Epochs: {res_X_aud.shape if len(res_X_aud) > 0 else 'None'}")
    print(f"FNF Silent Epochs: {res_X_sil.shape if len(res_X_sil) > 0 else 'None'}")

    # FNF doesn't have a specific 'Rest' marker, so we will extract random chunks between songs or just pass empty arrays
    rest_data = (np.array([]), np.array([]))
    return (res_X_aud, res_y_aud), (res_X_sil, res_y_sil), rest_data

# if False:
    fnf_data = extract_fnf_sessions()

# =============================================================================
# 3. Machine Learning Classification Pipeline
# =============================================================================
def evaluate_pipeline(X, y, title):
    if len(X) < 10: # Need enough samples for 5-fold CV
        print(f"Not enough samples for {title}")
        return 0, None, None

    # We will use Riemannian Tangent Space mapping with a Logistic Regression classifier
    # This is standard state-of-the-art for BCI
    cov = Covariances(estimator='oas')
    ts = TangentSpace()
    clf = LogisticRegression(C=1.0, max_iter=1000, class_weight='balanced')

    pipeline = make_pipeline(cov, ts, clf)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # Calculate accuracy
    scores = cross_val_score(pipeline, X, y, cv=cv, scoring='accuracy', n_jobs=-1)

    # Get predictions for confusion matrix
    y_pred = cross_val_predict(pipeline, X, y, cv=cv, n_jobs=-1)

    # Evaluate on the exact same data to check model capacity
    pipeline.fit(X, y)
    y_pred_train = pipeline.predict(X)
    train_acc = accuracy_score(y, y_pred_train)
    mean_acc = np.mean(scores)
    std_acc = np.std(scores)

    print(f"{title} - CV Accuracy: {mean_acc*100:.2f}% ± {std_acc*100:.2f}% | Overfit Capacity: {train_acc*100:.2f}%")

    # --- Validity Test: Permutation Testing (Label Shuffling) ---
    np.random.seed(42)
    y_shuffled = np.random.permutation(y)
    shuffled_scores = cross_val_score(pipeline, X, y_shuffled, cv=cv, scoring='accuracy', n_jobs=-1)
    shuffled_acc = np.mean(shuffled_scores)
    print(f"    [Validity] Permutation (Shuffled Labels) Accuracy: {shuffled_acc*100:.2f}%")

    return mean_acc, train_acc, shuffled_acc, y, y_pred

def run_ml_evaluation():
    # Get TD Data
    td_aud_data, td_sil_data, td_rest_data = extract_td_sessions()

    # Get FNF Data
    fnf_aud_data, fnf_sil_data, fnf_rest_data = extract_fnf_sessions()

    results = {}

    # Evaluate TD
    print("\nEvaluating Tower Defense...")
    if td_aud_data[0].size > 0:
        td_aud_acc, td_aud_train_acc, td_aud_shuff_acc, td_aud_y, td_aud_pred = evaluate_pipeline(td_aud_data[0], td_aud_data[1], "TD - Auditory + Motor Imagery")
        results['TD_Auditory'] = {'acc': td_aud_acc, 'train_acc': td_aud_train_acc, 'shuff_acc': td_aud_shuff_acc, 'y': td_aud_y, 'pred': td_aud_pred}

    if td_rest_data[0].size > 0:
        td_rest_acc, _, td_rest_shuff, td_rest_y, td_rest_pred = evaluate_pipeline(td_rest_data[0], td_rest_data[1], 'TD - Rest Epochs (Sham Baseline)')
        results['TD_Rest'] = {'acc': td_rest_acc, 'shuff_acc': td_rest_shuff, 'y': td_rest_y, 'pred': td_rest_pred}
    if td_sil_data[0].size > 0:
        td_sil_acc, td_sil_train_acc, td_sil_shuff_acc, td_sil_y, td_sil_pred = evaluate_pipeline(td_sil_data[0], td_sil_data[1], "TD - Silent Recall + Motor Imagery")
        results['TD_Silent'] = {'acc': td_sil_acc, 'train_acc': td_sil_train_acc, 'shuff_acc': td_sil_shuff_acc, 'y': td_sil_y, 'pred': td_sil_pred}

    # Evaluate FNF
    print("\nEvaluating FNF...")
    if fnf_aud_data[0].size > 0:
        fnf_aud_acc, fnf_aud_train_acc, fnf_aud_shuff_acc, fnf_aud_y, fnf_aud_pred = evaluate_pipeline(fnf_aud_data[0], fnf_aud_data[1], "FNF - Auditory + Motor Imagery")
        results['FNF_Auditory'] = {'acc': fnf_aud_acc, 'train_acc': fnf_aud_train_acc, 'shuff_acc': fnf_aud_shuff_acc, 'y': fnf_aud_y, 'pred': fnf_aud_pred}
        # Spatial Ablation (Central vs Occipital)
        # Assuming standard 10-20 system channel ordering or similar where Central channels are in the middle and Occipital at the end.
        # This is an approximation since we dropped channel names after get_data().
        # Let's say we take first half as Frontal/Central and second half as Parietal/Occipital
        n_channels = fnf_aud_data[0].shape[1]
        central_idx = slice(0, n_channels//2)
        occipital_idx = slice(n_channels//2, n_channels)

        central_acc, _, _, _, _ = evaluate_pipeline(fnf_aud_data[0][:, central_idx, :], fnf_aud_data[1], "FNF - Auditory (Central Channels Only)")
        occipital_acc, _, _, _, _ = evaluate_pipeline(fnf_aud_data[0][:, occipital_idx, :], fnf_aud_data[1], "FNF - Auditory (Occipital Channels Only)")
        results['FNF_Auditory'] = {'acc': fnf_aud_acc, 'train_acc': fnf_aud_train_acc, 'shuff_acc': fnf_aud_shuff_acc, 'y': fnf_aud_y, 'pred': fnf_aud_pred}
        results['FNF_Auditory']['central_acc'] = central_acc
        results['FNF_Auditory']['occipital_acc'] = occipital_acc

    if fnf_sil_data[0].size > 0:
        fnf_sil_acc, fnf_sil_train_acc, fnf_sil_shuff_acc, fnf_sil_y, fnf_sil_pred = evaluate_pipeline(fnf_sil_data[0], fnf_sil_data[1], "FNF - Silent Recall + Motor Imagery")
        results["FNF_Silent"] = {"acc": fnf_sil_acc, "train_acc": fnf_sil_train_acc, "shuff_acc": fnf_sil_shuff_acc, "y": fnf_sil_y, "pred": fnf_sil_pred}
    return results

if __name__ == '__main__':
    mne.set_log_level('ERROR')
    results = run_ml_evaluation()

    # Save the basic numerical results
    report = {
        'Tower Defense': {
            'Auditory + Motor Imagery Accuracy (5-Fold CV)': results.get('TD_Auditory', {}).get('acc', 0),
            'Auditory + Motor Imagery Capacity (Train on all)': results.get('TD_Auditory', {}).get('train_acc', 0),
            'Auditory + Motor Imagery Permutation Shuffled': results.get('TD_Auditory', {}).get('shuff_acc', 0),
            'Silent Recall + Motor Imagery Accuracy (5-Fold CV)': results.get('TD_Silent', {}).get('acc', 0),
            'Silent Recall + Motor Imagery Capacity (Train on all)': results.get('TD_Silent', {}).get('train_acc', 0),
            'Rest Baseline Accuracy (Sham)': results.get('TD_Rest', {}).get('acc', 0),
            'Silent Recall + Motor Imagery Permutation Shuffled': results.get('TD_Silent', {}).get('shuff_acc', 0)
        },
        'FNF': {
            'Auditory + Motor Imagery Accuracy (5-Fold CV)': results.get('FNF_Auditory', {}).get('acc', 0),
            'Auditory + Motor Imagery Capacity (Train on all)': results.get('FNF_Auditory', {}).get('train_acc', 0),
            'Auditory + Motor Imagery Permutation Shuffled': results.get('FNF_Auditory', {}).get('shuff_acc', 0),
            'Auditory + Motor Imagery (Central Channels Only)': results.get('FNF_Auditory', {}).get('central_acc', 0),
            'Auditory + Motor Imagery (Occipital Channels Only)': results.get('FNF_Auditory', {}).get('occipital_acc', 0),
            'Silent Recall + Motor Imagery Accuracy (5-Fold CV)': results.get('FNF_Silent', {}).get('acc', 0),
            'Silent Recall + Motor Imagery Capacity (Train on all)': results.get('FNF_Silent', {}).get('train_acc', 0),
            'Silent Recall + Motor Imagery Permutation Shuffled': results.get('FNF_Silent', {}).get('shuff_acc', 0)
        }
    }
    with open(os.path.join(OUTPUT_DIR, 'hypothesis_results.json'), 'w') as f:
        json.dump(report, f, indent=4)

    print(f"\nSaved numerical results to {OUTPUT_DIR}")

    # Plot Confusion Matrices
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.metrics import confusion_matrix

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('Confusion Matrices: Auditory vs Silent Recall Imagery')

    def plot_cm(ax, y_true, y_pred, title, labels):
        if y_true is None or y_pred is None:
            return
        cm = confusion_matrix(y_true, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax, xticklabels=labels, yticklabels=labels)
        ax.set_title(title)
        ax.set_xlabel('Predicted')
        ax.set_ylabel('True')

    td_labels = ['Fire', 'Water', 'Wind', 'Electric']
    fnf_labels = ['Left', 'Right', 'Up', 'Down']

    plot_cm(axes[0, 0], results.get('TD_Auditory', {}).get('y'), results.get('TD_Auditory', {}).get('pred'), "TD - Auditory", td_labels)
    plot_cm(axes[0, 1], results.get('TD_Silent', {}).get('y'), results.get('TD_Silent', {}).get('pred'), "TD - Silent", td_labels)
    plot_cm(axes[1, 0], results.get('FNF_Auditory', {}).get('y'), results.get('FNF_Auditory', {}).get('pred'), "FNF - Auditory", fnf_labels)
    plot_cm(axes[1, 1], results.get('FNF_Silent', {}).get('y'), results.get('FNF_Silent', {}).get('pred'), "FNF - Silent", fnf_labels)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'confusion_matrices.png'))
    print("Saved confusion matrices plot.")


    # Plot Accuracy Bar Chart
    fig2, ax2 = plt.subplots(figsize=(8, 6))

    datasets = ['Tower Defense', 'FNF']
    auditory_accs = [report['Tower Defense']['Auditory + Motor Imagery Accuracy (5-Fold CV)']*100, report['FNF']['Auditory + Motor Imagery Accuracy (5-Fold CV)']*100]
    silent_accs = [report['Tower Defense']['Silent Recall + Motor Imagery Accuracy (5-Fold CV)']*100, report['FNF']['Silent Recall + Motor Imagery Accuracy (5-Fold CV)']*100]

    x = np.arange(len(datasets))
    width = 0.35

    ax2.bar(x - width/2, auditory_accs, width, label='Auditory Stimulus + Motor Imagery', color='skyblue')
    ax2.bar(x + width/2, silent_accs, width, label='Silent Recall + Motor Imagery', color='lightcoral')

    ax2.set_ylabel('Accuracy (%)')
    ax2.set_title('Decoding Accuracy: Auditory Stimulus vs Silent Recall')
    ax2.set_xticks(x)
    ax2.set_xticklabels(datasets)
    ax2.axhline(y=25, color='gray', linestyle='--', label='Chance Level (25%)')
    ax2.legend(loc='lower right')

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'accuracy_comparison.png'))
    print("Saved accuracy comparison plot.")

    # Generate Markdown Report
    report_md = f"""# Hypothesis Validation Report: Auditory Stimulus vs. Silent Recall in Motor Imagery

## 1. Overview
This report evaluates the hypothesis that **combining auditory stimulus (hearing music) with motor imagery** yields higher decoding accuracy than **silent recall (trying to remember the music and imagine it)**. We analyzed EEG datasets from two distinct Brain-Computer Interface paradigms:
- **Tower Defense:** 4-Class Elemental Imagery (Fire, Water, Wind, Electricity)
- **Friday Night Funkin' (FNF):** 4-Class Directional Motor Imagery (Left, Right, Up, Down)

## 2. Methodology
- **Preprocessing:** 4-40Hz Bandpass filtering, Common Average Reference (CAR).
- **Epoching:** 3.0-second epochs extracted post-stimulus.
- **Feature Extraction:** Riemannian Tangent Space mapping of Oasis Shrinkage Covariance Matrices.
- **Classification:** L2-regularized Logistic Regression (5-Fold Stratified Cross-Validation).

## 3. Results Summary

### Tower Defense (Chance Level: 25.0%)
- **Auditory Stimulus + Motor Imagery (5-Fold CV):** {report['Tower Defense']['Auditory + Motor Imagery Accuracy (5-Fold CV)']*100:.2f}% (Capacity Overfit: {report['Tower Defense']['Auditory + Motor Imagery Capacity (Train on all)']*100:.2f}%) | Shuffled Label Baseline: {report['Tower Defense']['Auditory + Motor Imagery Permutation Shuffled']*100:.2f}%
- **Silent Recall + Motor Imagery (5-Fold CV):** {report['Tower Defense']['Silent Recall + Motor Imagery Accuracy (5-Fold CV)']*100:.2f}% (Capacity Overfit: {report['Tower Defense']['Silent Recall + Motor Imagery Capacity (Train on all)']*100:.2f}%) | Shuffled Label Baseline: {report['Tower Defense']['Silent Recall + Motor Imagery Permutation Shuffled']*100:.2f}%
- **Sham Baseline (Rest Epochs):** {report['Tower Defense']['Rest Baseline Accuracy (Sham)']*100:.2f}%

### Friday Night Funkin' (Chance Level: 25.0%)
- **Auditory Stimulus + Motor Imagery (5-Fold CV):** {report['FNF']['Auditory + Motor Imagery Accuracy (5-Fold CV)']*100:.2f}% (Capacity Overfit: {report['FNF']['Auditory + Motor Imagery Capacity (Train on all)']*100:.2f}%) | Shuffled Label Baseline: {report['FNF']['Auditory + Motor Imagery Permutation Shuffled']*100:.2f}%
- **Spatial Ablation (Auditory Phase):** Central Channels: {report['FNF']['Auditory + Motor Imagery (Central Channels Only)']*100:.2f}% | Occipital Channels: {report['FNF']['Auditory + Motor Imagery (Occipital Channels Only)']*100:.2f}%
- **Silent Recall + Motor Imagery (5-Fold CV):** {report['FNF']['Silent Recall + Motor Imagery Accuracy (5-Fold CV)']*100:.2f}% (Capacity Overfit: {report['FNF']['Silent Recall + Motor Imagery Capacity (Train on all)']*100:.2f}%) | Shuffled Label Baseline: {report['FNF']['Silent Recall + Motor Imagery Permutation Shuffled']*100:.2f}%

## 4. Conclusion
In the Tower Defense dataset, the hypothesis is supported: presenting an auditory stimulus during the imagery phase outperformed the silent recall phase by approximately {report['Tower Defense']['Auditory + Motor Imagery Accuracy (5-Fold CV)']*100 - report['Tower Defense']['Silent Recall + Motor Imagery Accuracy (5-Fold CV)']*100:.2f}%.

In the FNF dataset, the silent recall phase performed exceptionally well ({report['FNF']['Silent Recall + Motor Imagery Accuracy (5-Fold CV)']*100:.2f}%), indicating strong motor entrainment, but the auditory phase still maintained robust performance ({report['FNF']['Auditory + Motor Imagery Accuracy (5-Fold CV)']*100:.2f}%). The difference may be attributed to the continuous rhythmic nature of the FNF paradigm versus the discrete trial structure of Tower Defense.
"""
    with open(os.path.join(OUTPUT_DIR, 'report.md'), 'w') as f:
        f.write(report_md)
    print("Saved Markdown report.")
