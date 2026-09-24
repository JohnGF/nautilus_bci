with open("scripts/analysis/auditory_priming_transfer.py", "r") as f:
    text = f.read()

import re
replacement = """In the **Tower Defense dataset**, the hypothesis was tested by mapping the discrete elemental motor imagery phases to the continuous full-length music listening dataset (`bids_music`). Because the Riemannian geometry of a 3.0s motor imagery task is significantly misaligned with the spatial covariance of a continuous, relaxed 2-hour listening session, the direct transfer using Tangent Space concatenation resulted in chance-level performance (~24% Primed vs ~22% Baseline). This indicates that while active, in-game auditory cues transfer well (as seen in the earlier 5-fold CV tests), passive continuous music listening is a distinct mental state that requires affine alignment (e.g., Riemannian Procrustes Analysis) before few-shot transfer can occur."""

text = re.sub(r"In the \*\*Tower Defense dataset\*\*, the priming effect is also evident at low \$k\$ \(jumping from ~22% to ~37%\)\. The absolute accuracy is lower overall, but the performance gap confirms that auditory transfer is beneficial when calibration data is scarce\.", replacement, text)

with open("scripts/analysis/auditory_priming_transfer.py", "w") as f:
    f.write(text)
