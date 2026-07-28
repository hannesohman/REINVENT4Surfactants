#!/usr/bin/env python
"""
Thin wrapper around `python -m reinvent` that optionally applies the Loss
Modulation (LM) patch (reinvent_lm_patch.py) from Medina's master's thesis
"Uncertainty-aware reinforcement learning for chemical de novo design"
before running.

Behaves IDENTICALLY to `python -m reinvent <args>` unless
REINVENT_LM_ENABLED=1 is set in the environment -- in which case the RL loss
is reweighted per-sample by uncertainty (see reinvent_lm_patch.py), while the
scoring function / reward itself is completely unaffected.

This lets the exact same TOML/scoring setup run under Score Modulation (SM)
only, Loss Modulation (LM) only, or both (SM & LM) simultaneously -- SM is
controlled purely by WEIGHT_COMBOS weights (pCMC_Uncertainty/
SurfTen_Uncertainty weight > 0 or 0), and LM by this env var. No separate
implementation or TOML is needed for any combination.

Usage: identical to `python -m reinvent`, e.g.
    python workflow/reinvent_with_lm.py -l run.log run.toml                    # SM only (or neither, depending on weights)
    REINVENT_LM_ENABLED=1 python workflow/reinvent_with_lm.py -l run.log run.toml   # + LM
"""
import os
import sys
import runpy

if os.environ.get("REINVENT_LM_ENABLED", "").lower() in ("1", "true", "yes"):
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import reinvent_lm_patch  # noqa: F401 (import applies the monkeypatch)

runpy.run_module("reinvent", run_name="__main__")
