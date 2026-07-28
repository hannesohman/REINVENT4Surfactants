"""
Monkeypatch implementing Loss Modulation (LM), per Medina's master's thesis
"Uncertainty-aware reinforcement learning for chemical de novo design"
(Section 2.3, Eq. 7-8):

    S_LM(x)  = MPO(s_1, ..., s_K)                    -- score/reward untouched
    L_LM(X)  = (1/N) * sum_j [w_j / mean(w)] * L_j    -- loss reweighted

where w_j is an uncertainty-based weight per sample: here, the arithmetic
mean of the named uncertainty component scores (e.g. pCMC_Uncertainty,
SurfTen_Uncertainty -- "combined as an arithmetic mean to prevent extreme
values from dominating the policy update", per the thesis). These are
already in [0,1] with 1=reliable/certain, matching the thesis's
w_unc_j = 1 - d_j convention exactly, since UncertaintyPenalty
(scoring_functions/comp_uncertainty.py) already returns 1-normalized_std
when minimize=true.

Molecules with lower uncertainty (higher w_j) contribute more to the policy
gradient update; molecules with high uncertainty contribute less. The
reward/score itself (the MPO composite REINVENT reports as `Score`) is
completely unaffected by this -- only how strongly each sample's loss
influences the gradient step.

This is a pure monkeypatch applied at runtime: nothing is installed into
reinvent4-env's site-packages, unlike the scoring_functions/ components. It
only takes effect when imported by workflow/reinvent_with_lm.py with
REINVENT_LM_ENABLED set -- importing this module with the env var unset is a
safe no-op (the patched methods are never installed).

Env vars (read once at import time):
    REINVENT_LM_ENABLED     -- "1"/"true"/"yes" to activate; anything else
                                (including unset) leaves REINVENT's original
                                behavior completely untouched.
    REINVENT_LM_COMPONENTS  -- comma-separated scoring component names to
                                combine into the per-sample weight (default:
                                "pCMC_Uncertainty,SurfTen_Uncertainty").
"""
import os
import logging

import numpy as np
import torch

from reinvent.models.model_factory.sample_batch import SmilesState
from reinvent.runmodes.RL import reward as _reward_module
from reinvent.runmodes.RL import reinvent as _reinvent_learning_module

logger = logging.getLogger(__name__)

LM_ENABLED = os.environ.get("REINVENT_LM_ENABLED", "").lower() in ("1", "true", "yes")
LM_COMPONENTS = [
    c.strip()
    for c in os.environ.get(
        "REINVENT_LM_COMPONENTS", "pCMC_Uncertainty,SurfTen_Uncertainty"
    ).split(",")
    if c.strip()
]


def _weighted_rlreward_call(
    self, orig_smilies, scores, agent_nlls, prior_nlls, mask_idx, inception, agent, weights=None
):
    """Drop-in replacement for RLReward.__call__ that averages the per-sample
    loss with optional weights (Loss Modulation) instead of a plain mean.
    Behaves identically to the original when weights=None."""

    scores_t = torch.from_numpy(scores).to(prior_nlls)
    nan_idx = torch.isnan(scores_t)

    agent_lls = -agent_nlls[~nan_idx]
    prior_lls = -prior_nlls[~nan_idx]
    scores_nonnan = scores_t[~nan_idx]

    loss, augmented_lls = self._strategy(agent_lls, scores_nonnan, prior_lls, self._sigma)

    if inception is not None:
        _orig_smilies, _scores, _prior_lls = inception(
            np.array(orig_smilies)[mask_idx], scores_nonnan[mask_idx], prior_lls[mask_idx]
        )

        lls = agent.likelihood_smiles(_orig_smilies)
        _agent_lls = -lls if isinstance(lls, torch.Tensor) else -lls.likelihood

        inception_loss, _ = self._strategy(
            _agent_lls,
            torch.tensor(_scores).to(_agent_lls),
            torch.tensor(_prior_lls).to(_agent_lls),
            self._sigma,
        )
        loss = torch.cat((loss, inception_loss), 0)

        if weights is not None:
            # Inception samples have no fresh uncertainty estimate of their
            # own for this step; weight them neutrally.
            weights = torch.cat((weights, torch.ones(inception_loss.shape[0]).to(weights)), 0)

    if weights is not None and weights.numel() == loss.numel():
        w_norm = weights / weights.mean()
        loss_scalar = (w_norm * loss).mean()
    else:
        loss_scalar = loss.mean()

    self._optimizer.zero_grad()
    loss_scalar.backward()
    self._optimizer.step()

    return agent_lls, prior_lls, augmented_lls, loss_scalar


def _lm_reinvent_update(self, results, orig_smilies):
    """Drop-in replacement for ReinventLearning.update that additionally
    computes per-sample uncertainty weights (Loss Modulation) from the
    configured scoring components and passes them through to RLReward."""

    agent_nlls = self._state.agent.likelihood_smiles(self.sampled.items2)
    prior_nlls = self.prior.likelihood_smiles(self.sampled.items2)

    weights = None

    if LM_ENABLED and LM_COMPONENTS:
        per_component = []

        for comp in results.completed_components:
            for name, comp_scores in zip(comp.component_names, comp.transformed_scores):
                if name in LM_COMPONENTS:
                    per_component.append(np.asarray(comp_scores, dtype=np.float64))

        if per_component:
            w = np.nanmean(np.stack(per_component, axis=0), axis=0)  # arithmetic mean, per thesis
            w = np.nan_to_num(w, nan=0.0)

            nan_idx = np.isnan(results.total_scores)
            w = w[~nan_idx]
            weights = torch.from_numpy(w).to(agent_nlls)
        else:
            logger.warning(
                f"[LossModulation] none of {LM_COMPONENTS} found among scoring "
                f"components this step; falling back to unweighted loss."
            )

    return self.reward_nlls(
        orig_smilies,
        results.total_scores,
        agent_nlls,
        prior_nlls,
        np.argwhere(self.sampled.states == SmilesState.VALID).flatten(),
        self.inception,
        self._state.agent,
        weights=weights,
    )


if LM_ENABLED:
    _reward_module.RLReward.__call__ = _weighted_rlreward_call
    _reinvent_learning_module.ReinventLearning.update = _lm_reinvent_update
    print(
        f"[reinvent_lm_patch] Loss Modulation ENABLED, combining components: {LM_COMPONENTS}",
        flush=True,
    )
