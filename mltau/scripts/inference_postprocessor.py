"""
Inference postprocessor: translates raw ParTau model predictions into physical quantities
and packages them as an awkward array.

Model output dict:
  predictions["is_tau"]      shape (N,)    sigmoid score ∈ [0, 1]
  predictions["charge"]      shape (N,)    sigmoid score ∈ [0, 1]  (1 = positive charge)
  predictions["decay_mode"]  shape (N, 6)  softmax probabilities over DM classes [0,1,2,10,11,15]
  predictions["kinematics"]  shape (N, 4)  raw regression:
      [:,0] = log(pt_gen / pt_reco)         → pred_pt = exp(pred[:,0]) * reco.pt
      [:,1] = delta_eta (gen - reco)        → pred_eta = pred[:,1] + reco.eta
      [:,2] = sin(delta_phi(gen - reco))
      [:,3] = cos(delta_phi(gen - reco))    → pred_phi via atan2(sin, cos) + reco.phi

Visible mass is not learned in this configuration and is taken directly from the reco jet.

Output awkward array fields per candidate:
  tagging_score        float   sigmoid score (is_tau)
  charge_score         float   sigmoid score (positive charge)
  decay_mode           int     decoded decay mode class ∈ {0, 1, 2, 10, 11, 15}
  decay_mode_probs     float[6]  softmax probabilities for each DM class
  pred_pt              float   [GeV]
  pred_eta             float
  pred_phi             float   [rad]
  pred_energy          float   [GeV]
  pred_mass            float   [GeV]
"""

import vector
import numpy as np
import awkward as ak
import torch

from mltau.tools.general import reinitialize_p4, one_hot_decoding


def postprocess_predictions(predictions: dict, reco_jet_p4s: ak.Array) -> ak.Array:
    """Decode raw ParTau predictions into physical quantities."""

    def to_np(x):
        if isinstance(x, torch.Tensor):
            return x.detach().cpu().numpy()
        return np.asarray(x)

    reco = reinitialize_p4(reco_jet_p4s)
    reco_pt = np.asarray(reco.pt)
    # Original angular decoding inputs kept for reference:
    # reco_theta = np.asarray(reco.theta)
    reco_eta = np.asarray(reco.eta)
    reco_phi = np.asarray(reco.phi)
    reco_mass = np.asarray(reco.mass)

    kin = to_np(predictions["kinematics"])

    # Original decode kept for reference:
    # pred_pt = np.exp(kin[:, 0]) * reco_pt
    # pred_theta = kin[:, 1] + reco_theta
    # pred_phi = kin[:, 2] + reco_phi
    # pred_mass = np.exp(kin[:, 3]) * reco_mass
    # theta_clipped = np.clip(pred_theta, 1e-6, np.pi - 1e-6)
    # pred_eta = -np.log(np.tan(theta_clipped / 2.0))
    # sin_theta = np.clip(np.sin(theta_clipped), 1e-6, None)
    # pred_p = pred_pt / sin_theta
    # pred_energy = np.sqrt(pred_p**2 + pred_mass**2)

    pred_pt = np.exp(kin[:, 0]) * reco_pt
    pred_eta = kin[:, 1] + reco_eta
    pred_delta_phi = np.arctan2(kin[:, 2], kin[:, 3])
    pred_phi = pred_delta_phi + reco_phi
    pred_phi = np.arctan2(np.sin(pred_phi), np.cos(pred_phi))
    pred_mass = reco_mass
    pred_p = pred_pt * np.cosh(pred_eta)
    pred_energy = np.sqrt(pred_p**2 + pred_mass**2)

    dm_probs = to_np(predictions["decay_mode"])
    dm_idx = np.argmax(dm_probs, axis=-1)
    dm_class = one_hot_decoding(dm_idx)

    tagging_score = to_np(predictions["is_tau"])
    charge_score = to_np(predictions["charge"])

    pred_p4 = vector.awk(
        ak.zip(
            {
                "pt": pred_pt,
                "eta": pred_eta,
                "phi": pred_phi,
                "mass": pred_mass,
            }
        )
    )

    return ak.Array(
        {
            "tagging_score": tagging_score,
            "charge_score": charge_score,
            "decay_mode": dm_class,
            "decay_mode_probs": dm_probs,
            "pred_p4": pred_p4,
        }
    )
