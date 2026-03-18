"""
Inference postprocessor: translates raw ParTau model predictions into physical quantities
and packages them as an awkward array.

Model output dict:
  predictions["is_tau"]      shape (N,)    sigmoid score ∈ [0, 1]
  predictions["charge"]      shape (N,)    sigmoid score ∈ [0, 1]  (1 = positive charge)
  predictions["decay_mode"]  shape (N, 6)  softmax probabilities over DM classes [0,1,2,10,11,15]
  predictions["kinematics"]  shape (N, 4)  raw regression:
      [:,0] = log(pt_gen / pt_reco)         → pred_pt = exp(pred[:,0]) * reco.pt
      [:,1] = delta_theta (gen - reco)      → pred_theta = pred[:,1] + reco.theta
      [:,2] = delta_phi   (gen - reco)      → pred_phi   = pred[:,2] + reco.phi
      [:,3] = log(m_gen / m_reco)           → pred_mass  = exp(pred[:,3]) * reco.mass

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
    """Decode raw ParTau predictions into physical quantities.

    Args:
        predictions: dict returned by ParTau.forward() / predict_step.
            Values may be torch.Tensor or numpy-convertible arrays.
        reco_jet_p4s: awkward array with reco-jet 4-momentum fields
            (must be compatible with reinitialize_p4).

    Returns:
        ak.Array with fields:
            tagging_score, charge_score, decay_mode, decay_mode_probs,
            pred_p4 (vector p4 with pt, eta, phi, energy)
    """
    # --- helpers to convert any tensor/array to numpy ---
    def to_np(x):
        if isinstance(x, torch.Tensor):
            return x.detach().cpu().numpy()
        return np.asarray(x)

    # --- reco 4-vector ---
    reco = reinitialize_p4(reco_jet_p4s)
    reco_pt    = np.asarray(reco.pt)
    reco_theta = np.asarray(reco.theta)   # vector derives theta from eta
    reco_phi   = np.asarray(reco.phi)
    reco_mass  = np.asarray(reco.mass)

    # --- decode kinematics ---
    kin = to_np(predictions["kinematics"])        # (N, 4)

    pred_pt    = np.exp(kin[:, 0]) * reco_pt
    pred_theta = kin[:, 1] + reco_theta
    pred_phi   = kin[:, 2] + reco_phi
    pred_mass  = np.exp(kin[:, 3]) * reco_mass

    # theta → eta:  eta = -log(tan(theta/2))
    theta_clipped = np.clip(pred_theta, 1e-6, np.pi - 1e-6)
    pred_eta = -np.log(np.tan(theta_clipped / 2.0))

    # energy:  E = sqrt((pt/sin(theta))^2 + m^2)
    sin_theta = np.clip(np.sin(theta_clipped), 1e-6, None)
    pred_p     = pred_pt / sin_theta
    pred_energy = np.sqrt(pred_p**2 + pred_mass**2)

    # --- decode decay mode ---
    dm_probs = to_np(predictions["decay_mode"])   # (N, 6) – already softmax
    dm_idx   = np.argmax(dm_probs, axis=-1)        # (N,) indices 0-5
    dm_class = one_hot_decoding(dm_idx)             # (N,) e.g. {0,1,2,10,11,15}

    # --- pass-through scores ---
    tagging_score = to_np(predictions["is_tau"])   # (N,)
    charge_score  = to_np(predictions["charge"])   # (N,)

    # --- build predicted p4 as a vector awkward array (mirrors reinitialize_p4) ---
    pred_p4 = vector.awk(
        ak.zip(
            {
                "pt":     pred_pt,
                "eta":    pred_eta,
                "phi":    pred_phi,
                "energy": pred_energy,
            }
        )
    )

    return ak.Array(
        {
            "tagging_score":    tagging_score,
            "charge_score":     charge_score,
            "decay_mode":       dm_class,
            "decay_mode_probs": dm_probs,
            "pred_p4":          pred_p4,
        }
    )
