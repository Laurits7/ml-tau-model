import vector
import numpy as np
import awkward as ak


def reinitialize_p4(p4_obj: ak.Array):
    """Reinitialized the 4-momentum for particle in order to access its properties.

    Args:
        p4_obj : ak.Array
            The particle represented by its 4-momenta

    Returns:
        p4 : ak.Array
            Particle with initialized 4-momenta.
    """
    # Initialize from all the p4 fields
    name_map = {
        "x": "px",
        "y": "py",
        "z": "pz",
        "tau": "mass",
        "t": "energy",
        "rho": "pt",
    }
    p4 = vector.awk(
        ak.zip({name_map.get(field, field): p4_obj[field] for field in p4_obj.fields})
    )
    # Now make it so that the 4-vector is always saved in a similar fashion:
    p4 = vector.awk(
        ak.zip(
            {
                "pt": p4.pt,
                "eta": p4.eta,
                "phi": p4.phi,
                "energy": p4.t,
            }
        )
    )
    return p4


def get_reduced_decaymodes(decaymodes: np.array):
    """Maps the full set of decay modes into a smaller subset, setting the rarer decaymodes under "Other" (# 15)"""
    target_mapping = {
        -1: 15,  # As we are running DM classification only on signal sample, then HPS_dm of -1 = 15 (Rare)
        0: 0,
        1: 1,
        2: 2,
        3: 2,
        4: 2,
        5: 10,
        6: 11,
        7: 11,
        8: 11,
        9: 11,
        10: 10,
        11: 11,
        12: 11,
        13: 11,
        14: 11,
        15: 15,
        16: 16,
    }
    return np.vectorize(target_mapping.get)(decaymodes)


def prepare_one_hot_encoding(values, classes=[0, 1, 2, 10, 11, 15]):
    mapping = {class_: i for i, class_ in enumerate(classes)}
    return np.vectorize(mapping.get)(values)


def one_hot_decoding(values, classes=[0, 1, 2, 10, 11, 15]):
    mapping = {i: class_ for i, class_ in enumerate(classes)}
    return np.vectorize(mapping.get)(values)
