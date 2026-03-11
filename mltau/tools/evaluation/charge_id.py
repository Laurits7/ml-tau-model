import os
import numpy as np


class ChargeIdEvaluator:
    """Class for evaluating charge ID performance."""

    def __init__(
        self,
        predicted: np.array,
        truth: np.array,
        output_dir: str,
        sample: str,
        algorithm: str,
    ):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        self.sample = sample
        self.algorithm = algorithm
        self.predicted = predicted
        self.truth = truth
