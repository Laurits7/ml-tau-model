import numpy as np
import matplotlib.pyplot as plt

from mltau.tools.evaluation import decay_mode as dm
from mltau.tools.logging.general import log_metrics_dict


def log_all_decay_mode_metrics(
    targets: np.array,
    predictions: np.array,
    tb_logger,
    # output_dir: str,
    current_epoch: int,
):
    predictions = np.argmax(predictions["decay_mode"], axis=-1)
    targets = np.argmax(targets["decay_mode"], axis=-1)
    evaluator = dm.DecayModeEvaluator(
        predicted=predictions,
        truth=targets,
        output_dir="",
        sample="all",
        algorithm="all",
    )

    cm_fig, _ = evaluator.plot_confusion_matrix()
    tb_logger.add_figure("decay_mode/confusion_matrix", cm_fig, current_epoch)
    plt.close(cm_fig)

    log_metrics_dict(tb_logger, evaluator.general_metrics, "decay_mode", current_epoch)
