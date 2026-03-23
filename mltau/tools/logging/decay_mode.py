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
    predictions_proba = np.asarray(predictions["decay_mode"])
    targets_class = np.argmax(np.asarray(targets["decay_mode"]), axis=-1)
    predictions_class = np.argmax(predictions_proba, axis=-1)

    evaluator = dm.DecayModeEvaluator(
        predicted=predictions_class,
        truth=targets_class,
        output_dir="",
        sample="all",
        algorithm="all",
    )

    cm_fig, _ = evaluator.plot_confusion_matrix()
    tb_logger.add_figure("decay_mode/confusion_matrix", cm_fig, current_epoch)
    plt.close(cm_fig)

    log_metrics_dict(tb_logger, evaluator.general_metrics, "decay_mode", current_epoch)

    roc_plot = dm.DecayModeROCPlot(
        predictions_proba=predictions_proba,
        targets=targets_class,
        categories=evaluator.categories,
    )
    tb_logger.add_figure("decay_mode/ROC", roc_plot.fig, current_epoch)
    plt.close(roc_plot.fig)
