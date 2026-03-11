def log_metrics_dict(tb_logger, metrics_dict, prefix, step):
    """Log dictionary of metrics with a common prefix"""
    for metric_name, metric_value in metrics_dict.items():
        tb_logger.add_scalar(f"{prefix}/{metric_name}", metric_value, step)
