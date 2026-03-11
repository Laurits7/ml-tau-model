from mltau.tools.evaluation import charge_id as c


def log_charge_id_performance(predictions, truth, output_dir, sample, algorithm):
    evaluator = c.ChargeIdEvaluator(
        predicted=predictions,
        truth=truth,
        output_dir=output_dir,
        sample=sample,
        algorithm=algorithm,
    )
    print(evaluator)
