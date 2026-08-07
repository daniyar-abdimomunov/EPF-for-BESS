from numpy import array, concatenate, mean, ndarray
from pyepo.metric import calRegret

def regret(
        pred: ndarray,
        true: ndarray,
        optModel
):
    agg_regret = []

    # Calculate individual regret for each time-series.
    for pred_i, true_i in zip(pred, true):
        pred_cost_vector = concatenate([pred_i + optModel.C, -pred_i + optModel.C])
        true_cost_vector = concatenate([true_i + optModel.C, -true_i + optModel.C])
        optModel.setObj(true_cost_vector)
        _, true_obj = optModel.solve()

        r = calRegret(optModel, pred_cost_vector, true_cost_vector, true_obj)
        agg_regret.append(r)
    agg_regret = array(agg_regret)

    # Reduce individual regrets to average.
    avg_regret = mean(agg_regret)
    return avg_regret