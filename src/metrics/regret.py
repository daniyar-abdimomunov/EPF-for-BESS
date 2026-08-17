from numpy import array, concatenate, mean, ndarray
from pyepo.metric import calRegret

def regret(
        pred: ndarray,
        true: ndarray,
        optModel,
        reduction: str = 'mean'
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

    if reduction == 'none':
        return agg_regret
    elif reduction == 'mean':
        return mean(agg_regret)