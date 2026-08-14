from numpy import array, concatenate, mean, ndarray
from pyepo.metric import calRegret
from typing import Optional

def regret(
        pred: ndarray,
        true: ndarray,
        true_objs: Optional[ndarray],
        optModel,
        reduction: str = 'mean'
):
    agg_regret = []

    # Calculate individual regret for each time-series.
    for i, (pred_i, true_i) in enumerate(zip(pred, true)):
        pred_cost_vector = concatenate([pred_i + optModel.C, -pred_i + optModel.C])
        true_cost_vector = concatenate([true_i + optModel.C, -true_i + optModel.C])
        if true_objs is not None:
            true_obj = true_objs[i]
        else:
            optModel.setObj(true_cost_vector)
            _, true_obj = optModel.solve()

        r = calRegret(optModel, pred_cost_vector, true_cost_vector, true_obj)
        agg_regret.append(r)
    agg_regret = array(agg_regret)

    if reduction == 'none':
        return agg_regret
    elif reduction == 'mean':
        return mean(agg_regret)