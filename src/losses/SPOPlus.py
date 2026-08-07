import numpy as np
from pyepo.func import SPOPlus as _SPOPlus
from torch import cat, from_numpy, Tensor
from typing import Optional

from src import BESSSchedulingOptModel


class SPOPlus(_SPOPlus):
    def __init__(
            self,
            optmodel: BESSSchedulingOptModel,
            *args,
            **kwargs
    ):
        self.optmodel = optmodel
        super().__init__(optmodel=optmodel, *args, **kwargs)

    def forward(
            self,
            pred_cost: Tensor,
            true_cost: Tensor,
            true_sol: Optional[Tensor],
            true_obj: Optional[Tensor],
    ):
        """
        Wrapper for parent forward function to handle re-shaping the input cost vectors
        and calculating the optimal true solution if not provided.
        """
        # Prepare pred and true cost vectors.
        if pred_cost.shape[1] == true_cost.shape[1] == self.optmodel.num_cost / 2:
            C = self.optmodel.C

            pred_cost = cat([pred_cost + C, -pred_cost + C], dim=1)
            true_cost = cat([true_cost + C, -true_cost + C], dim=1)

        # Compute optimal solution and objective value if not provided.
        if true_sol is None or true_obj is None:
            true_sol, true_obj = list(), list()
            for i in range(true_cost.shape[0]):
                # Use the fully-formed true_cost vector to solve for the optimal solution
                current_true_cost_np = true_cost[i].cpu().detach().numpy()
                self.optmodel.setObj(current_true_cost_np)
                best_x, objective_val = self.optmodel.solve()
                true_sol.append(best_x)
                true_obj.append(objective_val)

            true_sol, true_obj = [from_numpy(np.array(array)).float().to(true_cost.device) for array in [true_sol, true_obj]]

        return super().forward(pred_cost, true_cost, true_sol, true_obj)
