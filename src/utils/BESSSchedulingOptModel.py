import highspy
from pyepo.EPO import MINIMIZE
from pyepo.model.omo import optOmoModel
import pyomo.environ as ampo


class BESSSchedulingOptModel(optOmoModel):
    def __init__(
            self,
            solver: str = 'highs', # one of allowed solvers for optOmoModel
            P: float = 30.0, # maximum power (MW)
            E: float = 100.0, # storage capacity (MWh)
            eta_ch: float = 0.92, # charging efficiency
            eta_dis: float = 0.92, # discharging efficiency
            C: float = 2.5, # operating costs (EUR/MWh)
            num_timesteps=24, # number of timsteps
    ):
        """
        Custom PyEPO optModel for BESS scheduling to be used in SPO+ loss function in model training.
        """
        self.P = P
        self.E = E
        self.eta_ch = eta_ch
        self.eta_dis = eta_dis
        self.C = C
        self.num_timesteps = num_timesteps

        super().__init__(solver=solver)

    def _getModel(self):
        """
        Initializes the structural constraints and variables.
        """
        # set model objective sense
        self.modelSense = MINIMIZE

        # Initialize model and indices for decision variables.
        m = ampo.ConcreteModel()
        num_actions = 2 * self.num_timesteps
        m.actions = ampo.RangeSet(0, num_actions - 1)
        m.t = ampo.RangeSet(0, self.num_timesteps - 1)

        # Define decision variable: both charge and discharge decisions in a single vector.
        # NOTE: The first `num_timesteps` indices are for charging, the next are for discharging.
        m.x = ampo.Var(m.actions, bounds=(0, self.P))

        # Define State-of-Charge (SoC) variable
        m.soc = ampo.Var(m.t, bounds=(0, self.E))

        # Define State-of-Charge (SoC) Constraints
        def soc_balance_rule(model, t):
            if t == 0:
                return ampo.Constraint.Skip
            # Map indices from m.x to charge/discharge for the previous time step
            charge_t_minus_1 = model.x[t-1]
            discharge_t_minus_1 = model.x[t-1 + self.num_timesteps]
            return model.soc[t] == model.soc[t-1] + (charge_t_minus_1 * self.eta_ch) - (discharge_t_minus_1 / self.eta_dis)
        m.soc_balance = ampo.Constraint(m.t, rule=soc_balance_rule)

        # Define Cyclic SoC Constraint - period must start and end with same SoC
        final_t = self.num_timesteps - 1
        charge_final_t = m.x[final_t]
        discharge_final_t = m.x[final_t + self.num_timesteps]
        m.cyclic_soc = ampo.Constraint(
            expr=m.soc[0] == m.soc[final_t] + (charge_final_t * self.eta_ch) - (discharge_final_t / self.eta_dis))

        return m, m.x