import highspy
from pyepo.EPO import MINIMIZE
from pyepo.model.omo import optOmoModel
import pyomo.environ as ampo


class BESSSchedulingOptModel(optOmoModel):
    def __init__(
            self,
            solver: str = 'highs', # one of allowed solvers for optOmoModel
            Pow: float = 30.0, # maximum power (MW)
            E: float = 100.0, # storage capacity (MWh)
            eta_ch: float = 0.92, # charging efficiency
            eta_dis: float = 0.92, # discharging efficiency
            C: float = 2.5, # operating costs (EUR/MWh)
            H: int = 24, # number of timesteps
            **kwargs,
    ):
        """
        Custom PyEPO optModel for BESS scheduling to be used in SPO+ loss function in model training.
        """
        self.Pow = Pow
        self.E = E
        self.eta_ch = eta_ch
        self.eta_dis = eta_dis
        self.C = C
        self.H = H

        super().__init__(solver=solver)

    def _getModel(self):
        """
        Initializes the structural constraints and variables.
        """
        # set model objective sense
        self.modelSense = MINIMIZE

        # Initialize model and indices for decision variables.
        m = ampo.ConcreteModel()
        num_actions = 2 * self.H
        m.actions = ampo.RangeSet(0, num_actions - 1)
        m.h = ampo.RangeSet(0, self.H - 1)

        # Define decision variable: both charge and discharge decisions in a single vector.
        # NOTE: The first `H` indices are for charging, the next are for discharging.
        m.x = ampo.Var(m.actions, bounds=(0, self.Pow))

        # Binary variable to indicate charging state (1 if charging, 0 if discharging/idle)
        m.b_ch = ampo.Var(m.h, domain=ampo.Binary)

        # Constraint: Can only charge if b_ch is 1
        def charge_limit_rule(model, h):
            return model.x[h] <= self.Pow * model.b_ch[h]
        m.charge_limit_constraint = ampo.Constraint(m.h, rule=charge_limit_rule)

        # Constraint: Can only discharge if b_ch is 0
        def discharge_limit_rule(model, t):
            return model.x[t + self.H] <= self.Pow * (1 - model.b_ch[t])
        m.discharge_limit_constraint = ampo.Constraint(m.h, rule=discharge_limit_rule)

        # Define State-of-Charge (SoC) variable
        m.soc = ampo.Var(m.h, bounds=(0, self.E))

        # Define State-of-Charge (SoC) Balance Constraints
        def soc_balance_rule(model, h):
            if h == 0:
                return ampo.Constraint.Skip
            # Map indices from m.x to charge/discharge for the previous time step
            charge_h_minus_1 = model.x[h-1]
            discharge_h_minus_1 = model.x[h - 1 + self.H]
            return model.soc[h] == model.soc[h-1] + (charge_h_minus_1 * self.eta_ch) - (discharge_h_minus_1 / self.eta_dis)
        m.soc_balance = ampo.Constraint(m.h, rule=soc_balance_rule)

        # Define Cyclic SoC Constraint - period must start and end with same SoC
        final_h = self.H - 1
        charge_final_h = m.x[final_h]
        discharge_final_h = m.x[final_h + self.H]
        m.cyclic_soc = ampo.Constraint(
            expr=m.soc[0] == m.soc[final_h] + (charge_final_h * self.eta_ch) - (discharge_final_h / self.eta_dis))

        return m, m.x