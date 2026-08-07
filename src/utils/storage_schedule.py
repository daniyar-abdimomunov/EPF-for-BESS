import matplotlib.pyplot as plt
from numpy import arange, array, concatenate, ndarray
from src.utils import BESSSchedulingOptModel


def solve_storage_schedule(
        prices: ndarray,
        optModel: BESSSchedulingOptModel
):
    """
    Wrapper function to cleanly process inputs and outputs for BESSSchedulingModel.
    """
    # Process prices to fit to cost_vector input for optimization model.
    cost_vector = concatenate([
        prices + optModel.C,
        -prices + optModel.C
    ])

    # Solve optimization model.
    optModel.setObj(cost_vector)
    best_x, obj_value = optModel.solve()

    # Process optimization model outputs.
    charge_sol, discharge_sol = best_x[:optModel.num_timesteps], best_x[optModel.num_timesteps:]
    soc = array([optModel._model.soc[t].value for t in optModel._model.t])
    profit = -obj_value

    return profit, charge_sol, discharge_sol, soc


def plot_storage_schedule(
        charge_sol: ndarray,
        discharge_sol: ndarray,
        prices: ndarray,
        soc: ndarray,
):
    # Initialize figure.
    fig, ax = plt.subplots(1,1, sharex='col', sharey='all', figsize=(7,6))

    # Define masks for prices based on decision to charge or discharge.
    charge_prices = prices.copy()
    discharge_prices = prices.copy()
    charge_prices[charge_sol <= 0] = None
    discharge_prices[discharge_sol <= 0] = None

    # Plot prices with masks.
    ax.plot(prices, label='prices', linestyle='--')
    ax.plot(charge_prices, label='charge', color='green')
    ax.plot(discharge_prices, label='discharge', color='red')
    ax.legend(loc='upper left')
    ax.grid(False)

    # Plot State-of-Charge (SoC).
    ax2 = ax.twinx()
    ax2.bar(arange(len(prices)), soc, width=1, linewidth=0, alpha=0.2, label='soc')
    ax2.legend(loc='upper right')
    ax2.grid(False)

    # Add labels and formatting.
    fig.supxlabel('Time')
    fig.supylabel('Price (EUR/MWh)')
    fig.autofmt_xdate()
    plt.suptitle('Optimized Energy Storage Dispatch for Different Storage Durations\n', size=20)
    plt.legend()
    plt.tight_layout()
    plt.show()
    return