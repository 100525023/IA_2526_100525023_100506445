import numpy as np
import argparse
import json
import os
from Reactor import Reactor
from ControlModule import ControlModule
from DemandGenerator import generate_demand
from Metrics import *
from Plotter import *


def get_args() -> tuple[Reactor, np.float64, int, str]:
    """
    Parses command-line arguments and constructs the Reactor object from the provided JSON file.

    Expected arguments:
        --input-reactor (-i): path to the reactor configuration JSON file.
        --gamma (-g):         discount factor for the MDP (float).
        --random-seed (-r):   seed for NumPy's random number generator (int).

    Returns
    -------
    tuple
        A 4-tuple containing the Reactor instance, the discount factor gamma,
        the random seed, and the path to the reactor JSON file.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-reactor", "-i", type=str,
                        help="Path to the reactor's JSON configuration file.")
    parser.add_argument("--gamma", "-g", type=float,
                        help="Discount factor used in the MDP (typically in (0, 1]).")
    parser.add_argument("--random-seed", "-r", type=int,
                        help="Seed for the pseudo-random number generator.")

    args = parser.parse_args()

    print(f"Loading reactor from:    {args.input_reactor}")
    print(f"Discount factor (gamma): {args.gamma}")
    print(f"Random seed:             {args.random_seed}")

    with open(args.input_reactor, 'r', encoding='utf-8') as file:
        json_data = json.load(fp=file)
        reactor = Reactor(
            model=json_data['model'],
            effective_section=float(json_data['effective_section']),
            neutron_flux=float(json_data['neutron_flux']),
            core_volume=float(json_data['core_volume']),
            fision_energy=float(json_data['fision_energy']),
            probabilities=dict(json_data['probabilities'])
        )

    print(reactor)

    return reactor, args.gamma, args.random_seed, args.input_reactor


def main() -> None:
    """
    Entry point of the nuclear reactor control simulation.

    Orchestrates the full pipeline: loading the reactor, generating a random demand
    curve, running the MDP-based control loop, computing regression metrics, and
    saving all plots to disk.
    """
    reactor, gamma, random_seed, reactor_path = get_args()

    np.random.seed(random_seed)

    # Redirect all plots to a subdirectory named after the reactor model
    import Plotter
    Plotter._PLOTS_DIR = os.path.join("plots", reactor.model)
    os.makedirs(Plotter._PLOTS_DIR, exist_ok=True)
    Plotter._plot_counter[0] = 0
    print(f"Plots will be saved to: {Plotter._PLOTS_DIR}/")

    # Extract the stochastic transition probabilities for each action
    probs = np.array([
        reactor.probabilities['decrease'],
        reactor.probabilities['maintain'],
        reactor.probabilities['increase']
    ], dtype=np.float64)

    # Plot the reactor's stochastic profile as a radar chart
    plot_reactor_as_radar(probs=probs)

    # Generate a synthetic normalized demand curve
    demand = generate_demand(n_samples=512)

    # MDP configuration
    n_states  = 100
    n_actions = 3

    # Run the MDP-based control loop to generate the reactor's response
    response = ControlModule.control_loop(
        demand=demand,
        probs=probs,
        n_states=n_states,
        n_actions=n_actions,
        gamma=gamma
    )

    # Generate and save all plots
    plot_demand(demand=demand)
    plot_demand_response(demand=demand, response=response)
    plot_control_bars_usage(reactor=reactor, response=response)
    plot_correlation(demand=demand, response=response)

    # Compute and display regression quality metrics
    _MAE  = MAE(y_true=demand, y_pred=response)
    _MSE  = MSE(y_true=demand, y_pred=response)
    _R2   = R2(y_true=demand, y_pred=response)
    _Corr = Corr(y_true=demand, y_pred=response)

    print(f"MAE  = {_MAE:.6f}")
    print(f"MSE  = {_MSE:.6f}")
    print(f"R²   = {_R2:.6f}")
    print(f"Corr = {_Corr:.6f}")

    plot_mae_and_mse(MAE=_MAE, MSE=_MSE)
    plot_r2_and_pearson(R2=_R2, Pearson=_Corr)

    print(f"\nSimulation complete. Plots saved to: {Plotter._PLOTS_DIR}/")


if __name__ == '__main__':
    main()
