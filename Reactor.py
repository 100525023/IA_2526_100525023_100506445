import numpy as np


class Reactor:
    """
    Physical model of a nuclear reactor.

    Encapsulates the reactor's core parameters and provides methods to compute
    its maximum thermal power, the k constant governing the exponential decay of
    power with control rod insertion, and the bidirectional mapping between control
    rod position and normalized power output.
    """

    def __init__(self,
                 model: str,
                 effective_section: np.float64,
                 neutron_flux: np.float64,
                 core_volume: np.float64,
                 fision_energy: np.float64,
                 probabilities: dict):
        """
        Initializes the reactor with its physical characteristics and stochastic dynamics.

        Parameters
        ----------
        model : str
            Human-readable name or identifier of the reactor model.
        effective_section : float
            Macroscopic fission cross-section of the fuel (cm^-1).
        neutron_flux : float
            Neutron flux in the reactor core (neutrons / cm^2 · s).
        core_volume : float
            Volume of the reactor core (cm^3).
        fision_energy : float
            Energy released per fission event (J).
        probabilities : dict
            Stochastic transition probabilities for each control action
            (decrease, maintain, increase), as read from the reactor's JSON file.
        """
        self.model             = model
        self.effective_section = effective_section
        self.neutron_flux      = neutron_flux
        self.core_volume       = core_volume
        self.fision_energy     = fision_energy
        self.probabilities     = probabilities
        self.max_power         = self.compute_max_power()
        self.k                 = self.compute_k()

    def __str__(self) -> str:
        """Returns a human-readable summary of the reactor's physical parameters."""
        lines = [
            f"Model:             {self.model}",
            f"Effective section: {self.effective_section} cm^-1",
            f"Neutron flux:      {self.neutron_flux} neutrons / (cm^2 · s)",
            f"Core volume:       {self.core_volume} cm^3",
            f"Fission energy:    {self.fision_energy} J",
            f"Probabilities:     {self.probabilities}",
        ]
        return "\n".join(lines)

    def compute_max_power(self) -> np.float64:
        """
        Computes the theoretical maximum thermal power of the reactor.

        Uses the standard neutronics formula:
            P_max = Sigma_f · phi · V · E_f

        Returns
        -------
        float
            Maximum thermal power in watts (W).
        """
        return self.effective_section * self.neutron_flux * self.core_volume * self.fision_energy

    def compute_k(self) -> np.float64:
        """
        Computes the decay constant k for the control rod insertion model.

        The constant is derived so that full insertion (B = 1) produces a power
        of approximately zero (10^-6 W), while no insertion (B = 0) yields P_max:
            k = -ln(10^-6 / P_max)

        Returns
        -------
        float
            Decay constant k (dimensionless).
        """
        return -np.log(1e-6 / self.max_power)

    def compute_power(self, control_bars_insertion: np.float64) -> np.float64:
        """
        Returns the normalized power fraction [0, 1] for a given control rod insertion.

        The model follows an exponential decay:
            P(B) = P_max · e^(-k·B) / P_max = e^(-k·B)

        Where B = 0 corresponds to no insertion (maximum power) and B = 1 to
        full insertion (near-zero power).

        Parameters
        ----------
        control_bars_insertion : float
            Fraction of control rods inserted, in the range [0, 1].

        Returns
        -------
        float
            Normalized power output, in the range [~0, 1].
        """
        B = np.clip(control_bars_insertion, 0.0, 1.0)
        return np.exp(-self.k * B)

    def compute_control_bars_insertion(self, power: np.float64) -> np.float64:
        """
        Returns the control rod insertion fraction required to achieve a given power level.

        This is the analytical inverse of compute_power:
            B = -ln(power) / k

        Parameters
        ----------
        power : float
            Desired normalized power level, in the range (0, 1].

        Returns
        -------
        float
            Required control rod insertion fraction, in the range [0, 1].
        """
        p = np.clip(power, 1e-10, 1.0)
        return -np.log(p) / self.k
