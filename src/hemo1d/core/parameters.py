from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class BloodParameters:
    """
    Physical blood parameters.

    Units:
        rho: g / cm^3
        mu:  g / (cm s)

    The viscosity mu is the dynamic viscosity.
    """

    rho: float = 1.06
    mu: float = 0.035


@dataclass(frozen=True)
class VesselParameters:
    """
    Parameters of one straight compliant vessel.

    Units:
        length: cm
        area0:  cm^2
        beta:   pressure-area elastic coefficient

    The tube law used by the 1D model is:

        psi(A) = beta * (sqrt(A) - sqrt(A0)) / A0
    """

    length: float
    area0: float
    beta: float


@dataclass(frozen=True)
class ModelParameters:
    """
    Full parameter set for one 1D vessel model.

    gamma_profile:
        Exponent of the assumed velocity profile.

        gamma = 2 corresponds to the parabolic/Poiseuille profile.
        With the profile convention used in the thesis:

            alpha = (gamma + 2) / (gamma + 1)

        so gamma = 2 gives alpha = 4/3.

    p0:
        Reference internal pressure.

    p_ext:
        External pressure.

    gamma_pressure_loss:
        Coefficient of the pressure loss on junctions.
    """

    blood: BloodParameters
    vessel: VesselParameters

    gamma_profile: float = 2.0
    p0: float = 0.0
    p_ext: float = 0.0
    gamma_pressure_loss: float = 0.0

    @property
    def rho(self) -> float:
        return self.blood.rho

    @property
    def mu(self) -> float:
        return self.blood.mu

    @property
    def length(self) -> float:
        return self.vessel.length

    @property
    def area0(self) -> float:
        return self.vessel.area0

    @property
    def beta(self) -> float:
        return self.vessel.beta

    @property
    def gamma(self) -> float:
        return self.gamma_profile

    @property
    def gamma_pressure(self) -> float:
        return self.gamma_pressure_loss

    @property
    def alpha(self) -> float:
        gamma = self.gamma_profile
        return (gamma + 2.0) / (gamma + 1.0)