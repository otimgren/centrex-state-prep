"""
Define some intensity profiles for microwaves
"""
from dataclasses import dataclass

import numpy as np
from scipy import constants

from .microwaves import Intensity


@dataclass
class GaussianBeam(Intensity):
    """
    Class for Gaussian microwave intensity profiles.
    """

    power: float
    sigma: float
    R0: np.ndarray
    k: np.ndarray
    freq: float

    def I_R(self, R: np.ndarray, power = None) -> float:
        """
        Calculates the intensity at point R.
        """
        if not power:
            power = self.power

        return power * self.profile_R(R)

    def profile_R(self, R) -> float:
        """
        Calculates the relative instensity at position R.

        Normalized so that integral over profile equals 1. 
        """
        k = self.k
        R0 = self.R0
        sigma = self.sigma
        # Calculate position along beam
        z = np.dot(k, R) - np.dot(k, R0)

        # Calculate radial offset from center of beam
        r = np.sqrt(np.sum(((R - np.dot(k, R) * k) - (R0 - np.dot(k, R0) * k)) ** 2))

        # Calculate Rayleigh range
        wavelength = constants.c / self.freq
        z_R = 4 * np.pi * sigma ** 2 / wavelength

        return (
            (1 / (2 * sigma))
            * (1 / np.sqrt(1 + (z / z_R) ** 2))
            * np.exp(-(r ** 2) / (2 * sigma ** 2))
        )

