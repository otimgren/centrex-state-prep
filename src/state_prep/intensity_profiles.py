"""
Define some intensity profiles for microwaves
"""
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
from scipy import constants
from scipy.integrate import quad
from scipy.special import jv


class Intensity:
    """
    Class for representing spatial dependence of microwave field intensity.
    """

    def E_R(self, R: np.ndarray, power: float = None) -> np.ndarray:
        """
        Convert intensity (W/m^2) to electric field in V/cm and return electric
        field magnitude.
        """
        return (
            np.sqrt(2 * self.I_R(R, power) / (constants.c * constants.epsilon_0)) / 100
        )


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

    def I_R(self, R: np.ndarray, power=None) -> float:
        """
        Calculates the intensity at point R.
        """
        if not power:
            power = self.power

        return power * self.profile_R(R)

    def profile_R(self, R) -> float:
        """
        Calculates the relative intensity at position R.

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
            (1 / (2 * np.pi * sigma ** 2))
            * (1 / (1 + (z / z_R) ** 2))
            * np.exp(-(r ** 2) / (2 * (sigma ** 2 * (1 + (z / z_R) ** 2))))
        )

@dataclass
class BesselGaussianBeam(Intensity):
    """
    Class for Bessel-Gaussian microwave intensity profiles.
    """

    power: float
    w0: float
    alpha: float
    R0: np.ndarray
    k: np.ndarray
    freq: float

    def __post_init__(self):
        self.wavelength = constants.c / self.freq
        self.z_R = np.pi * self.w0 ** 2 / self.wavelength
        self.integral = self.calculate_profile_integral()

    def I_R(self, R: np.ndarray, power=None) -> float:
        """
        Calculates the intensity at point R.
        """
        if not power:
            power = self.power

        return power * self.profile_R(R) / self.integral

    def E_R(self, R: np.ndarray, power: float = None) -> np.ndarray:
        """
        Convert intensity (W/m^2) to electric field in V/cm and return electric
        field magnitude.
        """
        k = self.k
        R0 = self.R0
        w0 = self.w0
        if not power:
            power = self.power
        # Calculate position along beam
        z = np.dot(k, R) - np.dot(k, R0)

        # Calculate radial offset from center of beam
        r = np.sqrt(np.sum(((R - np.dot(k, R) * k) - (R0 - np.dot(k, R0) * k)) ** 2))

        # Calculate Rayleigh range
        z_R = self.z_R

        # Calculate inverse of radius of curvature of wavefronts
        invroc = z/(z**2 + z_R**2)

        return (
            np.sqrt(2 * power/self.integral / (constants.c * constants.epsilon_0)) / 100
            *jv(0, self.alpha*r)
            * np.sqrt((1 / (1 + (z / z_R) ** 2)))
            * np.exp(-(r ** 2) / ((w0 ** 2 * (1 + (z / z_R) ** 2))))
            #* np.exp(-1j*2*np.pi/self.wavelength * invroc * r**2/2)
        )

    def profile_R(self, R) -> float:
        """
        Calculates the relative intensity at position R.

        Normalized so that intensity = 1 at r = 0, z = 0.  
        """
        k = self.k
        R0 = self.R0
        w0 = self.w0
        # Calculate position along beam
        z = np.dot(k, R) - np.dot(k, R0)

        # Calculate radial offset from center of beam
        r = np.sqrt(np.sum(((R - np.dot(k, R) * k) - (R0 - np.dot(k, R0) * k)) ** 2))

        # Calculate Rayleigh range
        z_R = self.z_R

        return (
            jv(0, self.alpha*r)**2
            * (1 / (1 + (z / z_R) ** 2))
            * np.exp(-2*(r ** 2) / ((w0 ** 2 * (1 + (z / z_R) ** 2))))
        )

    def calculate_profile_integral(self):
        """
        Calculates the integral of the radial intensity profile over all of space.
        """
        # Find a unit vector that is perpendicular to k
        r_vec = (np.array([self.k[1], -self.k[0],0])
                    /np.sqrt(self.k[1]**2+self.k[0]**2))

        integrand = lambda r: r * self.profile_R(r*r_vec)
        integral = 2 * np.pi * quad(integrand, 0.0, np.inf, limit = 100)[0]

        return integral


@dataclass
class MeasuredBeam(Intensity):
    """
    Class for microwave intensity profiles that approximate the profile measured
    for the 13.4 GHz microwaves.
    """

    power: float
    sigma: float
    R0: np.ndarray
    k: np.ndarray
    freq: float

    def __post_init__(self):
        self.wavelength = constants.c / self.freq
        self.z_R = 4 * np.pi * self.sigma ** 2 / self.wavelength
        self.integral = self.calculate_profile_integral()

    def I_R(self, R: np.ndarray, power=None) -> float:
        """
        Calculates the intensity at point R.
        """
        if not power:
            power = self.power

        return power * self.profile_R(R) / self.integral

    def profile_R(self, R) -> float:
        """
        Calculates the relative intensity at position R.

        Normalized so that intensity = 1 at r = 0, z = 0. 
        """
        k = self.k
        R0 = self.R0
        sigma = self.sigma
        # Calculate position along beam
        z = np.dot(k, R) - np.dot(k, R0)

        # Calculate radial offset from center of beam
        r = np.sqrt(np.sum(((R - np.dot(k, R) * k) - (R0 - np.dot(k, R0) * k)) ** 2))

        # Calculate Rayleigh range
        z_R = self.z_R

        return (1 / (1 + (z / z_R) ** 2)) * self.radial_profile_R(r, z)

    def radial_profile_R(self, r: float, z: float) -> float:
        """
        Returns the intensity at given radial position. Normalized so that
        intensity at r = 0 is 1.
        """
        c2 = -1.53275189
        c4 = -3.28776915
        c6 = 10.0840643
        c8 = -7.63098245
        c10 = 1.85758515
        sigma = 0.43530142

        z_R = self.z_R

        # Convert r to inches due since the fit parameters are in inches
        r = r / 0.0254

        # Scale to take into account the divergence of the beam
        r = r / np.sqrt(1 + (z / z_R) ** 2)

        return np.exp(-1 / 2 * (r / sigma) ** 2) * (
            1
            + c2 * (r) ** 2
            + c4 * (r) ** 4
            + c6 * (r) ** 6
            + c8 * (r) ** 8
            + c10 * (r) ** 10
        )

    def calculate_profile_integral(self, z: float = 0):
        """
        Calculates the integral of the radial intensity profile over all of space.
        """
        integrand = lambda r: r * self.radial_profile_R(r, z)
        integral = 2 * np.pi * quad(integrand, 0.0, np.inf)[0]

        return integral


@dataclass
class BackgroundField(Intensity):
    """
    Uniform field profile for representing a bacground due to scattered microwaves.
    
    inputs:
    lims:   List of limits for the box where the field is defined (m)
    intensity:  Intensity of the field (W/m^2)
    """

    lims: List[Tuple[float]]
    intensity: float = 0

    def I_R(self, R: np.ndarray, power=None) -> float:
        """
        Calculates the intensity at point R.
        """
        # If R is not inside box return zero
        for i in range(len(self.lims)):
            if not (self.lims[i][0] < R[i] < self.lims[i][1]):
                return 0

        # Otherwise return intensity
        return self.intensity

