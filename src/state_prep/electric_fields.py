from dataclasses import dataclass
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
import scipy
from scipy import constants

from .core import StaticField
from .trajectory import Trajectory


@dataclass
class ElectricField(StaticField):
    """
    Class for representing electric fields.
    """

    def __init__(self, E_R: Callable = None):
        self.E_R = E_R
        self.R_to_r = np.array([[0, 0, 1], [0, -1, 0], [1, 0, 0]])

    def __add__(self, other):
        """
        Addition of electric fields.
        """
        assert (
            type(other) != ElectricField
        ), f"Can't add {type(other)} and ElectricField"

        E_R = lambda R: self.E_R(R) + other.E_R(R)
        return ElectricField(E_R)

    def __sub__(self, other):
        """
        Subtraction of electric fields
        """
        return self + (-1) * other

    def __mul__(self, a):
        """
        Scalar multiplication of electric field.
        """
        E_R = lambda R: a * self.E_R(R)
        return ElectricField(E_R)

    def __neg__(self):
        """
        Negative of electric field
        """
        return -1 * self

    def get_E_R(self, R: np.ndarray) -> np.ndarray:
        """
        Returns the value of the electric field at position R in the XYZ coordinates.
        """
        return self.E_R(R)

    def get_E_r(self, R: np.ndarray) -> np.ndarray:
        """
        Returns the value of the electric field in the xyz basis at position R in XYZ
        coordinates.        
        """
        return self.R_to_r @ self.get_E_R(R)

    def get_E_t_func(self, R_t: Callable) -> Callable:
        """
        Returns a function that gives the electric field experienced along
        a given trajectory over time.
        """
        return lambda t: self.get_E_R(R_t(t))

    def plot(self, trajectory: Trajectory, ax=None):
        """
        Plots the electric field along the given trajectory
        """
        E_t = self.get_E_t_func(trajectory.R_t)
        T = trajectory.get_T()

        t_array = np.linspace(0, T, 1000)

        Es = E_t(t_array)

        if not ax:
            fig, ax = plt.subplots()

        ax.plot(t_array / 1e-6, Es[0], label=r"E_x")
        ax.plot(t_array / 1e-6, Es[1], label=r"E_y")
        ax.plot(t_array / 1e-6, Es[2], label=r"E_z")
        ax.set_xlabel(r"Time / $\mu$s")
        ax.set_ylabel("Electric field / V/cm")
        ax.set_title("Electric field experienced by molecule over time")
        ax.legend()


def linear_E_field(x, z0=0, E0=200, k=100, n=np.array((0, 0, 1))):
    """
    Function that gives the electric due to an electric field that varies linearly along z.
    
    inputs:
    x = position where E-field is to be evaluated [m]
    z0 = position where value of E-field is E0 [m]
    E0 = magnitude of E-field at z = z0 [V/cm]
    k = gradient of electric field magnitude along z [V/cm/m]
    n = orientation of E-field in space

    returns
    E = Electric field vector
    """

    # Determine z-position
    z = x[2]

    # Calculate electric field
    E = n * (k * (z - z0) + E0)

    return E


def E_field_lens(x, z0=0, V=3e4, R=1.75 * 0.0254 / 2, L=0.60, l=20e-3):
    """
    A function that gives the electric field due to to the electrostatic lens
    E_vec = electric field vector in V/cm
    x = position (m)
    z0 = center of lens (m)
    V = voltage on electrodes (V)
    R = radius of lens (m)
    L = length of lens (m)
    l = decay length of lens field (m)
    """

    # Calculate electric field vector (assumed to be azimuthal and perpendicular to r_vec)
    E_vec = 2 * V / R ** 2 * np.array((-x[1], x[0], 0)).reshape(3, 1)

    # Scale the field by a tanh function so it falls off outside the lens
    E_vec = (
        E_vec
        * (np.tanh((x[2] - z0 + L / 2) / l) - np.tanh((x[2] - z0 - L / 2) / l))
        / 2
    )

    return E_vec / 100




# Define a function that gives the Ez component of the lens field as a function of position
def lens_Ez(x, lens_z0, lens_L):
    """
    Function that evaluates the z-component of the electric field produced by the lens based on position
    
    inputs:
    x = position (in meters) where Ex is evaluated (np.array) 
    
    returns:
    E = np.array that only has z-component (in V/cm)
    """

    # Determine radial position and the angle phi in cylindrical coordinates
    r = np.sqrt(x[0] ** 2 + x[1] ** 2)

    # Calculate the value of the electric field
    # Calculate radial scaling
    c2 = 13673437.6
    c4 = 9.4893e09
    radial_scaling = c2 * r ** 2 + c4 * r ** 4

    # Angular function
    angular = 2 * x[0] * x[1] / (r ** 2)

    # In z the field varies as a Gaussian
    sigma = 12.811614314258744 / 1000
    z1 = lens_z0 - lens_L / 2
    z2 = lens_z0 + lens_L / 2
    z_function = np.exp(-((x[2] - z1) ** 2) / (2 * sigma ** 2)) - np.exp(
        -((x[2] - z2) ** 2) / (2 * sigma ** 2)
    )

    E_z = radial_scaling * angular * z_function

    return np.array((0, 0, E_z))


def E_field_ring(x, z0=0, V=2e4, R=2.25 * 0.0254):
    """
    A function that calculates the axial electric field due to a ring electrode
    E_vec = electric field vector in V/m
    x = position (m)
    V = voltage applied to ring (V)
    R = radius of ring (m)
    """
    # Determine z-position
    z = x[2]

    E = np.zeros(x.shape)

    # Calculate electric field
    # The electric field is scaled so that for R = 2.25*0.0254m, get a max field
    # of E = 100000 V/m for a voltage of 20 kV
    scaling_factor = (2.25 * 0.0254) ** 2 / 20e3 * (1e5) * 3 * np.sqrt(3) / 2
    mag_E = scaling_factor * (z - z0) / ((z - z0) ** 2 + R ** 2) ** (3 / 2) * V
    E[2] = mag_E

    # Return the electric field as an array which only has a z-component (approximation)
    return E / 100

def E_field_tanh(x, z0=0, V=30e3, l = 1):
    """
    calculates the electric field along trajectory thatd decays in a tanh fashion
    """
    z = x[2]
    mag_E = V* (1- np.tanh((z - z0)/l))
    
    E = np.zeros(x.shape)

    E[2] = mag_E
    
    return E