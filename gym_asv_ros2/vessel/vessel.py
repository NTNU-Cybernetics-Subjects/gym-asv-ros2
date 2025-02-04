from typing import NamedTuple
import matplotlib.pyplot as plt


import numpy as np
import shapely
from numpy import linalg
from scipy.integrate import solve_ivp

import gym_asv_ros2.utils.geom_utils as geom

from rich.traceback import install
install()

class VesselParamters(NamedTuple):
    """TODO:"""

    ## Dynamic model terms
    # Mass and Inertia
    m: float = 15.0  # mass (kg) / added 0.5 kg for batteries
    x_g: float = -0.1  # CG for hull in x-axis only (m)
    I_z: float = 3.87  # calculated from estimating the pontoons as boxes

    # Hydrodynamic added mass
    X_udot: float = -26.7704
    Y_vdot: float = -7.5579
    N_rdot: float = -21.7790

    # Linear damping terms
    X_u: float = -29.3484
    Y_v: float = -51.5469
    N_r: float = -44.6517
    Y_r: float = -0.1  # Just a guess, making the sim more realistic when driving
    N_v: float = -0.1  # Just a guess, making the sim more realistic when driving


class Vessel:
    def __init__(self, init_state: np.ndarray, width: float, length: float) -> None:
        """Initialize a vessel."""

        self._state = init_state
        self._prev_inputs = np.array([0,0])
        self._prev_states = np.array([init_state])
        self._input = np.array([0,0])

        self._step_counter = 0

        self._width = width
        self._length = length

        self._model_params = VesselParamters()

        # Thruster params  TODO: The thruster logic should be seperated in its own module somehow
        self._left_thrust_arm_dissplacment = 55.21
        self._right_thrust_arm_dissplacment = 27.56
        self._max_forward_force = -0.285
        self._max_backward_force = 0.285

    @property
    def width(self) -> float:
        """Width of vessel in meters."""
        return self._width

    @property
    def length(self) -> float:
        """Lenght of the vessel in meters."""
        return self._length

    @property
    def position(self) -> np.ndarray:
        """Returns an array holding the position of the AUV in cartesian
        coordinates."""
        return self._state[0:2]

    @property
    def heading(self) -> float:
        """Returns the heading of the AUV with respect to true north."""
        return self._state[2]

    @property
    def velocity(self) -> np.ndarray:
        """Returns the surge and sway velocity of the AUV."""
        return self._state[3:5]

    @property
    def speed(self) -> float:
        """Returns the speed of the AUV."""
        return float(linalg.norm(self.velocity))

    @property
    def yaw_rate(self) -> float:
        """Returns the rate of rotation about the z-axis."""
        return self._state[5]

    @property
    def max_speed(self) -> float:
        """Returns the maximum speed of the AUV."""
        return 3

    @property
    def course(self) -> float:
        """Returns the course angle of the AUV with respect to true north."""
        crab_angle = np.arctan2(self.velocity[1], self.velocity[0])
        return self.heading + crab_angle

    # TODO: Implement
    @property
    def boundary(self) -> shapely.geometry.Polygon:
        """Returns the boundary of the vessel."""
        return NotImplemented

    # The ODE to solve (Continuous)
    def _state_dot(self, h, state, input):
        # States
        psi = state[2]
        nu = state[3:]
        u, v, r = nu

        u1 = input[0]
        u2 = input[1]
        l1 = self._left_thrust_arm_dissplacment
        l2 = self._right_thrust_arm_dissplacment

        # Model matrices TODO: force should be modeleded?
        tau = np.array([u1 + u2, 0, -l1 * u1 - l2 * u2]).T

        p = self._model_params
        M = np.array(
            [
                [p.m - p.X_udot, 0, 0],
                [0, p.m - p.Y_vdot, p.m * p.x_g],
                [0, p.m * p.x_g, p.I_z - p.N_rdot],
            ]
        )
        M_inv = np.linalg.inv(M)

        N = np.array(
            [
                [-p.X_u, -p.m * r, -p.m * p.x_g * r + p.Y_vdot * v],
                [p.m * r, -p.Y_v, -p.X_udot * u],
                [p.m * p.x_g * r - p.Y_vdot * v, p.X_udot * u, -p.N_r],
            ]
        )
        
        # RHS
        eta_dot = geom.Rzyx(0, 0, geom.princip(psi)).dot(nu)
        nu_dot = M_inv.dot(tau - N.dot(nu))
        state_dot = np.concatenate([eta_dot, nu_dot])
        return state_dot


    def action_to_thrust(self, action):
        """Simple linear trasformation from action to force."""
        force = 0
        if action > 0:
            force = np.clip(action, -1, 1) * self._max_forward_force
        if action < 0:
            force = np.clip(action, -1, 1) * self._max_backward_force

        return force

    def step(self, action: np.ndarray, h: float) -> None:
        """Simulates the vessel one step foward after applying the given action.

        @params
        -------
        action: np.ndarray[left_motor_input, right_motor_input]
        """

        self._input = np.array(
            [self.action_to_thrust(action[0]), self.action_to_thrust(action[1])]
        )

        solution = solve_ivp(self._state_dot, [0, h], self._state, args=(self._input,), method="BDF") # TODO: check this
        self._state = solution.y[:,-1]
        self._state[2] = geom.princip(self._state[2])

        print(self._prev_states)
        print(self._state)
        self._prev_states = np.vstack([self._prev_states, self._state])
        self._prev_inputs = np.vstack([self._prev_inputs, self._input])

        self._step_counter += 1

        # TODO: Store states?

    # TODO: Implement
    def perceive(self) -> np.ndarray:
        """
        Simulates the sensor suite and returns observation arrays of the environment.

        Returns
        -------
        """
        return NotImplemented

if __name__ == '__main__':
    init_state = np.zeros((6,))
    # print(init_state)
    vessel = Vessel(init_state, 1, 1)

    action = np.array([1,1])
    # print(action)

    for i in range(10):
        vessel.step(action, 0.1)

    t = np.arange(0, vessel._step_counter+1)
    plt.plot(t, vessel._prev_states[:,0])
    plt.show()
    # print(t, vessel._prev_states[:,0])


