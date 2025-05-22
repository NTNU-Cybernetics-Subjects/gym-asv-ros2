# from __future__ import annotations
from typing import NamedTuple

import matplotlib.pyplot as plt
import numpy as np
import shapely
from numpy import linalg
from scipy.integrate import solve_ivp

from gym_asv_ros2.gym_asv.entities import BaseEntity
import gym_asv_ros2.gym_asv.utils.geom_utils as geom

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

class ThrusterParams(NamedTuple):

    left_thrust_dissplacement: float = -0.285
    right_arm_dissplacement: float = 0.285

    max_forward_force: float = 55.21 - 11.21 # Lower force because we get realistic high max speed (This gives max speed = 3 m/s)
    max_backward_force: float = 27.5


class Vessel:
    def __init__(self, init_state: np.ndarray, width: float, length: float) -> None:
        """Initialize a vessel.
        @param init state: [x_ned, y_ned, psi, surge, sway, yaw_rate]"""

        self._init_state = init_state
        self._state = init_state
        self._prev_inputs = np.array([0,0])
        self._prev_states = np.array([init_state])
        self._input = np.array([0,0])

        self._step_counter = 0

        self._width = width
        self._length = length

        # Modeling constants
        self._model_params = VesselParamters() # TODO: Config input on values?

        # Thruster constants
        self._thruster_params = ThrusterParams() # TODO: Config input on values?

        self._max_speed = 3

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
        """Returns the maximum speed of the AUV. [m/s]"""
        return self._max_speed

    @property
    def course(self) -> float:
        """Returns the course angle of the AUV with respect to true north."""
        crab_angle = np.arctan2(self.velocity[1], self.velocity[0])
        return self.heading + crab_angle


    # Original shape
    @property
    def boundary(self) -> shapely.geometry.Polygon:
        """Vessels shape defined keeping the CO in origo"""
        # Defined as x, y in ned
        vertices = [
            (-self.length/2, -self.width/2),
            (-self.length/2, self.width/2),
            (self.length/2, self.width/2),
            (3/2*self.length, 0),
            (self.length/2, -self.width/2),
        ]
        return shapely.geometry.Polygon(vertices)


    # The ODE to solve (Continuous)
    def _state_dot(self, h, state, input):
        # States
        psi = state[2]
        nu = state[3:]
        u, v, r = nu

        u1 = input[0]
        u2 = input[1]
        l1 = self._thruster_params.left_thrust_dissplacement
        l2 = self._thruster_params.right_arm_dissplacement

        # Model matrices TODO: force should be modeleded?
        tau = np.array([u1 + u2, 0, - (l1 * u1) - (l2 * u2)]).T

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


    def action_to_thrust(self, action: np.ndarray):
        """Simple linear trasformation from action to force.

        Action should be [-1,1]
        """
        # force = 0
        if action > 0:
            force = np.clip(action, -1, 1) * self._thruster_params.max_forward_force
        elif action < 0:
            force = np.clip(action, -1, 1) * self._thruster_params.max_backward_force
        else:
            force = 0
        return force

    def step(self, action: np.ndarray, h: float) -> None:
        """Simulates the vessel one step foward after applying the given action.

        @params
        -------
        action: np.ndarray[left_motor_input, right_motor_input]
        """

        self._prev_states = np.vstack([self._prev_states, self._state])
        self._prev_inputs = np.vstack([self._prev_inputs, self._input])

        self._input = np.array(
            [self.action_to_thrust(action[0]), self.action_to_thrust(action[1])]
        )

        # if np.linalg.norm(self._input) > 0:
        #     solution = solve_ivp(self._state_dot, [0, h], self._state, args=(self._input,), method="BDF") # TODO: check this
        #     self._state = solution.y[:,-1]
        #     self._state[2] = geom.princip(self._state[2])
        # else:
        #     self._state[3:5] = np.full_like(self._state[3:5], 0.0)
        
        solution = solve_ivp(self._state_dot, [0, h], self._state, args=(self._input,), method="BDF") # TODO: check this
        self._state = solution.y[:,-1]
        self._state[2] = geom.princip(self._state[2])


        self._step_counter += 1

    def reset(self) -> None:
        self._state = self._init_state
        self._prev_inputs = np.array([0,0])
        self._prev_states = np.array([self._state])
        self._input = np.array([0,0])

        self._step_counter = 0

    # TODO: Implement
    def perceive(self) -> np.ndarray:
        """
        Simulates the sensor suite and returns observation arrays of the environment.

        Returns
        -------
        """
        return NotImplemented



### -- Test Vessel class ---
if __name__ == '__main__':
    init_state = np.zeros((6,))
    # print(init_state)
    vessel = Vessel(init_state, 1, 2)
    # print(list( vessel.boundary.exterior.coords ))

    action1 = np.array([-1.0,1.0])
    action2 = np.array([1.0,-1.0])

    for i in range(200):

        if i < 100:
            a = action1
        else:
            a = action2

        vessel.step(a, 0.1)
        # print(vessel._prev_states)
        # print(vessel.boundary)
        # print(vessel.position)

    # print(vessel._prev_states.shape)
    x_ned = vessel._prev_states[:,0]
    y_ned = vessel._prev_states[:,1]
    psi = vessel._prev_states[:,2]
    surge = vessel._prev_states[:,3]
    sway = vessel._prev_states[:,4]
    yaw_rate = vessel._prev_states[:, 5]

    # t = np.arange(0, len(surge))
    print(np.min(yaw_rate), np.max(yaw_rate))
    # plt.plot(t, surge)


    # t = np.arange(0, vessel._step_counter+1)
    # plt.plot(t, vessel._prev_states[:,0])
    # plt.plot(vessel._prev_states[:,0], vessel._prev_states[:,1])
    # t = len(vessel._prev_states[:,0])
    # plt.plot(t, vessel._prev_states[:,0])
    plt.show()
    # print(t, vessel._prev_states[:,0])


