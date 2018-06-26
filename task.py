import numpy as np
from physics_sim import PhysicsSim

class Task():
    """Task (environment) that defines the goal and provides feedback to the agent."""
    def __init__(self, init_pose=None, init_velocities=None, 
        init_angle_velocities=None, runtime=5., target_pos=None):
        """Initialize a Task object.
        Params
        ======
            init_pose: initial position of the quadcopter in (x,y,z) dimensions and the Euler angles
            init_velocities: initial velocity of the quadcopter in (x,y,z) dimensions
            init_angle_velocities: initial radians/second for each of the three Euler angles
            runtime: time limit for each episode
            target_pos: target/goal (x,y,z) position for the agent
        """
        # Simulation
        self.sim = PhysicsSim(init_pose, init_velocities, init_angle_velocities, runtime) 
        self.action_repeat = 3

        self.state_size = self.action_repeat * 6
        self.action_low = 0
        self.action_high = 900
        self.action_size = 4

        # Goal
        self.target_pos = target_pos if target_pos is not None else np.array([0., 0., 1.]) 

    def get_reward(self,rotor_speeds):
        """Uses current pose of sim to return reward."""
        # Calculate a 5 % of the thrust as incentive until it reaches the correct high
        thrust_incentive = 0.0 if self.target_pos.all() == np.array([0., 0., 1., 0., 0., 0.]).all() else (np.asarray(self.sim.get_propeler_thrust(rotor_speeds)) * 0.05).sum()
        # If you are at the correct high give a 0.3 of incentive so you stay
        hovering_incentive = 0.01 if self.sim.pose[:3].all() != self.target_pos.all() else 0.0
        # I don't want it to move sideways so I will penalized it if the speed on x or y changes from 0
        velocity_penalty = 0.0 if self.sim.v[0] == 0.0 and self.sim.v[1] == 0.0 else .1 * (self.sim.v[0] + self.sim.v[1])
        # Penalty if your euler angles are not 0
        euler_angles = .2*(abs(self.sim.pose[3:] - self.target_pos)).sum() if self.sim.pose[3:] != self.target_pos[3:] else 0.0
        
        # Calculate reward
        reward = 1.-.2*(abs(self.sim.pose[:3] - self.target_pos)).sum() - euler_angles - velocity_penalty
        return reward

    def step(self, rotor_speeds):
        """Uses action to obtain next state, reward, done."""
        reward = 0
        pose_all = []
        for _ in range(self.action_repeat):
            done = self.sim.next_timestep(rotor_speeds) # update the sim pose and velocities
            reward += self.get_reward(rotor_speeds) 
            pose_all.append(self.sim.pose)
        next_state = np.concatenate(pose_all)
        return next_state, reward, done

    def reset(self):
        """Reset the sim to start a new episode."""
        self.sim.reset()
        state = np.concatenate([self.sim.pose] * self.action_repeat) 
        return state