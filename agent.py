from actor import Actor
from critic import Critic
from task import Task
from utils import OUNoise
from utils import ReplayBuffer
import numpy as np

class AgentDDPG():

    def __init__(self, task):
        """

        :param task: (class instance) Instructions about the goal and reward
        """

        self.task = task
        self.state_size = task.state_size
        self.action_size = task.action_size
        self.action_low = task.action_low
        self.action_high = task.action_high
        self.score = 0.0
        self.best = 0.0

        # Instances of the policy function or actor and the value function or critic
        # Actor critic with Advantage

        # Actor local and target
        self.actor_local = Actor(self.state_size,
                                 self.action_size,
                                 self.action_low,
                                 self.action_high)
        self.actor_target = Actor(self.state_size,
                                  self.action_size,
                                  self.action_low,
                                  self.action_high)

        # Critic local and target
        self.critic_local = Critic(self.state_size,
                                   self.action_size)
        self.critic_target = Critic(self.state_size,
                                    self.action_size)

        # Initialize target model with local model
        self.critic_target.model.set_weights(self.critic_local.model.get_weights())
        self.actor_target.model.set_weights(self.actor_local.model.get_weights())

        # Initialize the Gaussin Noise process
        self.exploration_mu = 0
        self.exploration_theta = 0.15
        self.exploration_sigma = 0.2 
        self.noise = OUNoise(self.action_size,
                             self.exploration_mu,
                             self.exploration_theta,
                             self.exploration_sigma)

        # Initialize the Replay Memory
        self.buffer_size = 100000
        self.batch_size = 64
        self.memory = ReplayBuffer(self.buffer_size, self.batch_size)


        # Parameters for the Algorithm
        self.gamma = 0.99 # Discount factor
        self.tau = 0.01 # Soft update for target parameters Actor Critic with Advantage

    # Actor can reset the episode
    def reset_episode(self):
        self.total_reward = 0.0
        self.count = 0
        # Reset the gaussian noise
        self.noise.reset()
        # Gets a new state from the task
        state = self.task.reset()
        # Protect the state obtaned from the task
        # by storing it as last state
        self.last_state = state
        # Return the state obtained from task
        return state

    # Actor can executes a leraning step
    def step(self, action, reward, next_state, done):
        self.total_reward += reward
        self.count += 1
        # Stored previous state in the replay buffer
        self.memory.add(self.last_state,
                        action,
                        reward,
                        next_state,
                        done)
        # Check to see if you have enough to produce a batch
        # and learn from it
        if len(self.memory) > self.batch_size:
            experiences = self.memory.sample()
            # Train the networks using the experiences
            self.learn(experiences)

    # Actor can interact with the environment by acting
    def act(self,state):
        # Given a state return the action recommended by the policy
        # Reshape the state to fit the keras model input
        state = np.reshape(state, newshape=[-1,self.state_size])
        # Pass the state to the actor local model to get an action
        # recommend for the policy in a state
        action = self.actor_local.model.predict(state)[0]
        # Because we are exploring we add some noise to the
        # action vector
        return list(action + self.noise.sample())

    # This is the Actor learning logic called when the agent
    # take a step to learn
    def learn(self, experiences):
        """
        Learning means that the networks parameters needs to be updated
        Using the experineces batch.
        Network learns from experiences not form interaction with the
        environment
        """
        self.score = self.total_reward / float(self.count) if self.count else 0.0
        if self.score > self.best:
            self.best = self.score
            
        # Reshape the experience tuples in separate arrays of states, actions
        # rewards, next_state, done
        # Your are converting every memeber of the tuple in a column or vector
        states = np.vstack([e.state for e in experiences if e is not None])
        actions = np.array([e.action for e in experiences if e is not None]).astype(np.float32).reshape(-1,self.action_size)
        rewards = np.array([e.reward for e in experiences if e is not None]).astype(np.float32).reshape(-1,1)
        dones = np.array([e.done for e in experiences if e is not None]).astype(np.uint8).reshape(-1,1)
        next_states = np.vstack([e.next_state for e in experiences if e is not None])

        # Firs we pass a batch of next states to the actor so it tell us what actions
        # to execute, we use the actor target network instead of the actor local network
        # because of the advantage principle
        actions_next = self.actor_target.model.predict_on_batch(next_states)

        # The critic evaluates the actions taking by the actor and generates the
        # Q(a,s) value of those actions. This action, state tuple comes from the
        # ReplayBuffer not from interacting with the environment.
        # Remember the Critic or value function inputs is states, actions
        Q_targets_next = self.critic_target.model.predict_on_batch(([next_states,actions_next]))

        # With the Q_targets_next that is a vector of action values Q(s,a) of a random selected
        # next_states from the replay buffer. We calculate the target Q(s,a).
        # For that we use the TD one-step Sarsa equations
        # We make terminal states target Q(s,a) 0 and Non terminal the Q_targtes value
        # This is done to train the critic in a supervise learning fashion.
        Q_targets = rewards + self.gamma * Q_targets_next * (1 - dones)
        self.critic_local.model.train_on_batch(x=[states,actions],y=Q_targets)

        # Train the actor
        action_gradients = np.reshape(self.critic_local.get_action_gradients([states,actions,0]),
                                      (-1, self.action_size))
        self.actor_local.train_fn([states, action_gradients,1]) # Custom training function

        # Soft-update target models
        self.soft_update(self.critic_local.model, self.critic_target.model)
        self.soft_update(self.actor_local.model, self.actor_target.model)

    def soft_update(self, local_model, target_model):
        local_weights = np.array(local_model.get_weights())
        target_weights = np.array(target_model.get_weights())

        assert len(local_weights) == len(target_weights)

        new_weights = self.tau * local_weights + (1 - self.tau) * target_weights
        target_model.set_weights(new_weights)

        