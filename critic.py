from keras import layers, models, optimizers
from keras import backend as K

class Critic:

    def __init__(self, state_size, action_size):
        """

        :param state_size: (int) Dimension of each state
        :param action_size: (int) Dimension of each action
        """
        self.state_size = state_size
        self.action_size = action_size

        # Fuction to build the model
        self.build_model()

    def build_model(self):
        # This model has to account for the states vector and
        # the actions vector
        states = layers.Input(shape=(self.state_size,),
                              name='states')
        actions = layers.Input(shape=(self.action_size,),
                               name='actions')

        # States network path
        net_states = layers.Dense(units=128,
                                  activation='relu')(states)
        net_states = layers.BatchNormalization()(net_states)
        net_states = layers.Dense(units=256,
                                  activation='relu')(net_states)

        # Actions network path
        net_actions = layers.Dense(units=128,
                                   activation='relu')(actions)
        net_actions = layers.BatchNormalization()(net_actions)
        net_actions = layers.Dense(units=256,
                                   activation='relu')(net_actions)

        # Combine both paths
        net = layers.Add()([net_states, net_actions])
        net = layers.Activation('relu')(net)

        # Final layer that produces q^
        Q_values = layers.Dense(units=1,
                                name='q_values')(net)

        # Create model
        self.model = models.Model(inputs=[states,actions],
                                  output=Q_values)

        # Define the optimizer. Using loss function "Mean Square Error"
        # Between the Q_values (this logits) and the Q_targets
        # Calculated interacting with the environment and retrieved
        # from the ReplayBuffer.
        optimizer = optimizers.Adam(lr=0.01)
        self.model.compile(optimizer=optimizer, loss='mse')

        # Compute the action gradients used in the loss function of the actor
        action_gradients = K.gradients(Q_values, actions)

        # Function to share the action_gradients with the actor
        self.get_action_gradients = K.function(inputs=[*self.model.input,
                                                       K.learning_phase()],
                                               outputs=action_gradients)