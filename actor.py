from keras import layers, models, optimizers
from keras import backend as K

class Actor:

    def __init__(self, state_size, action_size, action_low, action_high):
        """

        :param state_size: (int): Dimension of each state (# features)
        :param action_size: (int): Dimension of each action
        :param action_low: (array) Min value of each continues action (rotors revolutions)
        :param action_high: (array) Min value of each continues action (rotors revolutions)
        """

        self.state_size = state_size
        self.action_size = action_size
        self.action_low = action_low
        self.action_high = action_high
        self.action_range = self.action_high - self.action_low

        # Function to initialize the model
        self.build_model()

    def build_model(self):
        # mu network
        states = layers.Input(shape=(self.state_size,),
                              name='states')
        net = layers.BatchNormalization()(states)
        net = layers.Dense(units=64,
                           activation='relu')(net)
        net = layers.BatchNormalization()(net)
        net = layers.Dense(units=128,
                           activation='relu')(net)
        net = layers.BatchNormalization()(net)
        net = layers.Dense(units=64,
                           activation='relu')(net)
        net = layers.BatchNormalization()(net)
        raw_actions = layers.Dense(units=self.action_size,
                                   activation='sigmoid',
                                   name='raw_actions')(net)

        # Scale output from sigmod funcion that is between 0 and 1 to
        # a probability distribution equivalent to softmax.
        actions = layers.Lambda(lambda x: (x * self.action_range) + self.action_low,
                                name='actions')(raw_actions)

        # Create the keras model
        self.model = models.Model(inputs=states, outputs=actions)

        # Loss function is defined using the Q_value gradients that
        # comes from the Critic value state function and the action
        # value produced by this model (read the paper)
        action_gradients = layers.Input(shape=(self.action_size,))
        loss = K.mean(actions * (- action_gradients))

        # Network optiomizer
        # Set learning rate to 0.01 so the model learns faster.
        optimizer = optimizers.Adam(lr=0.0001)
        updates_op = optimizer.get_updates(params=self.model.non_trainable_weights,loss=loss)

        # Training function
        self.train_fn = K.function(inputs=[self.model.input,
                                           action_gradients,
                                           K.learning_phase()],
                                   outputs=[],
                                   updates=updates_op)

