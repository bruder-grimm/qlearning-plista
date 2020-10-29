import argparse
import os

import numpy as np
import retro
from tensorflow.keras.layers import Dense, InputLayer
from tensorflow.keras.models import Sequential
from Actions import Actions


class Player:
    def __init__(self, filepath):
        self.env = retro.make(
            'StreetFighterIISpecialChampionEdition-Genesis',
            scenario='scenario',
            obs_type=retro.Observations.RAM
        )
        self.actions = Actions()
        self.input_size = self.env.observation_space.shape[0]
        self.output_size = len(self.actions.action_set)

        self.model = Sequential()

        self.model.add(InputLayer(input_shape=self.input_size))

        self.model.add(Dense(512, activation='relu', kernel_initializer='RandomNormal'))
        self.model.add(Dense(265, activation='relu', kernel_initializer='RandomNormal'))
        self.model.add(Dense(128, activation='relu', kernel_initializer='RandomNormal'))
        self.model.add(Dense(64, activation='relu', kernel_initializer='RandomNormal'))
        self.model.add(Dense(self.output_size, activation='linear', kernel_initializer='RandomNormal'))

        self.model.compile(loss='mse')

        if os.path.isfile(filepath):
            self.model.load_weights(filepath)

    def run(self):
        state = self.env.reset()
        state = np.reshape(scale(state), [1, self.input_size])

        done = False

        while not done:
            act_values = self.model.predict(state)
            _, actions = self.actions.get_next_action(np.argmax(act_values[0]))

            for action in actions:
                observation, _, done, _ = self.env.step(action)
                self.env.render()
                state = np.reshape(scale(observation), [1, self.input_size])


def scale(x):
    return x / 255


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', help='Name of the model in this directory')
    args = parser.parse_args()

    streetfighter_play = Player(args.model)
    streetfighter_play.run()
