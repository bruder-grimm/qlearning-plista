import numpy as np
import logging

# -----------------------------------------------------------------------------------------------------------------


# define some cool streetfighter actions here
#     #     [   "B",       "A",        "MODE", "START",  "UP", "DOWN", "LEFT", "RIGHT",      "C",      "Y",         "X",        "Z"     ]
#     #     [med kick, light kick,          ----                                          hard kick, med punch, light punch, hard punch ]
#   duck = np.array([0, 0,                  0, 0,                0, 1, 0, 0,                  0,        0,           0,          0      ])
#  punch = np.array([0, 0,                  0, 0,                0, 0, 0, 0,                  0,        1,           0,          0      ])

duck    = np.array([0, 0,           0, 0,                   0, 1, 0, 0,                     0, 0, 0, 0])
roll    = np.array([0, 0,           0, 0,                   0, 1, 0, 1,                     0, 0, 0, 0])
forward = np.array([0, 0,           0, 0,                   0, 0, 0, 1,                     0, 0, 0, 0])
punch   = np.array([0, 0,           0, 0,                   0, 0, 0, 0,                     0, 1, 0, 0])


# -----------------------------------------------------------------------------------------------------------------
# No need to worry about these functions

def frame_skip(frames_to_skip):
    return [np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])] * frames_to_skip


def wait_for_reward(max_frames):
    return [np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])] * max_frames


def hold(action, period):
    return [action] * period


def one(action):
    return [action]


class Actions:
    noop = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    wait = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])

    # -----------------------------------------------------------------------------------------------------------------
    # Create your predefined actions like this, some tweaking may be necessay

    hadouken = [
        hold(duck, 3),       # you wanna hold down for 3 frames, pretty self explanatory no?
        hold(roll, 3),
        hold(forward, 3),
        frame_skip(2),       # add some grace period to make the move feel more human ;)
        one(punch),          # and then just a 1 frame punch to get going
        wait_for_reward(35)  # account for travel time and observe if we've hit gold
    ]

    shoryoken = [
        hold(forward, 3),
        hold(duck, 3),
        hold(roll, 3),
        one(punch),
        wait_for_reward(20)  # this is close combat, but the animation takes time to finish
    ]

    # Don't forget to add your actions to the action set, the agent will learn using these actions
    # Maybe try and add some defensive actions as well?
    action_set = [
        ("hadouken", hadouken),
        ("shoryoken", shoryoken),
    ]

    # -----------------------------------------------------------------------------------------------------------------
    # This is only to be called by the agent, no need to worry about the function
    def get_next_action(self, selected_action):
        if selected_action > len(self.action_set) or selected_action < 0:
            logging.warning('Call for invalid action by agent, returning noop. Have you added actions yet?')
            return self.noop

        name, action = self.action_set[selected_action]
        logging.info("Agent requested action sequence for: ", name)

        return name, [item for sublist in action for item in sublist]
