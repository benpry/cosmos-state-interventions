import numpy as np


def suggestion(qtable, state, reward, new_state, action, learning_rate):
    # update q value for action closest to the intervention
    if new_state > state:
        action = 1
    else:
        action = 0

    qtable[state, action] = qtable[state, action] + learning_rate * (
        1 + np.max(qtable[new_state, :]) - qtable[state, action]
    )
    return qtable


def reset(qtable, state, reward, new_state, action, learning_rate):
    # get the reward of the new state, associate it with the action you would have taken
    qtable[state, action] = qtable[state, action] + learning_rate * (
        reward + np.max(qtable[new_state, :]) - qtable[state, action]
    )
    return qtable


def interrupt(qtable, state, reward, new_state, action, learning_rate):
    # ignore the timestep entirely
    return qtable


def impede(qtable, state, reward, new_state, action, learning_rate):
    # fixed negative reward for whatever you were going to do beforehand
    qtable[state, action] = qtable[state, action] + learning_rate * (
        -1 + np.max(qtable[new_state, :]) - qtable[state, action]
    )
    return qtable


INTERP_RULES = {
    "suggestion": suggestion,
    "reset": reset,
    "interrupt": interrupt,
    "impede": impede,
}
