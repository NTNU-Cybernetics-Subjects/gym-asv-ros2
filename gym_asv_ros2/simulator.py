from pynput import keyboard
from pynput.keyboard import Key
import gymnasium as gym
import numpy as np
# from time import time

class Game:
    def __init__(self) -> None:
        # self.env = env

        self.action = np.array([0.0, 0.0])  # Action to send to vessel

        self.quit = False
        self.restart = False

    def press(self, k):
        if k.char == "u":
            self.action[0] = 0.5

        if k.char == "i":
            self.action[1] = 0.5

        if k.char == "j":
            self.action[0] = -0.5

        if k.char == "k":
            self.action[1] = -0.5

    def release(self, k):
        if k.char == "r":
            self.restart = True

        if k.char == "q":
            self.quit = True

        if k.char == "u":
            self.action[0] = 0

        if k.char == "i":
            self.action[1] = 0

        if k.char == "j":
            self.action[0] = 0

        if k.char == "k":
            self.action[1] = 0

    def start_listner(self):
        listner = keyboard.Listener(on_press=self.press, on_release=self.release)
        listner.start()

