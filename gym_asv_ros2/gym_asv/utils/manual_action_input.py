import os
if "ROS_DISTRO" in os.environ:
    class keyboard:
        def Listener(self):
            pass
else:
    from pynput import keyboard

# from pynput.keyboard import Key
import gymnasium as gym
import numpy as np
# from time import time


class KeyboardListner:
    def __init__(self) -> None:
        # self.env = env

        self.action = np.array([0.0, 0.0])  # Action to send to vessel

        self.quit = False
        self.restart = False

        self.listner = None

    def press(self, k):
        try: 
            if k.char == "u":
                self.action[0] = 0.5

            if k.char == "i":
                self.action[1] = 0.5

            if k.char == "j":
                self.action[0] = -0.5

            if k.char == "k":
                self.action[1] = -0.5
        except AttributeError:
            pass

    def release(self, k):
        try:
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
        except AttributeError:
            pass

        # print(self.action)

    def start_listner(self):
        self.listner = keyboard.Listener(on_press=self.press, on_release=self.release)
        self.listner.start()

if __name__ == "__main__":
    g = KeyboardListner()
    g.start_listner()
    g.listner.join()
