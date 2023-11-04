import os
import imageio
import numpy as np
from PIL import Image
import PIL.ImageDraw as ImageDraw
import matplotlib.pyplot as plt    

def _label_with_episode_number(frame):
    im = Image.fromarray(frame)

    drawer = ImageDraw.Draw(im)
    # text = "Episode: {}".format(episode_num)
    # drawer.text((0, 0), text)
    return im


def save_agent_gif(frames, agent_type):
    imageio.mimwrite(os.path.join('./Videos/', f'agent_{agent_type}.gif'), frames, duration=20)
