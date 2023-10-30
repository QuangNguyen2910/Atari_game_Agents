import os
import imageio
import numpy as np
from PIL import Image
import PIL.ImageDraw as ImageDraw
import matplotlib.pyplot as plt    

def _label_with_episode_number(frame, episode_num):
    im = Image.fromarray(frame)

    drawer = ImageDraw.Draw(im)

    return im


def save_agent_gif(frames):
    imageio.mimwrite(os.path.join('./Videos/', 'agent.gif'), frames, duration=20)
