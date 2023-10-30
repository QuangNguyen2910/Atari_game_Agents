import gym
import numpy as np
import cv2
import torch
from matplotlib import pyplot as plt
from DeepQAgent import Agent
from atari_wrapper import wrap_deepmind
from Visualize import save_agent_gif, _label_with_episode_number
# from notebook_video_writer import VideoWriter
cv2.ocl.setUseOpenCL(False)

def plotLearning(scores, x=None, window=5):
  N = len(scores)
  running_avg = np.empty(N)
  for t in range(N):
    running_avg[t] = np.mean(scores[max(0, t-window):(t+1)])
  if x is None:
    x = [i for i in range(N)]
  plt.ylabel('Score')
  plt.xlabel('Game')
  plt.plot(x, running_avg)

def train(x):
  env = gym.make("ALE/Pong-v5", render_mode= 'rgb_array')
  env = wrap_deepmind(env, frame_stack=True, scale=True)
  env.seed(42)
  agent = Agent(gamma=0.99, epsilon=0.01, batch_size=32, n_actions=6, eps_end=0.01,
                  input_dims=(4, 84, 84), lr=0.0001)
  if x == True:
    agent.load_model()
  scores, eps_history = [], []
  n_games = 1001
  best_score = env.reward_range[0]
  # # load model been trained for 700 episodes - 6 hours
  # best_score = 12
  # agent.load_model()
  for i in range(n_games):
    score = 0

    # Run environment with agent control
    done = False
    observation = env.reset()
    observation = np.array(observation).reshape(4, 84, 84)

    while not done:
      action = agent.choose_action(observation)
      observation_, reward, done, info = env.step(action)
      observation_ = np.array(observation_).reshape(4, 84, 84)
      score += reward
      agent.store_transition(observation, action, reward, observation_, done)
      agent.learn()
      observation = observation_

    scores.append(score)
    eps_history.append(agent.epsilon)

    avg_score = np.mean(scores[-30:])

    if avg_score > best_score:
      best_score = avg_score
      agent.save_model()

    if i % 10 == 0:
      print('episode ', i, 'score %.2f' % score,
        'average score %.2f' % avg_score,
        'epsilon %.2f' % agent.epsilon, 'iters %i' % agent.iter_cntr)
  plotLearning(scores, window=10)

def test():
  env = gym.make("ALE/Pong-v5", render_mode= 'rgb_array')
  env = wrap_deepmind(env, frame_stack=True, scale=True)
  # env.seed(42)
  agent = Agent(gamma=0.99, epsilon=0.01, batch_size=32, n_actions=6, eps_end=0.01,
                  input_dims=(4, 84, 84), lr=0.0001)
  agent.load_model()
  done = False
  observation = env.reset()
  observation = np.array(observation).reshape(4, 84, 84)
  env_video = []
  score = 0
  frames = 1000
  while not done:
    action = agent.choose_action(observation)
    env_video.append(_label_with_episode_number(env.render(), score))
    observation, reward, done, info = env.step(action)
    observation = np.array(observation).reshape(4, 84, 84)
    score += reward
  
  print(score)
  save_agent_gif(env_video)

  # with VideoWriter(fps=60) as vw:
  #   for frame in env_video:
  #     vw.add(frame)


if __name__ == '__main__':
  option = input("Enter 1 to train, 2 to test: ")
  if option == '1':
    ops = input("Enter 1 to train from scratch, 2 to train from saved model: ")
    if ops == '1':
      train(False)
    elif ops == '2':
      train(True)
    else:
      print("Invalid option")
  elif option == '2':
    test()
  else:
    print("Invalid option")