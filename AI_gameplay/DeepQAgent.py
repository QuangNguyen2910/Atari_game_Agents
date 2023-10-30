import numpy as np
import torch
from DeepQNetwork import DeepQNetwork

class Agent:
    def __init__(self, gamma, epsilon, lr, input_dims, batch_size, n_actions,
                 max_mem_size=10000, eps_end=0.01, eps_dec=5e-6, update_after_actions= 1, replace_target_cnt=1000):

        # Hyper parameter
        self.gamma = gamma
        self.epsilon = epsilon
        self.eps_min = eps_end
        self.eps_dec = eps_dec
        self.lr = lr
        self.action_space = [i for i in range(n_actions)]
        self.mem_size = max_mem_size
        self.batch_size = batch_size
        self.mem_cntr = 0
        self.iter_cntr = 0
        self.replace_target_cnt = replace_target_cnt
        self.update_after_actions = update_after_actions

        # Online network
        self.Q_eval = DeepQNetwork(lr, n_actions=n_actions,
                                   input_dims=input_dims)

        # Target network
        self.Q_target = DeepQNetwork(lr, n_actions=n_actions,
                                   input_dims=input_dims)

        # Replay memory
        self.state_memory = np.zeros((self.mem_size, *input_dims), dtype=np.float32)
        self.new_state_memory = np.zeros((self.mem_size, *input_dims), dtype=np.float32)
        self.action_memory = np.zeros(self.mem_size, dtype=np.int8)
        self.reward_memory = np.zeros(self.mem_size, dtype=np.int8)
        self.terminal_memory = np.zeros(self.mem_size, dtype=np.bool_)

    # Push a transition to replay memory
    def store_transition(self, state, action, reward, state_, terminal):
        index = self.mem_cntr % self.mem_size
        self.state_memory[index] = state
        self.new_state_memory[index] = state_
        self.reward_memory[index] = reward
        self.action_memory[index] = action
        self.terminal_memory[index] = terminal

        self.mem_cntr += 1

    def save_model(self):
      print('...saving model...')
      self.Q_eval.save_checkpoint()

    def load_model(self):
      print('...loading model...')
      self.Q_eval.load_checkpoint()

    def choose_action(self, observation):
        # Epsilon Greedy
        if np.random.random() > self.epsilon:
            state = torch.tensor(np.array([observation])).to(self.Q_eval.device)
            actions = self.Q_eval.forward(state)
            action = torch.argmax(actions).item()
        else:
            action = np.random.choice(self.action_space)

        return action

    def learn(self):
        if self.mem_cntr < self.batch_size:
            return

        self.Q_eval.optimizer.zero_grad()

        # Update target network
        if self.iter_cntr % self.replace_target_cnt == 0:
          self.Q_target.load_state_dict(self.Q_eval.state_dict())

        # if self.iter_cntr % self.update_after_actions == 0:
        max_mem = min(self.mem_cntr, self.mem_size)

        # Take a random batch from replay memory
        batch = np.random.choice(max_mem, self.batch_size, replace=False)
        batch_index = np.arange(self.batch_size, dtype=np.int8)
        state_batch = torch.tensor(self.state_memory[batch]).to(self.Q_eval.device)
        new_state_batch = torch.tensor(self.new_state_memory[batch]).to(self.Q_eval.device)
        action_batch = self.action_memory[batch]
        reward_batch = torch.tensor(self.reward_memory[batch]).to(self.Q_eval.device)
        terminal_batch = torch.tensor(self.terminal_memory[batch]).to(self.Q_eval.device)

        # Estimate Q value
        q_eval = self.Q_eval.forward(state_batch)[batch_index, action_batch]
        q_next = self.Q_target.forward(new_state_batch)
        q_next[terminal_batch] = 0.0

        q_target = reward_batch + self.gamma*torch.max(q_next, dim=1)[0]

        # Propagation
        loss = self.Q_eval.loss(q_target, q_eval).to(self.Q_eval.device)
        loss.backward()
        self.Q_eval.optimizer.step()

        self.iter_cntr += 1

        # Epsilon decay
        self.epsilon = self.epsilon - self.eps_dec if self.epsilon > self.eps_min else self.eps_min