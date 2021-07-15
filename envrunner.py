import torch
import numpy as np

#Environment Runner
class EnvRunner:
    #Constructor
    def __init__(self, s_dim, a_dim, gamma=0.99, lamb=0.95, max_step=2048, device='cpu'):
        self.s_dim = s_dim
        self.a_dim = a_dim
        self.gamma = gamma
        self.lamb = lamb
        self.max_step = max_step
        self.device = device

        #Storages (state, action, value, reward, a_logp)
        self.mb_states = np.zeros((self.max_step, self.s_dim), dtype=np.float32)
        self.mb_actions = np.zeros((self.max_step, self.a_dim), dtype=np.float32)
        self.mb_values = np.zeros((self.max_step,), dtype=np.float32)
        self.mb_rewards = np.zeros((self.max_step,), dtype=np.float32)
        self.mb_a_logps = np.zeros((self.max_step,), dtype=np.float32)
    
    #Compute discounted return
    def compute_discounted_return(self, rewards, last_value):
        returns = np.zeros_like(rewards)
        n_step = len(rewards)

        for t in reversed(range(n_step)):
            if t == n_step - 1:
                returns[t] = rewards[t] + self.gamma * last_value
            else:
                returns[t] = rewards[t] + self.gamma * returns[t+1]

        return returns
    
    #Compute generalized advantage estimation (Optional)
    def compute_gae(self, rewards, values, last_value):
        advs = np.zeros_like(rewards)
        n_step = len(rewards)
        last_gae_lam = 0.0

        for t in reversed(range(n_step)):
            if t == n_step - 1:
                next_value = last_value
            else:
                next_value = values[t+1]

            delta = rewards[t] + self.gamma*next_value - values[t]
            advs[t] = last_gae_lam = delta + self.gamma*self.lamb*last_gae_lam

        return advs + values

    #Run an episode using the policy net & value net
    def run(self, env, policy_net, value_net):
        #Run an episode
        state = env.reset()   #Initial state
        episode_len = self.max_step

        for step in range(self.max_step):
            state_tensor = torch.tensor(np.expand_dims(state, axis=0), dtype=torch.float32, device=self.device)
            action, a_logp = policy_net(state_tensor)
            value = value_net(state_tensor)

            action = action.cpu().numpy()[0]
            a_logp = a_logp.cpu().numpy()
            value  = value.cpu().numpy()

            self.mb_states[step] = state
            self.mb_actions[step] = action
            self.mb_a_logps[step] = a_logp
            self.mb_values[step] = value

            state, reward, done, info = env.step(action)
            self.mb_rewards[step] = reward

            if done:
                episode_len = step + 1
                break
        
        #Compute returns
        last_value = value_net(
            torch.tensor(np.expand_dims(state, axis=0), dtype=torch.float32, device=self.device)
        ).cpu().numpy()

        mb_returns = self.compute_discounted_return(self.mb_rewards[:episode_len], last_value)
        '''
        mb_returns = self.compute_gae(
            self.mb_rewards[:episode_len], 
            self.mb_values[:episode_len],
            last_value
        )
        '''
        return self.mb_states[:episode_len], \
                self.mb_actions[:episode_len], \
                self.mb_a_logps[:episode_len], \
                self.mb_values[:episode_len], \
                mb_returns, \
                self.mb_rewards[:episode_len]
