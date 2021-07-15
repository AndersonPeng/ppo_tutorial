import torch
import torch.nn as nn
import numpy as np

#PPO algorithm
class PPO:
    #Constructor
    def __init__(self, policy_net, value_net, lr=1e-4, max_grad_norm=0.5, ent_weight=0.01, clip_val=0.2, sample_n_epoch=4, sample_mb_size=64, device='cpu'):
        self.policy_net = policy_net
        self.value_net = value_net
        self.max_grad_norm = max_grad_norm
        self.ent_weight = ent_weight
        self.clip_val = clip_val
        self.sample_n_epoch = sample_n_epoch
        self.sample_mb_size = sample_mb_size
        self.device = device
        self.opt_polcy = torch.optim.Adam(policy_net.parameters(), lr)
        self.opt_value = torch.optim.Adam(value_net.parameters(), lr)
    
    #Train the policy net & value net using PPO
    def train(self, mb_states, mb_actions, mb_old_values, mb_advs, mb_returns, mb_old_a_logps):
        #Convert numpy array to tensor
        mb_states = torch.from_numpy(mb_states).to(self.device)
        mb_actions = torch.from_numpy(mb_actions).to(self.device)
        mb_old_values = torch.from_numpy(mb_old_values).to(self.device)
        mb_advs = torch.from_numpy(mb_advs).to(self.device)
        mb_returns = torch.from_numpy(mb_returns).to(self.device)
        mb_old_a_logps = torch.from_numpy(mb_old_a_logps).to(self.device)
        episode_length = len(mb_states)
        rand_idx = np.arange(episode_length)
        sample_n_mb = episode_length // self.sample_mb_size

        if sample_n_mb <= 0:
            sample_mb_size = episode_length
            sample_n_mb = 1
        else:
            sample_mb_size = self.sample_mb_size

        for i in range(self.sample_n_epoch):
            np.random.shuffle(rand_idx)

            for j in range(sample_n_mb):
                #Randomly sample a batch for training
                sample_idx = rand_idx[j*sample_mb_size : (j+1)*sample_mb_size]
                sample_states = mb_states[sample_idx]
                sample_actions = mb_actions[sample_idx]
                sample_old_values = mb_old_values[sample_idx]
                sample_advs = mb_advs[sample_idx]
                sample_returns = mb_returns[sample_idx]
                sample_old_a_logps = mb_old_a_logps[sample_idx]

                sample_a_logps, sample_ents = self.policy_net.evaluate(sample_states, sample_actions)
                sample_values = self.value_net(sample_states)
                ent = sample_ents.mean()

                #Compute value loss
                v_pred_clip = sample_old_values + torch.clamp(sample_values - sample_old_values, -self.clip_val, self.clip_val)
                v_loss1 = (sample_returns - sample_values).pow(2)
                v_loss2 = (sample_returns - v_pred_clip).pow(2)
                v_loss = torch.max(v_loss1, v_loss2).mean()

                #Compute policy gradient loss
                ratio = (sample_a_logps - sample_old_a_logps).exp()
                pg_loss1 = -sample_advs * ratio
                pg_loss2 = -sample_advs * torch.clamp(ratio, 1.0-self.clip_val, 1.0+self.clip_val)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean() - self.ent_weight*ent

                #Train actor
                self.opt_polcy.zero_grad()
                pg_loss.backward()
                nn.utils.clip_grad_norm_(self.policy_net.parameters(), self.max_grad_norm)
                self.opt_polcy.step()

                #Train critic
                self.opt_value.zero_grad()
                v_loss.backward()
                nn.utils.clip_grad_norm_(self.value_net.parameters(), self.max_grad_norm)
                self.opt_value.step()

        return pg_loss.item(), v_loss.item(), ent.item()
