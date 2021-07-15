import os
import gym
import torch
import torch.nn as nn
import numpy as np
from envrunner import EnvRunner
from model import PolicyNet, ValueNet
from agent import PPO

if __name__ == '__main__':
    #Create the environment
    env = gym.make('BipedalWalker-v3')
    s_dim = env.observation_space.shape[0]
    a_dim = env.action_space.shape[0]

    #Create the policy net
    policy_net = PolicyNet(s_dim, a_dim)
    print(policy_net)

    #Load the models
    save_dir = './save'
    model_path = os.path.join(save_dir, "model.pt")

    if os.path.exists(model_path):
        print("Loading the model ... ", end="")
        checkpoint = torch.load(model_path)
        policy_net.load_state_dict(checkpoint["PolicyNet"])
        print("Done.")
    else:
        print('ERROR: No model saved')
        exit(1)

    #Run an episode using the policy net
    with torch.no_grad():
        state = env.reset()
        total_reward = 0
        length = 0

        while True:
            env.render()
            state_tensor = torch.tensor(np.expand_dims(state, axis=0), dtype=torch.float32, device='cpu')
            action = policy_net.choose_action(state_tensor, deterministic=True).cpu().numpy()
            state, reward, done, info = env.step(action[0])
            total_reward += reward
            length += 1

            if done:
                print("[Evaluation] Total reward = {:.6f}, length = {:d}".format(total_reward, length), flush=True)
                break

    env.close()
