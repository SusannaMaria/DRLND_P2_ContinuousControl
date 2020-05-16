from collections import deque
import matplotlib.pyplot as plt
import numpy as np
import time
import torch
import pandas as pd
from unityagents import UnityEnvironment
from types import SimpleNamespace
import configparser
from actor_critic_ctl import actor_critic_train, actor_critic_test

from td3_agent import AgentTD3
from ddpg_agent import AgentDDPG

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def plot_minmax(df):
    """Print min max plot of DQN Agent analytics

    Params
    ======
        df :    Dataframe with scores
    """
    ax = df.plot(x='episode', y='mean')
    plt.fill_between(x='episode', y1='min', y2='max',
                     color='lightgrey', data=df)
    x_coordinates = [0, 150]
    y_coordinates = [30, 30]
    plt.plot(x_coordinates, y_coordinates, color='red')
    plt.show()


def ddpg_test(agent, ckpt, n_episodes=100):
    if torch.cuda.is_available():
        agent.actor_local.load_state_dict(torch.load('actor_ckpt.pth'))
        agent.critic_local.load_state_dict(
            torch.load('critic_ckpt.pth'))
    else:
        agent.actor_local.load_state_dict(torch.load(
            'trained/td3_{}_actor_ckpt.pth'.format(ckpt), map_location=lambda storage, loc: storage))
        agent.critic_local.load_state_dict(torch.load(
            'trained/td3_{}_critic_ckpt.pth'.format(ckpt), map_location=lambda storage, loc: storage))

    df = pd.DataFrame(columns=['episode', 'duration',
                               'min', 'max', 'std', 'mean'])

    # list of mean scores from each episode
    mean_scores = []
    # list of lowest scores from each episode
    min_scores = []
    # list of highest scores from each episode
    max_scores = []
    durations = []

    for i_episode in range(1, n_episodes+1):
        env_info = env.reset(train_mode=True)[
            brain_name]     # reset the environment
        # get the current state (for each agent)
        states = env_info.vector_observations
        # initialize the score (for each agent)
        scores = np.zeros(num_agents)
        start_time = time.time()
        while True:
            # select an action
            actions = agent.act(states, add_noise=False)
            # send all actions to tne environment
            env_info = env.step(actions)[brain_name]
            # get next state (for each agent)
            next_states = env_info.vector_observations
            # get reward (for each agent)
            rewards = env_info.rewards
            dones = env_info.local_done                        # see if episode finished
            # update the score (for each agent)
            scores += rewards
            # roll over states to next time step
            states = next_states
            if np.any(dones):                                  # exit loop if episode finished
                break
        duration = time.time() - start_time
        
        df.loc[i_episode-1] = [i_episode] + list([round(duration), np.min(scores),
                                                  np.max(scores),
                                                  np.std(scores),
                                                  np.mean(scores)])

        print('\rEpisode {} ({} sec)\tMean: {:.1f}'.format(
            i_episode, round(duration), np.mean(scores)))

    return df

env = UnityEnvironment(file_name='Crawler_Linux/Crawler.x86_64')
#env = UnityEnvironment(file_name='Reacher_Linux_20/Reacher.x86_64')

brain_name = env.brain_names[0]
brain = env.brains[brain_name]

# reset the environment
env_info = env.reset(train_mode=True)[brain_name]

# number of agents
num_agents = len(env_info.agents)
print('Number of agents:', num_agents)

# size of each action
action_size = brain.vector_action_space_size
print('Size of each action:', action_size)

# examine the state space
states = env_info.vector_observations
state_size = states.shape[1]
print('There are {} agents. Each observes a state with length: {}'.format(
    states.shape[0], state_size))
print('The state for the first agent looks like:', states[0])
config = configparser.ConfigParser()
config.read('config.ini')
agent_cfg_td3 = config['td3']


agent = AgentTD3(state_size=state_size, action_size=action_size,
                  random_seed=1, cfg_path="config.ini")

filename = 'test_td3.hdf5'
store = pd.HDFStore(filename)

for it in range(100, 14600, 100):
    print("dataset_{:03}".format(it))
    df = actor_critic_test(env, agent, agent_cfg_td3, it, 100)
    store.put('dataset_{:03}'.format(it), df)
    md = {'trained_episodes': it}
    store.get_storer('dataset_{:03}'.format(it)).attrs.metadata = md
store.close()


# print('Total score (averaged over agents) for 100 episodes: {}'.format(np.mean(scores)))