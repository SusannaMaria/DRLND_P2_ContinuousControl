from unityagents import UnityEnvironment
import numpy as np

env = UnityEnvironment(file_name='Crawler_Linux/Crawler.x86_64')

# get the default brain
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


env_info = env.reset(train_mode=False)[brain_name]     # reset the environment
# get the current state (for each agent)
states = env_info.vector_observations
# initialize the score (for each agent)
scores = np.zeros(num_agents)
while True:
    # select an action (for each agent)
    actions = np.random.randn(num_agents, action_size)
    # all actions between -1 and 1
    actions = np.clip(actions, -1, 1)
    # send all actions to tne environment
    env_info = env.step(actions)[brain_name]
    # get next state (for each agent)
    next_states = env_info.vector_observations
    # get reward (for each agent)
    rewards = env_info.rewards
    dones = env_info.local_done                        # see if episode finished
    # update the score (for each agent)
    scores += env_info.rewards
    # roll over states to next time step
    states = next_states
    if np.any(dones):                                  # exit loop if episode finished
        break
print('Total score (averaged over agents) this episode: {}'.format(np.mean(scores)))


env.close()
