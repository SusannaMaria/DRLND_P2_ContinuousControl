import torch
from torch import nn
from torchviz import make_dot, make_dot_from_trace
from graphviz import Digraph

from ddpg_agent import AgentDDPG
from td3_agent import AgentTD3


x = torch.randn(5,33).cuda() 
y = torch.randn(5,4).cuda() 

agent = AgentDDPG(state_size=33, action_size=4,
                  random_seed=1, cfg_path="config.ini")

dot = make_dot(agent.critic_local(x,y), params=dict(agent.critic_local.named_parameters()))
dot.format = 'png'
dot.render("static/ddpg_critic_model")

dot = make_dot(agent.actor_local(x), params=dict(agent.actor_local.named_parameters()))
dot.format = 'png'
dot.render("static/ddpg_actor_model")


agent = AgentTD3(state_size=33, action_size=4,
                  random_seed=1, cfg_path="config.ini")

dot = make_dot(agent.critic_local(x,y), params=dict(agent.critic_local.named_parameters()))
dot.format = 'png'
dot.render("static/td3_critic_model")

dot = make_dot(agent.actor_local(x), params=dict(agent.actor_local.named_parameters()))
dot.format = 'png'
dot.render("static/td3_actor_model")

