import os
import torch
import argparse
import configparser
import run 
from env3d.agent.transition import Transition
from env3d.env import  Env

def parse():
    parser = argparse.ArgumentParser(description="MLDS&ADL HW3")
    parser.add_argument('--networkPath', default='network/', help='folder to put results of experiment in')
    parser.add_argument('--train_dqn', action='store_true', help='whether train DQN')
    parser.add_argument('--load_net', action='store_true', help='whether test DQN')
    parser.add_argument('--video_dir', default=None, help='output video directory')
    # Environment
    parser.add_argument('--do_render', action='store_true', help='whether render environment')
    parser.add_argument('--train', action='store_true', help='train the agent in the terminal')
    parser.add_argument('--episode_length', type=int, default=300, help='length of mapping environment episodes')
    #Agent
    parser.add_argument('--rainbow', action='store_true', help='off policy agent rainbow')
    parser.add_argument('--atoms', type=int, default=51, metavar='C', help='Discretised size of value distribution')
    parser.add_argument('--adam-eps', type=float, default=1.5e-4, metavar='ε', help='Adam epsilon')
    parser.add_argument('--discount', type=float, default=0.99, metavar='γ', help='Discount factor')
    parser.add_argument('--priority_weight', type=float, default=0.2, metavar='beta', help='priority weight beta')
    parser.add_argument('--T-max', type=int, default=int(50e6), metavar='STEPS', help='Number of training steps (4x number of frames)')
    parser.add_argument('--learn-start', type=int, default=int(20e3), metavar='STEPS', help='Number of steps before starting training')
    parser.add_argument('--evaluation-interval', type=int, default=12000, metavar='STEPS', help='Number of training steps between evaluations')
    parser.add_argument('--target-update', type=int, default=int(8e3), metavar='τ', help='Number of steps after which to update target network')
    parser.add_argument('--id', type=str, default='3Dexperiment', help='Experiment ID')
    parser.add_argument('--model_path', type=str, default = "results/3Dexperiment/checkpoint.pth", help='model used during testing / visulization') #testmoreFilters.h5
    parser.add_argument('--exp_name', type=str, default = "", help='')

    try:
        from argument import add_arguments
        parser = add_arguments(parser)
    except:
        pass
    args = parser.parse_args()
    if torch.cuda.is_available():
        args.device = torch.device('cuda')
        torch.cuda.manual_seed(np.random.randint(1, 10000))
        torch.backends.cudnn.enabled = True#args.enable_cudnn
    else:
        args.device = torch.device('cpu')
    return args


if __name__ == '__main__':
    import numpy as np
    import sys
    np.set_printoptions(threshold=sys.maxsize)
    args = parse()
    results_dir = os.path.join('results', args.exp_name)
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    os.makedirs(args.networkPath, exist_ok=True)
    config = configparser.ConfigParser()
    config.read('config.ini')
    metrics = {'steps': [], 'rewards': [], 'entropy': []}

    env = Env(args, config)
    if args.rainbow:
        from Agents.pcl_rainbow.agent import PCL_rainbow
        agent = PCL_rainbow(args, env)
    init(args, env, agent, config)
