from env.env import  Env
from RL.pcl_agent.rainbow import PCL_rainbow
import os
import numpy as np
from tqdm import trange



def init(agrs,env, agent,config):

    if args.train:
        agent.train()
        if type(agent)==PCL_rainbow:
            timeout=False
            mem = ReplayMemory(args, args.num_replay_memory)
            priority_weight_increase = (1 - args.priority_weight) / (args.T_max - args.learn_start)
            results_dir = os.path.join('results', args.id)
            if not os.path.exists(results_dir):
                os.makedirs(results_dir)
            T, done = 0, False
            sum_reward=0
            state, _ = env.reset(h_level=False)
            for T in trange(1, int(args.num_steps)):
                if done or timeout:

                    print(sum_reward,'  ------  Litter:', np.sum(env.reward_map_bel))#,(sum_reward+(0.35*done))/(env.start_entr_map))
                    sum_reward=0
                    state, _ = env.reset(h_level=False)


                    litter=env.litter
                    print(sum_reward,'------', litter)#,(sum_reward+(0.35*done))/(env.start_entr_map))
                    sum_reward=0
                    state, _ = env.reset(h_level=False)
                    episode+=1
                    writer.add_scalar("Litter", litter, episode)

                action = agent.epsilon_greedy(T,3000000, state, all_actions)


                agent.reset_noise()  # Draw a new set of noisy weights


                next_state, reward, done, actions, sim_i, timeout = env.step(action, h_level=False, agent="rainbow")  # Step
                sum_reward=sum_reward+reward
                 # Append transition to memory

                # Train and test
                if sim_i>0:
                    for j in range(sim_i-1):
                        mem.append(state, actions[j], 0, True)
                mem.append(state, actions[sim_i], reward, done) 
                if T >= 100000:#args.learn_start:
                    mem.priority_weight = min(mem.priority_weight + priority_weight_increase, 1)  # Anneal importance sampling weight β to 1

                    agent.learn(mem,T,writer)  # Train with n-step distributional double-Q learning

                    if T % args.target_update == 0:
                        agent.update_target_net()

                    # Checkpoint the network
                    if (args.checkpoint_interval != 0) and (T % args.checkpoint_interval == 0):
                        agent.save(results_dir, 'checkpoint.pth')

                state = next_state
