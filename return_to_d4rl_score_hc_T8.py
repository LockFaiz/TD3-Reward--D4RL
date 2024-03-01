import gym
import d4rl
import numpy as np
import os 

hc_path = "./TD3_Spiking_hc_v3_T8"

# hopper_path = "./TD3_Spiking_Hopper_v3_T1"

# walker2d_path = './TD3_Spiking_Walker2d_v3_T1'

hc_datasets = ['halfcheetah-random-v0','halfcheetah-medium-v0','halfcheetah-expert-v0','halfcheetah-medium-replay-v0','halfcheetah-medium-expert-v0']
# hopper_datasets = ["hopper-random-v0","hopper-medium-v0","hopper-expert-v0","hopper-medium-expert-v0","hopper-medium-replay-v0"]
# walker2d_datasets = ["walker2d-random-v0","walker2d-medium-v0","walker2d-expert-v0","walker2d-medium-expert-v0","walker2d-medium-replay-v0"]

def eval_policy(env_name, seed, path, seed_offset=100):
    eval_env = gym.make(env_name)
    eval_env.seed(seed + seed_offset)
    file_name = os.listdir(path)
    scores = []
    for file in file_name:
        file_split = file.split("_")
        if file_split[2] == f'{seed}.npy':
            reward = np.load(os.path.join(path,file))
            for each in reward:
                d4rl_score = eval_env.get_normalized_score(each) * 100
                print(f'[{file}]    reward:{each}-->d4rl_score:{d4rl_score}')
                scores.append(d4rl_score)
            print("---------------------------------------")
            # print(f"Evaluation over {eval_episodes} episodes: {avg_reward:.3f}, D4RL score: {d4rl_score:.3f}")
            print(f'[{env_name}]   reward length:{len(reward)}   d4rl_score_length:{len(scores)}')
            print("---------------------------------------")            
        else:
            continue

    return scores

for hc_dataset_name in hc_datasets:
    for i in range(5):
        hc_score = eval_policy(hc_dataset_name, i, hc_path)
        if not os.path.exists(f"./hc_conversion_results_T8/{hc_dataset_name}"):  
            os.makedirs(f"./hc_conversion_results_T8/{hc_dataset_name}")
        score_file_name = f'TD3_HalfCheetah-v3_{i}'
        np.save(f"./hc_conversion_results_T8/{hc_dataset_name}/{score_file_name}", hc_score)

# for hopper_dataset_name in hopper_datasets:
#     for i in range(5):
#         hopper_score = eval_policy(hopper_dataset_name, i, hopper_path)
#         if not os.path.exists(f"./hopper_conversion_results/{hopper_dataset_name}"):  
#             os.makedirs(f"./hopper_conversion_results/{hopper_dataset_name}")
#         score_file_name = f'TD3_Hopper-v3_{i}'
#         np.save(f"./hopper_conversion_results/{hopper_dataset_name}/{score_file_name}", hopper_score)

# for walker2d_dataset_name in walker2d_datasets:
#     for i in range(5):
#         walker2d_score = eval_policy(walker2d_dataset_name, i, walker2d_path)
#         if not os.path.exists(f"./walker2d_conversion_results/{walker2d_dataset_name}"):  
#             os.makedirs(f"./walker2d_conversion_results/{walker2d_dataset_name}")
#         score_file_name = f'TD3_Walker2d-v3_{i}'
#         np.save(f"./walker2d_conversion_results/{walker2d_dataset_name}/{score_file_name}", walker2d_score)
