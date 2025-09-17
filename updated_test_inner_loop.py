import numpy as np
import torch
from maml_rl.baseline import LinearFeatureBaseline
from maml_rl.policies.categorical_mlp import CategoricalMLPPolicy
import updated_sampler_inner_loop as S
from updated_sampler_inner_loop import MissionEncoder,MultiTaskSampler, preprocess_obs, BabyAIMissionTaskWrapper
from maml_rl.metalearners.maml_trpo import MAMLTRPO
from sklearn.feature_extraction.text import CountVectorizer
from environment import (LOCAL_MISSIONS, 
                         DOOR_MISSIONS, 
                         OPEN_DOOR_MISSIONS, 
                         PICKUP_MISSIONS, 
                         OPEN_DOOR_ALL_MISSIONS,
                         OPEN_TWO_DOORS_MISSIONS,
                         OPEN_DOORS_ORDER_MISSIONS,
                         ACTION_OBJ_DOOR_MISSIONS)
from environment import (GoToLocalMissionEnv, 
                         GoToOpenMissionEnv, 
                         GoToObjDoorMissionEnv,
                         PickupDistMissionEnv,
                         OpenDoorMissionEnv, 
                         OpenDoorLocMissionEnv,
                         OpenTwoDoorsMissionEnv,
                         OpenDoorsOrderMissionEnv,
                         ActionObjDoorMissionEnv)
import time
import gc

start_time = time.time()


# ... import your policy, baseline, meta-learner (MAMLTRPO), etc.
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
# print("Using device:", device)

# Create BabyAI env
room_size=7
num_dists=3
max_steps=300
num_rows=2
num_cols=2


vectorizer = CountVectorizer()


# GoToLocal
base_env = GoToLocalMissionEnv(room_size=room_size, num_dists=num_dists, max_steps=max_steps)
missions = LOCAL_MISSIONS
vectorizer.fit(missions)
env = BabyAIMissionTaskWrapper(base_env, missions=missions)
model = "GoToLocal"
print(f"room_size: {room_size} \nnum_dists: {num_dists} \nmax_steps: {max_steps} \n")


# # PickupDist
# base_env = PickupDistMissionEnv(room_size=room_size, num_dists=num_dists, max_steps=max_steps)
# missions = PICKUP_MISSIONS
# vectorizer.fit(missions)
# env = BabyAIMissionTaskWrapper(base_env, missions=missions)
# model = "PickupDist"
# print(f"room_size: {room_size} \nnum_dists: {num_dists} \nmax_steps: {max_steps} \n")



# # GoToObjDoor
# base_env = GoToObjDoorMissionEnv(max_steps=max_steps, num_distractors=num_dists)
# missions=LOCAL_MISSIONS + DOOR_MISSIONS
# CountVectorizer(ngram_range=(1, 2), lowercase=True)
# vectorizer.fit(missions)
# env = BabyAIMissionTaskWrapper(base_env, missions=missions)
# model = "GoToObjDoor"
# print(f"num_dists: {num_dists} \nmax_steps: {max_steps} \n")



# # GoToOpen
# base_env = GoToOpenMissionEnv(room_size=room_size, num_rows=num_rows, num_cols=num_cols, num_dists=num_dists, max_steps=max_steps)
# missions=LOCAL_MISSIONS
# CountVectorizer(ngram_range=(1, 2), lowercase=True)
# vectorizer.fit(missions)
# env = BabyAIMissionTaskWrapper(base_env, missions=missions)
# model = "GoToOpen"
# print(f"room_size: {room_size} \nnum_dists: {num_dists} \nmax_steps: {max_steps} \nnum_rows: {num_rows} \nnum_cols: {num_cols}")



# # OpenDoor
# base_env = OpenDoorMissionEnv(room_size=room_size, max_steps=max_steps)
# missions = OPEN_DOOR_MISSIONS
# vectorizer.fit(missions)
# env = BabyAIMissionTaskWrapper(base_env, missions=missions)
# model = "OpenDoor"
# print(f"room_size: {room_size}  \nmax_steps: {max_steps} \n")




# # OpenDoorLocation
# base_env = OpenDoorLocMissionEnv(room_size=room_size, max_steps=max_steps)
# missions = OPEN_DOOR_ALL_MISSIONS
# CountVectorizer(ngram_range=(1, 2), lowercase=True)
# vectorizer.fit(missions)
# env = BabyAIMissionTaskWrapper(base_env, missions=missions)
# model = "OpenDoorLoc"
# print(f"room_size: {room_size}  \nmax_steps: {max_steps} \n")




# # # OpenTwoDoors
# base_env = OpenTwoDoorsMissionEnv(room_size=room_size, max_steps=None)
# missions = OPEN_TWO_DOORS_MISSIONS
# vectorizer.fit(missions)
# env = BabyAIMissionTaskWrapper(base_env, missions=missions)
# model = "OpenTwoDoors"
# print(f"room_size: {room_size}  \n")
#     #   max_steps: {max_steps} \n")




# # OpenDoorsOrder
# base_env = OpenDoorsOrderMissionEnv(room_size=room_size)
# missions = OPEN_DOORS_ORDER_MISSIONS
# CountVectorizer(ngram_range=(1, 2), lowercase=True)
# vectorizer.fit(missions)
# env = BabyAIMissionTaskWrapper(base_env, missions=missions)
# meta_batch_size = 25
# model = "OpenDoorsOrder"
# print(f"room_size: {room_size}")
#     # \  \nmax_steps: {max_steps} \n")




# # ActionObjDoor
# base_env = ActionObjDoorMissionEnv(objects = None, door_colors=None, obj_colors=None)
# missions = ACTION_OBJ_DOOR_MISSIONS
# CountVectorizer(ngram_range=(1, 2), lowercase=True)
# vectorizer.fit(missions)
# env = BabyAIMissionTaskWrapper(base_env, missions=missions)
# model = "ActionObjDoor"
# meta_batch_size = 20
# print("General setup for ActionObjDoor")
# # print(f"room_size: {room_size}  \nmax_steps: {max_steps} \n num_distractors: {num_dists} \n")




# # PutNextLocal
# base_env = PutNextLocalMissionEnv(room_size=room_size, max_steps=max_steps, num_dists=None)
# missions = PUTNEXT_MISSIONS
# vectorizer = CountVectorizer(ngram_range=(1, 2), lowercase=True)
# vectorizer.fit(missions)
# env = BabyAIMissionTaskWrapper(base_env, missions=missions)
# model = "PutNextLocal"
# print(f"room_size: {room_size}  \nmax_steps: {max_steps} \n")




print("Using environment:", base_env)

# 2. Policy/baseline setup (replace with your actual setup)
hidden_sizes = (64, 64)
nonlinearity = torch.nn.functional.tanh


S.vectorizer = vectorizer

# Instantiate the encoder
mission_encoder_input_dim = len(vectorizer.get_feature_names_out())
mission_encoder_output_dim = 32
mission_encoder = MissionEncoder(mission_encoder_input_dim,  hidden_dim1=32, hidden_dim2=64, output_dim=mission_encoder_output_dim).to(device)
S.mission_encoder = mission_encoder

# Finding Policy Parameters shape
obs, _ = env.reset()
vec = preprocess_obs(obs)
input_size = vec.shape[0]
output_size = base_env.action_space.n

policy = CategoricalMLPPolicy(
    input_size=input_size,
    output_size=output_size,
    hidden_sizes=hidden_sizes,
    nonlinearity=nonlinearity,
).to(device)
policy.share_memory()
baseline = LinearFeatureBaseline(input_size).to(device)

# 3. Sampler setup
sampler = MultiTaskSampler(
    env=env,
    batch_size=30,         # Number of episodes per task
    policy=policy,
    baseline=baseline,
    seed=1,
    num_workers=0
)

# 4. Meta-learner setup
meta_learner = MAMLTRPO(policy=policy, fast_lr=1e-5, first_order=True, device=device)

# 5. Training loop
avg_steps_per_batch = []
meta_batch_size = globals().get("meta_batch_size") or min(12, len(env.missions))
num_batches = 100  # Number of meta-

tasks = sampler.sample_tasks(len(env.missions))
print(f"\nTotal {len(env.missions)} Tasks that can be sampled: {tasks}\n")

for batch in range(num_batches):
    print(f"Meta-batch {batch+1}/{num_batches}")
    train_episodes, valid_episodes, step_counts = sampler.sample(
        meta_batch_size=meta_batch_size,
        num_steps=2,
        fast_lr=1e-4,
        gamma=0.99,
        gae_lambda=1.0,
        device=device
    )
    avg_steps = np.mean(step_counts)
    avg_steps_per_batch.append(avg_steps)
    logs = meta_learner.step(train_episodes, valid_episodes)

    del train_episodes, valid_episodes, step_counts
    
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()

    print(f"Batch {batch+1}/{num_batches} done. Logs: {logs}")
    print(f"Average steps in batch {batch+1}: {avg_steps}")

end_time = time.time()

print(f"Execution time: {(end_time - start_time)/60} minutes")


# GoToLocal
# Save the trained meta-policy parameters
torch.save(policy.state_dict(), f"inner_loop_model/inner_loop_{model}_{room_size}_{num_dists}_{max_steps}.pth")
print(f"inner_loop-policy parameters saved to inner_loop_model/inner_loop_{model}_{room_size}_{num_dists}_{max_steps}.pth")

print("Meta-training for inner loop finished!")



# # Pickup
# # Save the trained meta-policy parameters
# torch.save(policy.state_dict(), f"inner_loop_model/inner_loop_{model}_{room_size}_{num_dists}_{max_steps}.pth")
# print(f"inner_loop-policy parameters saved to inner_loop_model/inner_loop_{model}_{room_size}_{num_dists}_{max_steps}.pth")

# print("Meta-training for inner loop finished!")



# # Go_To_Obj_Door
# # Save the trained meta-policy parameters
# torch.save(policy.state_dict(), f"inner_loop_model/inner_loop_{model}_{num_dists}_{max_steps}.pth")
# print(f"inner_loop-policy parameters saved to inner_loop_model/inner_loop_{model}_{num_dists}_{max_steps}.pth")

# print("Meta-training for inner loop finished!")



# Go_To_Open
# Save the trained meta-policy parameters
torch.save(policy.state_dict(), f"inner_loop_model/inner_loop_{model}_new_{room_size}_{num_dists}_{num_rows}x{num_cols}_{max_steps}.pth")
print(f"inner_loop-policy parameters saved to inner_loop_model/inner_loop_GoToOpen_{room_size}_{num_dists}_{num_rows}x{num_cols}_{max_steps}.pth")

print("Meta-training for inner loop finished!")



# # Open Door
# # Save the trained meta-policy parameters
# torch.save(policy.state_dict(), f"inner_loop_model/inner_loop_{model}_{room_size}_{max_steps}.pth")
# print(f"inner_loop-policy parameters saved to inner_loop_model/inner_loop_{model}_{room_size}_{max_steps}.pth")

# print("Meta-training for inner loop finished!")



# # Open Doors 
# # Save the trained meta-policy parameters
# torch.save(policy.state_dict(), f"inner_loop_model/inner_loop_{model}_{room_size}_{max_steps}.pth")
# print(f"inner_loop-policy parameters saved to inner_loop_model/inner_loop_{model}_{room_size}_{max_steps}.pth")

# print("Meta-training for inner loop finished!")




# # Open Doors Order
# # Save the trained meta-policy parameters
# torch.save(policy.state_dict(), f"inner_loop_model/inner_loop_{model}_{room_size}.pth")
# print(f"inner_loop-policy parameters saved to inner_loop_model/inner_loop_{model}_{room_size}.pth")

# print("Meta-training for inner loop finished!")




# # ActionObjDoor 
# # Save the trained meta-policy parameters
# torch.save(policy.state_dict(), f"inner_loop_model/inner_loop_{model}.pth")
# print(f"inner_loop-policy parameters saved to inner_loop_model/inner_loop_{model}.pth")

# print("Meta-training for inner loop finished!")




import os, json, numpy as np

env_name = str(model) if "model" in globals() else "UnknownEnv"
env_dir = os.path.join("metrics", env_name)
os.makedirs(env_dir, exist_ok=True)

np.save(os.path.join(env_dir, "inner_avg_steps.npy"), np.array(avg_steps_per_batch))
with open(os.path.join(env_dir, "inner_meta.json"), "w") as f:
    json.dump({"label": "MAML (Inner Loop)", "env": env_name}, f)

# # After training, plot
# import matplotlib.pyplot as plt     
# plt.plot(avg_steps_per_batch)
# plt.xlabel("Meta-inner_loop-batch")
# plt.ylabel("Average steps to goal")
# plt.title(f"inner-loop_{model}_{room_size}_{num_dists}")
# plt.show()
