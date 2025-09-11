import numpy as np
# import os
import torch
# import torch.multiprocessing as mp
import matplotlib.pyplot as plt 
import pickle
from maml_rl.baseline import LinearFeatureBaseline
from maml_rl.metalearners.maml_trpo3 import MAMLTRPO
from maml_rl.policies.categorical_mlp import CategoricalMLPPolicy
from sklearn.feature_extraction.text import CountVectorizer
import updated_sampler_lang as S
from updated_sampler_lang import (BabyAIMissionTaskWrapper, 
                          MissionEncoder, 
                          MissionParamAdapter, 
                          MultiTaskSampler, 
                          preprocess_obs)
from environment import (LOCAL_MISSIONS, 
                         DOOR_MISSIONS, 
                         OPEN_DOOR_MISSIONS, 
                         SEQ_MISSIONS, 
                         PICKUP_MISSIONS, 
                         OPEN_DOOR_ALL_MISSIONS,
                         OPEN_TWO_DOORS_MISSIONS,
                         OPEN_DOORS_ORDER_MISSIONS,
                         ACTION_OBJ_DOOR_MISSIONS,
                         PUTNEXT_MISSIONS)
from environment import (GoToLocalMissionEnv, 
                            GoToOpenMissionEnv, 
                            GoToSeqMissionEnv, 
                            GoToObjDoorMissionEnv, 
                            # OpenMissionEnv, 
                            PickupDistMissionEnv,
                            OpenDoorMissionEnv,
                            OpenDoorLocMissionEnv,
                            OpenTwoDoorsMissionEnv,
                            OpenDoorsOrderMissionEnv,
                            ActionObjDoorMissionEnv,
                            PutNextLocalMissionEnv)
import time
# os.environ.setdefault("OMP_NUM_THREADS", "1")
# os.environ.setdefault("MKL_NUM_THREADS", "1")
# try:
#     mp.set_start_method("spawn", force=True)
# except RuntimeError:
#     pass

start_time = time.time()


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


print("ONLY POLICY IS TRAINED")
# Create BabyAI env
room_size=5
num_dists=2
max_steps=25
num_rows=2
num_cols=2


vectorizer = CountVectorizer()

# GoToLocal
base_env = GoToLocalMissionEnv(room_size=room_size, num_dists=num_dists, max_steps=max_steps)
missions = LOCAL_MISSIONS
CountVectorizer(ngram_range=(1, 2), lowercase=True)
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
# base_env = OpenTwoDoorsMissionEnv(room_size=room_size, max_steps=max_steps)
# missions = OPEN_TWO_DOORS_MISSIONS
# vectorizer.fit(missions)
# env = BabyAIMissionTaskWrapper(base_env, missions=missions)
# model = "OpenTwoDoors"
# print(f"room_size: {room_size}  \nmax_steps: {max_steps} \n")




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
# base_env = ActionObjDoorMissionEnv()
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



# # Open
# base_env = OpenMissionEnv(room_size=room_size,num_rows=num_rows, num_cols=num_cols, num_dists=num_dists,max_steps=max_steps)
# missions = OPEN_DOOR_MISSIONS
# vectorizer.fit(missions)
# env = BabyAIMissionTaskWrapper(base_env, missions=missions)
# print(f"room_size: {room_size} \nnum_dists: {num_dists} \nmax_steps: {max_steps} \nnum_rows: {num_rows} \nnum_cols: {num_cols}")



# # GoToSeq
# base_env = GoToSeqMissionEnv(room_size=room_size, num_rows=num_rows, num_cols=num_cols, num_dists=num_dists, max_steps=max_steps)
# env = BabyAIMissionTaskWrapper(base_env, missions=SEQ_MISSIONS)


print("Using environment:", base_env)

# Policy/baseline setup (replace with your actual setup)
hidden_sizes = (64, 64)
nonlinearity = torch.nn.functional.tanh

# Instantiate the encoder (make sure to move to correct device later)
S.vectorizer = vectorizer
mission_encoder_input_dim = len(S.vectorizer.get_feature_names_out())
mission_encoder_output_dim = 32  
mission_encoder = MissionEncoder(mission_encoder_input_dim,  hidden_dim1=32, hidden_dim2=64, output_dim=mission_encoder_output_dim).to(device)
S.mission_encoder = mission_encoder  

dummy_obs, _ = env.reset()
dummy_vec = preprocess_obs(dummy_obs)
input_size = dummy_vec.shape[0]
output_size = base_env.action_space.n


policy = CategoricalMLPPolicy(
    input_size=input_size,
    output_size=output_size,
    hidden_sizes=hidden_sizes,
    nonlinearity=nonlinearity,
).to(device)
policy.share_memory()
baseline = LinearFeatureBaseline(input_size).to(device)

policy_param_shapes = [p.shape for p in policy.parameters()]

mission_adapter = MissionParamAdapter(mission_encoder_output_dim, policy_param_shapes).to(device)

# Sampler setup
sampler = MultiTaskSampler(
    env=env,
    batch_size=3,         # Number of episodes per task
    policy=policy,
    baseline=baseline,
    seed=1,
    num_workers=0
)

# 4. Meta-learner setup
meta_learner = MAMLTRPO(
    policy=policy,
    mission_encoder=mission_encoder,
    mission_adapter=mission_adapter,
    vectorizer=vectorizer,
    fast_lr=1e-4,
    first_order=True,
    device=device
)


def print_param_stats(module, name):
    for pname, param in module.named_parameters():
        print(f"{name}.{pname}: mean={param.data.mean():.6f}, std={param.data.std():.6f}, requires_grad={param.requires_grad}")

# 5. Training loop
avg_steps_per_batch = []
meta_batch_size = globals().get("meta_batch_size") or min(12, len(env.missions))
num_batches = 5  # Number of meta-batches

tasks = sampler.sample_tasks(len(env.missions))
print(f"\nTotal {len(env.missions)} Tasks that can be sampled : {tasks}\n")

for batch in range(num_batches):
    print(f"\nBatch {batch + 1}")
    valid_episodes, step_counts = sampler.sample(
        meta_batch_size,
        meta_learner,
        num_steps=1,
        fast_lr=1e-4,
        gamma=0.99,
        gae_lambda=1.0,
        device=device
    )
                
    avg_steps = np.mean(step_counts) if len(step_counts) > 0 else float('nan')
    avg_steps_per_batch.append(avg_steps)
    print(f"Average steps in Meta-batch {batch+1}: {avg_steps}\n")

    # print("=== BEFORE optimizer ===")
    # print_param_stats(policy, "policy")
    # print()
    # print_param_stats(mission_adapter, "mission_adapter")
    # print()
    # print_param_stats(mission_encoder, "mission_encoder")
    # print()

    logs = meta_learner.step(valid_episodes,valid_episodes)


end_time = time.time()

print(f"Execution time: {(end_time - start_time)/60}minutes")

# Save the trained meta-policy parameters

# # GoToLocal
# # PickupDist
# torch.save({
#     "policy": policy.state_dict(),
#     "mission_encoder": mission_encoder.state_dict(),
#     "mission_adapter": mission_adapter.state_dict()
# # }, f"ablation_study/lang_policy_unadapted_{model}_{room_size}_{num_dists}.pth")
# }, f"ablation_study/lang_policy_unadapted_{model}_{room_size}_{num_dists}_{max_steps}.pth")

# # Save the vectorizer
# # with open(f"ablation_study/vectorizer_lang_unadapted_{model}_{room_size}_{num_dists}_{max_steps}.pkl", "wb") as f:
# with open(f"ablation_study/vectorizer_lang_unadapted_{model}_{room_size}_{num_dists}_{max_steps}.pkl", "wb") as f:
#     pickle.dump(vectorizer, f)

# print(f"lang-ablation_study parameters saved to ablation_study/lang_policy_{model}.pth")

# print("ablation_study policy for training finished!")



# GoToObjDoor  
torch.save({
    "policy": policy.state_dict(),
    "mission_encoder": mission_encoder.state_dict(),
    "mission_adapter": mission_adapter.state_dict()
}, f"ablation_study/lang_policy_unadapted_{model}_{num_dists}_{max_steps}.pth")

# Save the vectorizer
with open(f"ablation_study/vectorizer_lang_unadapted_{model}_{num_dists}_{max_steps}.pkl", "wb") as f:
    pickle.dump(vectorizer, f)

print(f"lang-policy parameters saved to lang_model/lang_policy_{model}_{num_dists}_{max_steps}.pth")
print("lang_based policy for training Go To ObjDoor finished!")




# # Acton Obj Door
# torch.save({
#     "policy": policy.state_dict(),
#     "mission_encoder": mission_encoder.state_dict(),
#     "mission_adapter": mission_adapter.state_dict()
# }, f"ablation_study/lang_policy_unadapted_{model}.pth")

# # Save the vectorizer
# with open(f"ablation_study/vectorizer_lang_unadapted_{model}.pkl", "wb") as f:
#     pickle.dump(vectorizer, f)

# print(f"lang-policy parameters saved to ablation_study/lang_policy_unadapted_{model}.pth")
# print("lang_based policy for training Action object Door finished!")




# # GoToOpen
# torch.save({
#     "policy": policy.state_dict(),
#     "mission_encoder": mission_encoder.state_dict(),
#     "mission_adapter": mission_adapter.state_dict()
# }, f"ablation_study/lang_policy_unadapted_{model}_{room_size}_{num_dists}_{num_rows}x{num_cols}_{max_steps}.pth")

# # Save the vectorizer
# with open(f"ablation_study/vectorizer_lang_unadapted_{model}_{room_size}_{num_dists}_{num_rows}x{num_cols}_{max_steps}.pkl", "wb") as f:
#     pickle.dump(vectorizer, f)

# print("lang-policy parameters saved to ablation_study/lang_policy_unadapted_GoToOpen.pth")
# print("lang_based policy for training Go To Open finished!")



# # Open Door
# # Open Door Location
# torch.save({
#     "policy": policy.state_dict(),
#     "mission_encoder": mission_encoder.state_dict(),
#     "mission_adapter": mission_adapter.state_dict()
# }, f"ablation_study/lang_policy_unadapted_{model}_{room_size}.pth")

# # Save the vectorizer
# with open(f"ablation_study/vectorizer_lang_unadapted_{model}_{room_size}.pkl", "wb") as f:
#     pickle.dump(vectorizer, f)

# print(f"lang-policy parameters saved to lang_model/lang_policy_{model}_{room_size}.pth")
# print("lang_based policy for training OpenDoor finished!")


# # Put Next Local
# torch.save({
#     "policy": policy.state_dict(),
#     "mission_encoder": mission_encoder.state_dict(),
#     "mission_adapter": mission_adapter.state_dict()
# # }, f"ablation_study/lang_policy_unadapted_{model}_{room_size}_{num_dists}.pth")
# }, f"ablation_study/lang_policy_unadapted_{model}_{room_size}_{max_steps}.pth")

# # Save the vectorizer
# # with open(f"ablation_study/vectorizer_lang_unadapted_{model}_{room_size}_{num_dists}.pkl", "wb") as f:
# with open(f"updated/vectorizer_lang_unadapted_{model}_{room_size}_{max_steps}.pkl", "wb") as f:
#     pickle.dump(vectorizer, f)

# print(f"lang-ablation_study parameters saved to ablation_study/lang_policy_{model}.pth")

# print("ablation_study policy for training finished!")




# # GoToSeq
# torch.save({
#     "policy": policy.state_dict(),
#     "mission_encoder": mission_encoder.state_dict(),
#     "mission_adapter": mission_adapter.state_dict()
# }, f"lang_model/lang_policy_GoToSeq_{room_size}_{num_dists}_{num_rows}x{num_cols}.pth")

# # Save the vectorizer
# with open(f"lang_model/vectorizer_lang_GoToSeq_{room_size}_{num_dists}_{num_rows}x{num_cols}.pkl", "wb") as f:
#     pickle.dump(vectorizer, f)

# print("lang-policy parameters saved to lang_model/lang_policy_GoToSeq.pth")
# print("lang_based policy for training Go To Seq finished!")







# After training, plot    
plt.plot(avg_steps_per_batch)
plt.xlabel("Meta-batch")
plt.ylabel("Steps")
plt.title(f"Average Steps vs Meta-batch unadapted language {model}")
plt.show()