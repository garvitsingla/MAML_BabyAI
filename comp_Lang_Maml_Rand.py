import torch
import numpy as np
import random
from environment import (GoToLocalMissionEnv, 
                         GoToOpenMissionEnv, 
                         GoToSeqMissionEnv, 
                         GoToObjDoorMissionEnv,  
                         PickupDistMissionEnv,
                         OpenDoorMissionEnv, 
                         OpenDoorLocMissionEnv,
                         OpenTwoDoorsMissionEnv,
                         OpenDoorsOrderMissionEnv,
                         ActionObjDoorMissionEnv)
from sampler_lang import BabyAIMissionTaskWrapper, MissionEncoder, MissionParamAdapter
import sampler_lang as SL
from maml_rl.policies.categorical_mlp import CategoricalMLPPolicy
import pickle
import time
from maml_rl.utils.reinforcement_learning import reinforce_loss
from maml_rl.episode import BatchEpisodes
from maml_rl.baseline import LinearFeatureBaseline

start_time = time.time()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

OBJECTS = ['box']
COLORS = ['red', 'green', 'blue', 'purple','yellow', 'grey']
PREP_LOCS = ['on', 'at', 'to']

# Location names
LOC_NAMES = ['right', 'front']

DOOR_COLORS = ['yellow', 'grey']


# For Pickup
PICKUP_MISSIONS = [f"pick up the {color} {obj}" for color in COLORS for obj in OBJECTS]

# For GoToLocal
LOCAL_MISSIONS = [f"go to the {color} {obj}" for color in COLORS for obj in OBJECTS]

# For environments that include doors (GoToObjDoor, GoToOpen, Open)
DOOR_MISSIONS = [f"go to the {color} door" for color in DOOR_COLORS]
OPEN_DOOR_MISSIONS = [f"open the {color} door" for color in DOOR_COLORS]
DOOR_LOC_MISSIONS = [f"open the door {prep} the {loc}" for prep in PREP_LOCS for loc in LOC_NAMES]
OPEN_TWO_DOORS_MISSIONS = [f"open the {c1} door, then open the {c2} door" for c1 in DOOR_COLORS for c2 in DOOR_COLORS]
OPEN_DOORS_ORDER_MISSIONS = (
    [f"open the {c1} door" for c1 in DOOR_COLORS] +
    [f"open the {c1} door, then open the {c2} door" for c1 in DOOR_COLORS for c2 in DOOR_COLORS] +
    [f"open the {c1} door after you open the {c2} door" for c1 in DOOR_COLORS for c2 in DOOR_COLORS]
)

ACTION_OBJ_DOOR_MISSIONS = (
    [f"pick up the {c} box" for c in COLORS] +
    [f"go to the {c} box"   for c in COLORS] +
    [f"go to the {c} door"   for c in DOOR_COLORS] +
    [f"open a {c} door"      for c in DOOR_COLORS]
)



# Generate all valid 2-step missions (no repeats)
SEQ_MISSIONS = [
    f"{m1} and {m2}"
    for m1 in LOCAL_MISSIONS + DOOR_MISSIONS
    for m2 in LOCAL_MISSIONS + DOOR_MISSIONS
    if m1 != m2
]

# Open Door Missions including location-based
OPEN_DOOR_ALL_MISSIONS = OPEN_DOOR_MISSIONS + DOOR_LOC_MISSIONS

# (Optional) for general environments supporting both
ALL_MISSIONS = LOCAL_MISSIONS + DOOR_MISSIONS + OPEN_DOOR_MISSIONS + SEQ_MISSIONS

# Env setup to get dims
room_size=9
num_dists=20
max_steps=300
num_rows=2
num_cols=2

model = "PickupDist_7_1"  # Trained Model used for evaluation


# # GoToLocal
# base_env = GoToLocalMissionEnv(room_size=room_size, num_dists=num_dists, max_steps=max_steps)
# missions=LOCAL_MISSIONS
# env = BabyAIMissionTaskWrapper(base_env, missions=missions)
# print(f"room_size: {room_size}\n num_dists: {num_dists}\n max_steps: {max_steps}\n available missions: {LOCAL_MISSIONS}\n ")


# # Pickup
# base_env = PickupDistMissionEnv(room_size=room_size, num_dists=num_dists, max_steps=max_steps)
# missions=PICKUP_MISSIONS
# env = BabyAIMissionTaskWrapper(base_env, missions=missions)
# print(f"room_size: {room_size}\n num_dists: {num_dists}\n max_steps: {max_steps}\n available missions: {PICKUP_MISSIONS}\n ")



# # GoToObjDoor
# base_env = GoToObjDoorMissionEnv(max_steps=max_steps, num_distractors=num_dists)
# missions=LOCAL_MISSIONS + DOOR_MISSIONS
# env = BabyAIMissionTaskWrapper(base_env, missions=missions)
# print(f"num_dists: {num_dists}\n max_steps: {max_steps}\n")



# GoToOpen
base_env = GoToOpenMissionEnv(room_size=room_size, num_rows=num_rows, num_cols=num_cols, num_dists=num_dists, max_steps=max_steps)
missions=LOCAL_MISSIONS
env = BabyAIMissionTaskWrapper(base_env, missions=missions)
print(f"room_size: {room_size} \nnum_dists: {num_dists} \nmax_steps: {max_steps} \nnum_rows: {num_rows} \nnum_cols: {num_cols}")



# # OpenDoorMissionEnv
# base_env = OpenDoorMissionEnv(room_size=room_size, max_steps=max_steps)
# missions = OPEN_DOOR_MISSIONS
# env = BabyAIMissionTaskWrapper(base_env, missions=missions)
# print(f"room_size: {room_size}  \nmax_steps: {max_steps} \n")



# # OpenDoorLocMissionEnv
# base_env = OpenDoorLocMissionEnv(room_size=room_size, max_steps=max_steps)
# missions = OPEN_DOOR_ALL_MISSIONS
# env = BabyAIMissionTaskWrapper(base_env, missions=missions)
# print(f"room_size: {room_size}  \nmax_steps: {max_steps} \n")




# # OpenTwoDoors
# base_env = OpenTwoDoorsMissionEnv(room_size=room_size, max_steps=None)
# missions = OPEN_TWO_DOORS_MISSIONS
# env = BabyAIMissionTaskWrapper(base_env, missions=missions)
# print(f"room_size: {room_size}")
#         # \nmax_steps: {max_steps} \n")





# # OpenDoorsOrder
# base_env = OpenDoorsOrderMissionEnv(room_size=room_size)
# missions = OPEN_DOORS_ORDER_MISSIONS
# env = BabyAIMissionTaskWrapper(base_env, missions=missions)
# print(f"room_size: {room_size}\n")



# # ActionObjDoor
# base_env = ActionObjDoorMissionEnv()
# missions = ACTION_OBJ_DOOR_MISSIONS
# env = BabyAIMissionTaskWrapper(base_env, missions=missions)
# print("General setup for ActionObjDoor")
# # # print(f"room_size: {room_size}  \nmax_steps: {max_steps} \n num_distractors: {num_dists} \n")



# Open
# base_env = OpenMissionEnv(room_size=room_size,num_rows=num_rows, num_cols=num_cols, num_dists=num_dists, max_steps=max_steps)
# missions=OPEN_DOOR_MISSIONS
# env = BabyAIMissionTaskWrapper(base_env, missions=missions)
# print(f"room_size: {room_size}\n num_dists: {num_dists}\n max_steps: {max_steps}\n  num_rows: {num_rows}\n num_cols: {num_cols}\n model used: {model}")



# Environment 3
# base_env = GoToSeqMissionEnv(room_size=room_size, num_rows=num_rows, num_cols=num_cols, num_dists=num_dists, max_steps=max_steps)
# print(f"room_size: {room_size}\n num_dists: {num_dists}\n max_steps: {max_steps}\n  num_rows: {num_rows}\n num_cols: {num_cols}")



print(f"env name {base_env} \n model used: {model}\n")

# ----- Restore trained lang-adapted policy -----

ckpt = torch.load(f"lang_model/lang_policy_{model}.pth", map_location=device)
with open(f"lang_model/vectorizer_lang_{model}.pkl", "rb") as f:
    vectorizer = pickle.load(f)


SL.vectorizer = vectorizer  # Use the loaded vectorizer
# mission_encoder_input_dim = len(vectorizer.get_feature_names_out())
mission_encoder_output_dim = 32
SL.mission_encoder = MissionEncoder(len(SL.vectorizer.get_feature_names_out()), 32, 64, mission_encoder_output_dim).to(device)
SL.mission_encoder.load_state_dict(ckpt["mission_encoder"])
SL.mission_encoder.eval()
mission_encoder = SL.mission_encoder
preprocess_obs = SL.preprocess_obs

dummy_obs, _ = env.reset()
input_size = preprocess_obs(dummy_obs).shape[0]
output_size = base_env.action_space.n
hidden_sizes = (64, 64)
nonlinearity = torch.nn.functional.tanh

# --- Policy and adapters ---
policy_lang = CategoricalMLPPolicy(
    input_size=input_size,
    output_size=output_size,
    hidden_sizes=hidden_sizes,
    nonlinearity=nonlinearity,
).to(device)  
policy_lang.load_state_dict(ckpt["policy"])
policy_lang.eval()
policy_param_shapes = [p.shape for p in policy_lang.parameters()]

mission_adapter = MissionParamAdapter(mission_encoder_output_dim, policy_param_shapes).to(device)
mission_adapter.load_state_dict(ckpt["mission_adapter"])    
mission_adapter.eval()


policy_maml = CategoricalMLPPolicy(
        input_size=input_size,
        output_size=output_size,
        hidden_sizes=hidden_sizes,
        nonlinearity=nonlinearity,      
    ).to(device)
policy_maml.load_state_dict(torch.load(f"inner_loop_model/inner_loop_{model}.pth", map_location=device))
policy_maml.eval()


baseline = LinearFeatureBaseline(input_size).to(device)


def get_language_adapted_params(policy, mission_str, mission_encoder, mission_adapter, vectorizer, device):
    mission_vec = vectorizer.transform([mission_str]).toarray()[0]
    mission_tensor = torch.from_numpy(mission_vec.astype(np.float32)).unsqueeze(0).to(device)
    with torch.no_grad():
        mission_emb = mission_encoder(mission_tensor)
        mission_emb = mission_emb.to(device)
        delta_thetas = mission_adapter(mission_emb)
        delta_thetas = [torch.clamp(delta, -1.0, 1.0) * 0.01 for delta in delta_thetas]
    policy_params = list(policy.parameters())
    param_names = list(dict(policy.named_parameters()).keys())
    from collections import OrderedDict
    theta_prime = OrderedDict(
        (name, param + delta.squeeze(0))
        for name, param, delta in zip(param_names, policy_params, delta_thetas)
    )
    return theta_prime


def adapt_policy_for_task(task, policy, num_steps=1, fast_lr=0.5, batch_size=10,baseline=None):
    """Adapt the policy for a given task using MAML-style updates."""
    # Reset task
    env.reset_task(task)
    
    # Collect training episodes for task adaptation
    train_batches = []
    for _ in range(num_steps+1):
        # print(f"gradient step for task '{_+1}:{task}'")
        batch = BatchEpisodes(batch_size=batch_size, gamma=0.99, device=device)
        for ep in range(batch_size):
            # print(f"Collecting episode {ep+1}/{batch_size} for task '{task}'")
            obs, info = env.reset()
            done = False

            episode_obs = []
            episode_actions = []
            episode_rewards = []
            while not done:
                obs_vec = preprocess_obs(obs)
                obs_tensor = torch.from_numpy(obs_vec).float().unsqueeze(0).to(device)
                with torch.no_grad():
                    dist = policy(obs_tensor)
                    action = dist.sample().item()
                obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                episode_obs.append(obs_vec)
                episode_actions.append(np.array(action))
                episode_rewards.append(np.array(reward, dtype=np.float32))
            batch.append(episode_obs, episode_actions, episode_rewards, [ep]*len(episode_obs))
        train_batches.append(batch)

    # Compute advantages for the batch
    for batch in train_batches:
        batch.compute_advantages(baseline, gae_lambda=1.0, normalize=True)
    
    # Compute gradients and adapt policy parameters
    params = None
    for batch in train_batches:
        loss = reinforce_loss(policy, batch, params=params)
        params = policy.update_params(loss, params=params, step_size=fast_lr, first_order=True)
    return params



def evaluate_policy(env, policy, params=None, max_steps=max_steps, render=False):
    obs, _ = env.reset()
    steps = 0
    done = False
    while not done and steps < max_steps:
        if render:
            env.render("human")
        obs_vec = preprocess_obs(obs)
        obs_tensor = torch.from_numpy(obs_vec).float().unsqueeze(0).to(device)
        with torch.no_grad():
            if params is not None:
                dist = policy(obs_tensor, params=params)
            else:
                dist = policy(obs_tensor)
            action = dist.sample().item()
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        steps += 1
    return steps


# --- Evaluation ---
N_MISSIONS = 20
N_EPISODES = 40

results_lang = []
results_inner_loop = []
results_random = []

print("Comparing language-adapted policy and random policy on random missions:")
for i in range(N_MISSIONS):
    mission = random.choice(missions)
    print(f"\nMission {i+1}/{N_MISSIONS}: '{mission}'")

    # 1. Lang-adapted policy
    theta_prime = get_language_adapted_params(policy_lang, mission, mission_encoder, mission_adapter, vectorizer, device)
    lang_steps = []
    print("  [Lang-adapted policy episodes]")
    for ep in range(N_EPISODES):
        env.reset_task(mission)
        steps = evaluate_policy(env, policy_lang, params=theta_prime)
        lang_steps.append(steps)
    mean_lang = np.mean(lang_steps)
    std_lang = np.std(lang_steps)
    results_lang.append(mean_lang)
    print(f"    --> Avg steps: {mean_lang:.2f} ± {std_lang:.2f}")


    # 2. MAML policy
    maml_params = adapt_policy_for_task(mission, policy_maml, num_steps=2, fast_lr=0.25, batch_size=10, baseline=baseline)
    maml_steps = []
    print(" [Evaluating with inner-loop adaptation]")
    for ep in range(N_EPISODES):
        env.reset_task(mission)
        steps = evaluate_policy(env, policy_maml, params=maml_params)
        maml_steps.append(steps)
    mean_maml = np.mean(maml_steps)
    std_maml = np.std(maml_steps)
    results_inner_loop.append(mean_maml)
    print(f"    --> Avg steps: {mean_maml:.2f} ± {std_maml:.2f}")


    # 3. Randomly initialized policy
    scratch_policy = CategoricalMLPPolicy(
        input_size=input_size,
        output_size=output_size,
        hidden_sizes=hidden_sizes,
        nonlinearity=nonlinearity,
    ).to(device)
    scratch_policy.eval()
    rand_steps = []
    print("  [Random policy episodes]")
    for ep in range(N_EPISODES):
        env.reset_task(mission)
        steps = evaluate_policy(env, scratch_policy)
        rand_steps.append(steps)
    mean_rand = np.mean(rand_steps)
    std_rand = np.std(rand_steps)
    results_random.append(mean_rand)
    print(f"    --> Avg steps: {mean_rand:.2f} ± {std_rand:.2f}")

# Avoid multiple result prints for the final aggregate
end_time = time.time()

print(f"Execution time: {(end_time - start_time)/60} minutes")

# --- Overall results ---
print("\n===== FINAL AGGREGATE RESULTS =====")
print(f"Lang-adapted policy: {np.mean(results_lang):.2f} ± {np.std(results_lang):.2f}")
print(f"MAML policy:   {np.mean(results_inner_loop):.2f} ± {np.std(results_inner_loop):.2f}")
print(f"Random initializations:  {np.mean(results_random):.2f} ± {np.std(results_random):.2f}")
