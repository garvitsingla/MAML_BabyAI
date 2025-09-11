import numpy as np
import gym
import torch
import torch.nn as nn
import multiprocessing as mp
from queue import Empty as _QueueEmpty
from minigrid.envs.babyai.goto import GoToLocal, GoTo
from minigrid.envs.babyai.core.verifier import GoToInstr, ObjDesc
from minigrid.core.world_object import Ball, Box, Key
import re
from maml_rl.episode import BatchEpisodes
from maml_rl.utils.reinforcement_learning import reinforce_loss

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

vectorizer = None
mission_encoder = None

# Mission Wrapper for BabyAI environments
class BabyAIMissionTaskWrapper(gym.Wrapper):
    def __init__(self, env, missions=None):
        assert missions is not None, "You must provide a missions list!"
        super().__init__(env)
        self.missions = missions
        self.current_mission = None
        # print("missions", self.missions)

    def sample_tasks(self, n_tasks):
        return list(np.random.choice(self.missions, n_tasks, replace=False))

    def reset_task(self, mission):
        self.current_mission = mission
        if hasattr(self.env, 'set_forced_mission'):
            self.env.set_forced_mission(mission)

    def reset(self, **kwargs):        
        result = super().reset(**kwargs)
        # Gymnasium returns (obs, info), old Gym just obs
        if isinstance(result, tuple):
            obs, info = result
        else:
            obs = result
            info = {}
        if self.current_mission is not None:
            obs['mission'] = self.current_mission
        if isinstance(result, tuple):
            return obs, info
        else:
            return obs
        

# --- Mission Encoder MLP ---
class MissionEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim1=32, hidden_dim2=64, output_dim=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim1),
            nn.ReLU(),
            nn.Linear(hidden_dim1, hidden_dim2),
            nn.ReLU(),
            nn.Linear(hidden_dim2, output_dim)
        )
    def forward(self, x):
        return self.net(x)

# MissionParamAdapter 
class MissionParamAdapter(nn.Module):
    def __init__(self, mission_dim, policy_param_shapes):
        super().__init__()
        self.mission_dim = mission_dim
        self.policy_param_shapes = policy_param_shapes
        total_params = sum([torch.Size(shape).numel() for shape in policy_param_shapes])
        self.net = nn.Sequential(
            nn.Linear(mission_dim, 128),
            nn.ReLU(),
            nn.Linear(128, total_params),
            nn.Tanh()  # Ensure parameters are in a reasonable range
        )
    def forward(self, mission_emb):
        # mission_emb: (batch, mission_dim)
        out = self.net(mission_emb)  # (batch, total_params)
        split_points = []
        total = 0
        for shape in self.policy_param_shapes:
            num = torch.Size(shape).numel()
            split_points.append(total + num)
            total += num
        chunks = torch.split(out, [torch.Size(shape).numel() for shape in self.policy_param_shapes], dim=1)
        reshaped = [chunk.view(-1, *shape) for chunk, shape in zip(chunks, self.policy_param_shapes)]
        return reshaped  # List of [batch, *param_shape]
    

def preprocess_obs(obs):

    if vectorizer is None or mission_encoder is None:
        raise RuntimeError("Set sampler_lang_2.vectorizer and sampler_lang_2.mission_encoder before calling preprocess_obs().")
    image = obs["image"].flatten() / 255.0
    direction = np.eye(4)[obs["direction"]]
    mission_vec = vectorizer.transform([obs["mission"]]).toarray()[0]
    # # Project mission vector through MLP
    # mission_tensor = torch.from_numpy(mission_vec.astype(np.float32)).unsqueeze(0).to(device)
    
    # Make sure mission_encoder and mission_tensor are on the same device
    mission_tensor = torch.from_numpy(mission_vec.astype(np.float32)).unsqueeze(0).to(device)
    mission_encoder_device = next(mission_encoder.parameters()).device
    if mission_tensor.device != mission_encoder_device:
        mission_tensor = mission_tensor.to(mission_encoder_device)
    with torch.no_grad():
        mission_emb = mission_encoder(mission_tensor).cpu().numpy().squeeze()
    obs_vec = np.concatenate([image, direction, mission_emb])
    return obs_vec.astype(np.float32)

class MultiTaskSampler(object):
    def __init__(self,
                 env=None,          # Pre-wrapped BabyAI env
                #  env_kwargs,        # Unused for BabyAI mode; pass {}
                 batch_size=None,        # Number of episodes per task (fast_batch_size)
                 policy=None,
                 baseline=None,
                 seed=None,
                 num_workers=0):    # Must be 0 for this mode
        assert env is not None, "Must pass prebuilt BabyAI env!"
        self.env = env
        self.batch_size = batch_size
        self.policy = policy
        self.baseline = baseline
        self.seed = seed

    def sample_tasks(self, num_tasks):
        # Calls your env wrapper's sample_tasks (returns mission strings)
        return self.env.sample_tasks(num_tasks)

    def sample(self, meta_batch_size, meta_learner, num_steps=1, fast_lr=0.5, gamma=0.95, gae_lambda=1.0, device='cpu'):
        tasks = self.sample_tasks(meta_batch_size)  # Sample tasks of given number
        # print(f"Sampled {len(tasks)} tasks: {tasks}")

        all_step_counts = []
        valid_episodes_all = []
          # Collect step counts for each episode
        for task_index,task in enumerate(tasks):
            self.env.reset_task(task)
            # print(f"\nTask {task_index}: {task}")
            # ---- Language-based adaptation: get params for this task ----
            params = meta_learner.adapt_one(task)  # THETA_PRIME for each task (inner loop gor over here)            

            # print("Theta-prime is generated for task:", task)
            # --- Outer evaluation: collect validation episodes with adapted params ---
            valid_batch = BatchEpisodes(batch_size=self.batch_size, gamma=gamma, device=device)
            valid_batch.mission = task

            for ep in range(self.batch_size):
                # print(f"Collecting episode {ep+1} for task {task_index+1}")
                obs, info = self.env.reset()
                done = False
                step_count = 0
                while not done:
                    obs_vec = preprocess_obs(obs)
                    obs_tensor = np.expand_dims(obs_vec, axis=0)
                    obs_tensor = torch.from_numpy(obs_tensor).float().to(device)
                    with torch.no_grad():
                        pi = self.policy(obs_tensor, params=params)
                        action = pi.sample().item()
                    obs, reward, terminated, truncated, info = self.env.step(action)
                    done = terminated or truncated
                    step_count += 1
                    valid_batch.append([obs_vec], [np.array(action)], [np.array(reward)], [ep])
                all_step_counts.append(step_count)
                # print(f"Ep{ep+1}. steps taken: {step_count}")
            self.baseline.fit(valid_batch)
            valid_batch.compute_advantages(self.baseline, gae_lambda=gae_lambda, normalize=True)
            
            valid_episodes_all.append(valid_batch)   # for 1 particular task
        
        return (valid_episodes_all, all_step_counts)   

