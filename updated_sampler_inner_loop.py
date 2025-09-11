import numpy as np
import torch
import torch.nn as nn
import gym
from maml_rl.baseline import LinearFeatureBaseline
from maml_rl.episode import BatchEpisodes
from maml_rl.utils.reinforcement_learning import reinforce_loss
from sklearn.feature_extraction.text import CountVectorizer
import re


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



# Mission Encoder MLP 
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

def preprocess_obs(obs):

    image = obs["image"].flatten() / 255.0
    direction = np.eye(4)[obs["direction"]]
    mission_vec = vectorizer.transform([obs["mission"]]).toarray()[0]

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
                 env=None,              # Prebuilt BabyAI env with mission wrapper   
                 batch_size=None,        # Number of episodes per task (fast_batch_size)
                 policy=None,
                 baseline=None,     
                 seed=None,
                 num_workers=0):  
                 
        assert env is not None, "Must pass prebuilt BabyAI env!"
        self.env = env
        self.batch_size = batch_size
        self.policy = policy
        self.baseline = baseline
        self.seed = seed

    def sample_tasks(self, num_tasks):
        # Calls your env wrapper's sample_tasks (returns mission strings)
        return self.env.sample_tasks(num_tasks)

    def sample(self, meta_batch_size, num_steps=1, fast_lr=0.5, gamma=0.95, gae_lambda=1.0, device='cpu'):
        tasks = self.sample_tasks(meta_batch_size)  # Sample tasks of given number
        print(f"Sampled {len(tasks)} tasks: {tasks}")
        train_episodes_all = []
        valid_episodes_all = []
        all_step_counts = []  # Collect step counts for each episode
        for task_index, task in enumerate(tasks):
            self.env.reset_task(task)
            print(f"Starting episodes for task '{task}'")
            # --- Inner adaptation: collect training episodes and adapt policy ---
            train_batches = []
            params = None
            for _ in range(num_steps):
                batch = BatchEpisodes(batch_size=self.batch_size, gamma=gamma, device=device)
                for ep in range(self.batch_size):
                    obs, info = self.env.reset()
                    done = False

                    episode_obs = []
                    episode_actions = []
                    episode_rewards = []
                    step_count = 0

                    while not done:
                        obs_vec = preprocess_obs(obs)
                        if np.isnan(obs_vec).any():
                            print("NaN in obs_vec, skipping episode")
                            break
                        obs_tensor = np.expand_dims(obs_vec, axis=0)
                        obs_tensor = torch.from_numpy(obs_tensor).float().to(device)
                        with torch.no_grad():
                            pi = self.policy(obs_tensor, params=params)
                            action = pi.sample().item()
                        if np.isnan(action):
                            print("NaN in action, skipping episode")
                            break
                        obs, reward, terminated, truncated, info = self.env.step(action)
                        step_count += 1
                        # done = terminated or truncated
                        # self.env.render("human")
                        if np.isnan(reward):
                            print("NaN in reward, skipping episode")
                            break

                        done = terminated or truncated
                        episode_obs.append(obs_vec)
                        episode_actions.append(action)
                        episode_rewards.append(reward)
                    # Check if episode is valid (e.g., not all zeros, not empty, etc.)
                    all_step_counts.append(step_count)  # Collect step count
                    # print(f"Episode {ep} finished in {step_count} steps for task '{task}'")
                    if len(episode_obs) > 0 and not np.isnan(episode_obs).any():
                        valid = True
                        # batch.append(episode_obs, episode_actions, episode_rewards, [ep]*len(episode_obs))
                        batch.append(
                            episode_obs,
                            [np.array(a) for a in episode_actions],
                            [np.array(r) for r in episode_rewards],
                            [ep]*len(episode_obs)
                        )
                       
                       
                        # batch.append([obs_vec], [np.array(action)], [np.array(reward)], [ep])
                self.baseline.fit(batch)
                batch.compute_advantages(self.baseline, gae_lambda=gae_lambda, normalize=True)
                
                if torch.isnan(batch.advantages).any():
                    print("NaN in batch advantages!")
                if torch.isnan(batch.observations).any():
                    print("NaN in batch observations!")
                # print("Batch rewards:", batch.rewards)
                
                # MAML adaptation step
                if torch.isnan(batch.observations).any():
                    print("NaN in batch observations!")
                loss = reinforce_loss(self.policy, batch, params=params)
                params = self.policy.update_params(loss, params=params, step_size=fast_lr, first_order=True)
                train_batches.append(batch)
            train_episodes_all.append(train_batches)
            # --- Outer evaluation: collect validation episodes with adapted policy ---
            valid_batch = BatchEpisodes(batch_size=self.batch_size, gamma=gamma, device=device)
            for ep in range(self.batch_size):
                obs, info = self.env.reset()
                done = False
                while not done:
                    obs_vec = preprocess_obs(obs)
                    obs_tensor = np.expand_dims(obs_vec, axis=0)
                    obs_tensor = torch.from_numpy(obs_tensor).float().to(device)
                    with torch.no_grad():
                        pi = self.policy(obs_tensor, params=params)
                        action = pi.sample().item()
                    obs, reward, terminated, truncated, info = self.env.step(action)
                    done = terminated or truncated
                    valid_batch.append([obs_vec], [np.array(action)], [np.array(reward)], [ep])
            
            self.baseline.fit(valid_batch)
            valid_batch.compute_advantages(self.baseline, gae_lambda=gae_lambda, normalize=True)
            valid_episodes_all.append(valid_batch)
            print(f"Task {task_index} ({task}): {len(train_batches)} train batches, 1 valid batch")
        return (train_episodes_all, valid_episodes_all, all_step_counts)
