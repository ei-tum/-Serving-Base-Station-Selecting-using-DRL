import os
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import random
from enum import Enum

# Define the environment for the central paper with RLF and PP logic
# This environment simulates a scenario with multiple users and base stations,
# where users can choose a base station based on SNR values, with penalties for repeated actions
# and rewards based on SNR.
# The environment also includes logic for RLF (Radio Link Failure) and PP (Ping Pong) based on SNR thresholds.

class CentralEnvV2(gym.Env):
    def __init__(
        self,
        data_dir,           # Directory containing the SNR datasets
        num_users=15,       # Number of users in the environment
        num_bs=4,           # Number of Base stations in the environment
        num_datasets=20     # Number of Datasets
    ):
        super().__init__()
        self.data_dir = data_dir
        self.num_users = num_users
        self.num_bs = num_bs
        self.num_datasets = num_datasets

        # RLF & PP Parameter
        self.q_out = -6.7 
        self.q_in = -3.4 
        self.n310_thresh = 10
        self.n311_thresh = 3
        self.t310 = 6
        self.mts = 10 

        self.datasets = self._load_all_datasets()
        self.reset()

        # Observation: [SNR_i for each BS] + one-hot of last chosen BS + [pp_flag, rlf_flag]
        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,  # SNR values range from -10 to 23 dB
            shape=(self.num_bs * 2 + 2,), # Observationshape: 4*SNR + 4*one-hot + 2*flags = 12
            dtype=np.float32
        )
        self.action_space = spaces.Discrete(self.num_bs) # Action Space = 4
        

    # Load all datasets from the specified directory
    # Each dataset corresponds to a user and contains SNR values for each BS

    def _load_all_datasets(self):
        datasets = [] # List to hold all datasets
        for user_idx in range(self.num_users):
            user_data = []
            for file_idx in range(self.num_datasets):
                file_name = f"snr_values-seed_516000000-mac_30-mic_20_user_{user_idx}_{file_idx}.txt"
                file_path = os.path.join(self.data_dir, file_name)
                with open(file_path, "r") as f:
                    lines = [list(map(float, line.strip().split())) for line in f]
                user_data.append(lines)
            datasets.append(np.array(user_data))
        return np.array(datasets) # Array of shape (num_users, num_datasets, num_steps, num_bs)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Randomly select one dataset for each user
        # This ensures that each user has a different SNR dataset for the episode to avoid correlation
        selected_indices = [random.randint(0, self.num_datasets - 1) for _ in range(self.num_users)]
        
        # Extract SNR data for the selected datasets
        # self.snr_data is a 3D array of shape (num_users, num_steps, num_bs)
        # This allows each user to have a unique SNR profile
        self.snr_data = np.array([
            self.datasets[user_idx][file_idx]
            for user_idx, file_idx in enumerate(selected_indices)
        ])
        self.episode_length = self.snr_data.shape[1] # Number of steps in the episode 
        self.current_step = 0
        self.current_user = 0

        # Initialize last actions (initial serving BS)
        # and other state variables
        self.last_actions = np.zeros(self.num_users, dtype=int)
        for user in range(self.num_users):
            self.last_actions[user] = random.randint(0, self.num_bs - 1)

        self.rlf_timer = [0] * self.num_users
        self.out_of_sync_counter = [0] * self.num_users
        self.in_sync_counter = [0] * self.num_users
        self.last_handover_time = [-self.mts] * self.num_users
        self.prev_bs = [-1] * self.num_users

        return self._get_obs(), {}

    def _get_obs(self):
        # SNR values for current user-step
        snr = np.asarray(
            self.snr_data[self.current_user, self.current_step, :],
            dtype=np.float32
        ).flatten()
        assert snr.shape[0] == self.num_bs, f"Expected {self.num_bs} SNR values, got {snr.shape[0]}"

        snr_norm = (snr + 10.0) / 33.0 #normieren SNR-Werte auf [0, 1]
        # One-hot ID auf Serving BS
        bs_id = np.zeros(self.num_bs, dtype=np.float32)
        bs_id[self.last_actions[self.current_user]] = 1.0

        # PP- and RLF-Flags
        pp_flag = 1.0 if (self.current_step - self.last_handover_time[self.current_user] < self.mts) else 0.0
        rlf_flag = 1.0 if self.rlf_timer[self.current_user] > 0 else 0.0

        obs = np.concatenate([snr_norm, bs_id, [pp_flag, rlf_flag]])
        assert obs.shape[0] == self.num_bs * 2 + 2, f"Observation shape incorrect: {obs.shape}"
        return obs

    def step(self, action):
        assert 0 <= action < self.num_bs, "Ungültige Aktion"
        snr = self.snr_data[self.current_user, self.current_step, :]

        sorted_indices = np.argsort(snr)[::-1]
        max_snr = snr[sorted_indices[0]]
        chosen_snr = snr[action]
        pp_triggered = False
        rlf_triggered = False
        reward = 0.0
        
        
        # Full logic: SNR with RLF and PP
        #SNR-based reward
        if np.isclose(chosen_snr, max_snr):
            reward = 1.0
        else:
            reward = -((max_snr - chosen_snr) / 23.0) ** 2
        # Penalty for choosing the same BS too soon: PP Logic
        delta_t = self.current_step - self.last_handover_time[self.current_user]
        if action == self.prev_bs[self.current_user] and delta_t < self.mts:
            reward -= 0.95
            #pp_triggered = True
               
        # RLF logic
        if chosen_snr < self.q_out:
            self.out_of_sync_counter[self.current_user] += 1
            self.in_sync_counter[self.current_user] = 0
        elif chosen_snr > self.q_in:
            self.in_sync_counter[self.current_user] += 1
            self.out_of_sync_counter[self.current_user] = 0
        if self.out_of_sync_counter[self.current_user] >= self.n310_thresh:
            reward -= 1.0
            self.rlf_timer[self.current_user] += 1
        if self.in_sync_counter[self.current_user] >= self.n311_thresh:
            self.rlf_timer[self.current_user] = 0
        if self.rlf_timer[self.current_user] >= self.t310:
            reward -= 2.0
            #rlf_triggered = True
            self.rlf_timer[self.current_user] = 0

        
        # Update last actions and handover time
        if action != self.last_actions[self.current_user]:
            self.last_handover_time[self.current_user] = self.current_step
            self.prev_bs[self.current_user] = self.last_actions[self.current_user]
            self.last_actions[self.current_user] = action

        
        self.current_user += 1
        done = False
        
        if self.current_user >= self.num_users:
            self.current_user = 0
            self.current_step += 1

            if self.current_step >= self.episode_length - 1:
                done = True

        return self._get_obs(), reward, done, False, {}

    def close(self):
        pass
