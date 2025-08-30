import os
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import random
from enum import Enum

class CentralTestEnvV2(gym.Env):
    def __init__(
        self,
        data_dir,
        num_users=15,
        num_bs=4,
        num_datasets=20,
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
            shape=(self.num_bs * 2 + 2,),
            dtype=np.float32
        )
        self.action_space = spaces.Discrete(self.num_bs)


    def _load_all_datasets(self):
        datasets = []
        for user_idx in range(self.num_users):
            file_name = f"snr_values-seed_515000000-mac_30-mic_20_user_{user_idx}_0.txt"
            file_path = os.path.join(self.data_dir, file_name)
            with open(file_path, "r") as f:
                user_data = [list(map(float, line.strip().split())) for line in f]
            datasets.append(np.array(user_data))
        return np.stack(datasets)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.snr_data = self.datasets
        self.episode_length = self.snr_data.shape[1]
        self.current_step = 0
        self.current_user = 0
        self.correct = 0
        self.sec_correct = 0
        self.falsch_counter = 0
        self.pp_counter = 0
        self.rlf_counter = 0
        self.handover_counter = 0
        self.reward_history = []
        self.actual_sinrs = [[] for _ in range(self.num_users)]
        self.max_sinrs = [[] for _ in range(self.num_users)]


        # Initialize last actions (initial serving BS) and bs_load_history
        self.last_actions = np.zeros(self.num_users, dtype=int)
        self.bs_load_history = np.zeros((self.episode_length, self.num_bs), dtype=int)
        self.current_bs_load = np.zeros(self.num_bs, dtype=int)
        for user in range(self.num_users):
            self.last_actions[user] = random.randint(0, self.num_bs - 1)
            self.current_bs_load[self.last_actions[user]] += 1

        self.bs_load_history[0] = self.current_bs_load.copy()
        self.rlf_timer = [0] * self.num_users
        self.out_of_sync_counter = [0] * self.num_users
        self.in_sync_counter = [0] * self.num_users
        self.last_handover_time = [-self.mts] * self.num_users
        self.prev_bs = [-1] * self.num_users

        return self._get_obs(), {}

    def _get_obs(self):
        snr = np.asarray(
            self.snr_data[self.current_user, self.current_step, :],
            dtype=np.float32
        ).flatten()
        snr_norm = (snr + 10.0) / 33.0 #normieren SNR-Werte auf [0, 1]
        # One-hot ID auf Serving BS
        bs_id = np.zeros(self.num_bs, dtype=np.float32)
        bs_id[self.last_actions[self.current_user]] = 1.0

        pp_flag = 1.0 if (self.current_step - self.last_handover_time[self.current_user] < self.mts) else 0.0
        rlf_flag = 1.0 if self.rlf_timer[self.current_user] > 0 else 0.0

        obs = np.concatenate([snr_norm, bs_id, [pp_flag, rlf_flag]])
        assert obs.shape[0] == self.num_bs * 2 + 2, f"Observation shape incorrect: {obs.shape}"
        return obs

    def step(self, action):
        assert 0 <= action < self.num_bs, "Ungültige Aktion"
        snr = self.snr_data[self.current_user, self.current_step, :]

        sorted_idx = np.argsort(snr)[::-1]
        max_snr = snr[sorted_idx[0]]
        second_snr = snr[sorted_idx[1]] if len(sorted_idx) > 1 else -np.inf
        third_snr = snr[sorted_idx[2]] if len(sorted_idx) > 2 else -np.inf
        chosen_snr = snr[action]
        self.actual_sinrs[self.current_user].append(chosen_snr)
        self.max_sinrs[self.current_user].append(max_snr)
        
        pp_triggered = False
        rlf_triggered = False
        reward = 0.0

        
        # SNR_WITH_PP_RLF
        
        #SNR Logik   
        if np.isclose(chosen_snr, max_snr):
            reward = 1.0
            self.correct += 1
            status = "✅ Beste SNR"
        else:
            if np.isclose(chosen_snr, second_snr) and chosen_snr >= 0:
                self.sec_correct += 1
            reward = -((max_snr - chosen_snr) / 23.0) ** 2
            self.falsch_counter += 1
            status = "❌ Schlechte Wahl"
                
        #PP Logic
        prev = self.prev_bs[self.current_user]
        delta = self.current_step - self.last_handover_time[self.current_user]
        if action == prev and delta < self.mts:
            reward -= 0.95
            pp_triggered = True
            self.pp_counter += 1
                
        # RLF Logic
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
            rlf_triggered = True
            self.rlf_counter += 1
            self.rlf_timer[self.current_user] = 0

        # Visualization
        print(f"Step {self.current_step:04d} | User {self.current_user:02d} | Action (BS): {action}")
        print(f"  SNR: {snr}")
        print(f"  Chosen SNR: {status} {chosen_snr:.2f} / Max SNR: {max_snr:.2f}")
        print(f"  PP: {'Yes ❌' if pp_triggered else 'No ✅'} | RLF: {'Yes ❌' if rlf_triggered else 'No ✅'}")
        print("-" * 50)

        # Update last actions and handover time
        if action != self.last_actions[self.current_user]:
            old_bs = self.last_actions[self.current_user]
            self.last_handover_time[self.current_user] = self.current_step
            self.prev_bs[self.current_user] = old_bs
            self.last_actions[self.current_user] = action
            self.handover_counter += 1
            # BS Load aktualisieren
            self.current_bs_load[old_bs] -= 1
            self.current_bs_load[action] += 1
            
        
        self.current_user += 1
        self.reward_history.append(reward)
        done = False
        
        if self.current_user >= self.num_users:
            self.bs_load_history[self.current_step] = self.current_bs_load.copy() # BS Load Historie aktualisieren 
            self.current_user = 0
            self.current_step += 1
            
            if self.current_step >= self.episode_length - 1:
                done = True
                
                total_reward = sum(self.reward_history)
                total_possible = self.episode_length * self.num_users
                accuracy = self.correct / total_possible * 100
                slot_time = 100 # 100 ms per slot
                user_rates = self.compute_userwise_rates()

                print("=== Episode Finished ===")
                print(f"Total Steps: {total_possible}") # Total number of steps in the episode
                print(f"Total correct (Max SNR): {self.correct} Accuracy: ({accuracy:.2f}%)")
                print(f"total 2nd best SNR: {self.sec_correct} ({self.sec_correct / total_possible * 100:.2f}%)")
                print(f"Total Incorrect: {self.falsch_counter} ({self.falsch_counter / total_possible * 100:.2f}%)")
                print(f"Total Reward in an Episode: {total_reward:.3f} ({total_reward / total_possible * 100:.3f}%)")

                total_seconds = (self.episode_length * slot_time) / 1000

                print(f"Total PP Events: {self.pp_counter} ({self.pp_counter / total_possible * 100:.2f}%)")
                pp_event_per_sec = self.pp_counter / total_seconds
                print(f"PP Event per User per Second: {pp_event_per_sec / self.num_users:.3f}") # Korrigierte Zeile

                print(f"Total RLF Events: {self.rlf_counter} ({self.rlf_counter / total_possible * 100:.2f}%)")
                rlf_event_per_sec = self.rlf_counter / total_seconds
                print(f"RLF Event per User per Second: {rlf_event_per_sec / self.num_users:.3f}") # Korrigierte Zeile

                print(f"Total Handover Events: {self.handover_counter} ({self.handover_counter / total_possible * 100:.2f}%)")
                handover_event_per_sec = self.handover_counter / total_seconds
                print(f"Handover Event per User per Second: {handover_event_per_sec / self.num_users:.3f}")
                
                avg_bs_load = self.bs_load_history.mean(axis=0)
                
                print("Durchsatzrate pro Nutzer (bit/s/Hz):")
                for i, (r, r_max, gamma) in enumerate(user_rates):
                    print(f"  User {i:2d}: R = {r:.3f}, R_max = {r_max:.3f}, Γ_R = {gamma:.3f}")

                R_mean = np.mean([r for r, _, _ in user_rates])
                R_max_mean = np.mean([r_max for _, r_max, _ in user_rates])
                gamma_mean = np.mean([g for _, _, g in user_rates])
                print("\n== Durchschnitt über alle Nutzer ==")
                print(f"  R̄     = {R_mean:.3f} bit/s/Hz")
                print(f"  R̄_max = {R_max_mean:.3f} bit/s/Hz")
                print(f"  Γ̄_R   = {gamma_mean:.3f}")
                print("Durchschnittlicher BS Load pro BS über Episode:")
                for i, avg in enumerate(avg_bs_load):
                    print(f"  BS {i}: {avg:.2f} Nutzer im Mittel")
                print("=" * 50)

                             
        return self._get_obs(), reward, done, False, {}
    
    def compute_userwise_rates(self):
        def db_to_linear(db):
            return 10 ** (db / 10)

        user_rates = []
        for u in range(self.num_users):
            sinr_user = self.actual_sinrs[u]
            sinr_max_user = self.max_sinrs[u]

            r_u = np.mean([np.log2(1 + db_to_linear(s)) for s in sinr_user]) 
            r_max_u = np.mean([np.log2(1 + db_to_linear(s)) for s in sinr_max_user]) 
            gamma_u = r_u / r_max_u if r_max_u > 0 else 0.0

            user_rates.append((r_u, r_max_u, gamma_u))

        return user_rates


    def close(self):
        pass
