import os
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback, CallbackList
from central_paper_env_v2 import CentralEnvV2
from central_paper_test_env_v2 import CentralTestEnvV2

# Konfiguration
DATA_DIR = "516000000_train" 
DATA_DIR_TEST = "515000000_test"
NUM_USERS = 15
NUM_BS = 4
NUM_DATASETS = 20
TOTAL_TIMESTEPS = 4_000_000


log_dir = "central_paper_4_mio_steps_norm_neval_1_best_model"
os.makedirs(log_dir, exist_ok=True)

ppo_kwargs = {
    "learning_rate": 5e-5,
    "ent_coef": 0.1
}

policy_kwargs = {
    "net_arch": dict(pi=[64, 128, 64])
}

# Trainings- und Testumgebung vorbereiten
tr_env= CentralEnvV2(
    data_dir=DATA_DIR,
    num_users=NUM_USERS,
    num_bs=NUM_BS,
    num_datasets=NUM_DATASETS)
train_env = Monitor(tr_env)

test_env = CentralTestEnvV2(
    data_dir=DATA_DIR_TEST,
    num_users=NUM_USERS,
    num_bs=NUM_BS)
eval_env = Monitor(test_env)

# PPO initialisieren
model = PPO(
    "MlpPolicy",
    train_env,
    verbose=1,
    policy_kwargs=policy_kwargs,
    tensorboard_log="./ppo_tensorboard_log", # TensorBoard Log-Verzeichnis
    **ppo_kwargs
)

# EvalCallback für diese Phase
eval_callback = EvalCallback(
    eval_env,
    best_model_save_path=log_dir,
    log_path=log_dir,
    eval_freq=50000,  # alle 50.000 Schritte evaluieren
    n_eval_episodes=1,
    deterministic=True,
    verbose=1
    )
    
checkpoint = CheckpointCallback(
    save_freq=100000,  
    save_path=f"{log_dir}/checkpoints/",
    name_prefix="rl_model"
    )

callback_list = CallbackList([eval_callback, checkpoint])
 
    # Training starten
print(f"Training beginnt mit {TOTAL_TIMESTEPS} Mio Timesteps")
model.learn(
    total_timesteps=TOTAL_TIMESTEPS,
    reset_num_timesteps=False,
    callback=callback_list
    )

# Modell speichern
model_save_path = "central_paper_4_mio_steps_norm_neval_1_best_model"
model.save(model_save_path)
print(f"Letztes PPO Modell gespeichert unter: {model_save_path}")
