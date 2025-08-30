# Test- und Evaluierungsskript für das trainierte Modell im CentralPpoTestEnv

#test_script 

import torch
from stable_baselines3 import PPO
#from central_paper_test_env import CentralTestEnv # Importieren der Testumgebung
from central_paper_test_env_v2 import CentralTestEnvV2


# Konfiguration
MODEL_PATH = "./central_paper_4_mio_steps_norm_neval_1_best_model/best_model"
#MODEL_PATH = "./central_paper_12mio_steps_norm_best_model/best_model"
DATA_DIR = "515000000_test"  # Ordner mit allen Dateien 515000000_test
NUM_USERS = 15 # Anzahl der Benutzer
NUM_BS = 4 # Anzahl der Basisstationen

# Prüfen, ob CUDA verfügbar ist und das Modell auf die GPU laden
if torch.cuda.is_available():
    device = torch.device("cuda")
    print("CUDA verfügbar, Modell wird auf die GPU geladen.")
else:
    device = torch.device("cpu")
    print("CUDA nicht verfügbar, Modell wird auf die CPU geladen.")

# Environment laden
env = CentralTestEnvV2(
    data_dir=DATA_DIR,
    num_users=NUM_USERS,
    num_bs=NUM_BS,
)

# Modell laden
model = PPO.load(MODEL_PATH, env=env)

# Testen über eine Episode
obs, _ = env.reset()
done = False
total_reward = 0
test_episode = 1

for i in range(test_episode):
    
    while not done:
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, _, _ = env.step(action)
        total_reward += reward
        

    print(f"Gesamtreward dieser Episode: {total_reward}\n")



