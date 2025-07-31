import gymnasium as gym
from stable_baselines3 import DQN
from stable_baselines3.common.evaluation import evaluate_policy
import ale_py


gym.register_envs(ale_py)
# Create the environment
env = gym.make("ALE/MsPacman-v5", render_mode="human")

# Initialize the DQN model
model = DQN("CnnPolicy", env, verbose=1, buffer_size=50000, learning_starts=1000, train_freq=4, target_update_interval=1000)

# Train the model
model.learn(total_timesteps=100000)  # You can increase this later for better performance

# Save the model
model.save("dqn_pacman")

# Evaluate the trained agent
mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=5)
print(f"Mean reward: {mean_reward}, Std: {std_reward}")


env = gym.make("ALE/MsPacman-v5", render_mode="human")
obs, info = env.reset()
done = False

while not done:
    action, _states = model.predict(obs)
    obs, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated

env.close()
