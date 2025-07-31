import gymnasium as gym
import ale_py


gym.register_envs(ale_py)
env=gym.make("ALE/Pacman-v5" , render_mode="human")
observation, info = env.reset()
print(f"Starting observation: {observation}")
episode_over = False
total_reward = 0

while not episode_over:
    
    action = env.action_space.sample()  

    
    observation, reward, terminated, truncated, info = env.step(action)

    
    total_reward += reward
    episode_over = terminated or truncated

print(f"Episode finished! Total reward: {total_reward}")
env.close()