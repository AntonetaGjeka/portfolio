import gym
import numpy as np
import matplotlib.pyplot as plt

# Create Taxi environment
env = gym.make("Taxi-v3")

# Q-table represents the rewards (Q-values) the agent can expect performing a certain action in a certain state
state_space = env.observation_space.n  # total number of states
action_space = env.action_space.n  # total number of actions
qtable = np.zeros((state_space, action_space))  # initialize Q-table with zeros

# Variables for training/testing
test_episodes = 10000  # number of episodes for testing
train_episodes = 50000 # number of episodes for training
episodes = train_episodes + test_episodes  # total number of episodes
max_steps = 100  # maximum number of steps per episode

# Q-learning algorithm hyperparameters to tune
alpha = 0.1  # learning rate
gamma = 0.9  # discount factor

# Exploration-exploitation trade-off
epsilon = 1.0  # probability the agent will explore (initial value is 1.0)
epsilon_min = 0.001  # minimum value of epsilon
epsilon_decay = 0.9999  # decay multiplied with epsilon after each episode

# Lists to hold rewards and number of steps per episode
rewards_per_episode = []
steps_per_episode = []

# Q-learning algorithm
for episode in range(episodes):
    state = env.reset()
    total_reward = 0
    steps = 0

    for step in range(max_steps):
        # Ensure state is an integer (or extract the integer part if it is a tuple)
        if isinstance(state, tuple):
            state = state[0]
        state = int(state)

        if np.random.uniform(0, 1) < epsilon:
            action = env.action_space.sample()  # Explore: select a random action
        else:
            action = np.argmax(qtable[state, :])  # Exploit: select the action with max Q-value for current state

        new_state, reward, done, *_ = env.step(action)  # Take action and observe the result

        # Ensure new_state is an integer (or extract the integer part if it is a tuple)
        if isinstance(new_state, tuple):
            new_state = new_state[0]
        new_state = int(new_state)

        # Update Q-value
        qtable[state, action] = qtable[state, action] + alpha * (
                    reward + gamma * np.max(qtable[new_state, :]) - qtable[state, action])

        state = new_state
        total_reward += reward
        steps += 1

        if done:
            break

    # Decay epsilon
    epsilon = max(epsilon_min, epsilon * epsilon_decay)

    # Save rewards and steps per episode
    rewards_per_episode.append(total_reward)
    steps_per_episode.append(steps)

    if (episode + 1) % 1000 == 0:
        print(f"Episode {episode + 1}/{episodes} completed.")

# Plotting the results
plt.figure(figsize=(14, 5))

# Plot rewards per episode
plt.subplot(1, 2, 1)
plt.plot(range(episodes), rewards_per_episode, color='b')
plt.axvline(x=train_episodes, color='k', linestyle='--', label='Convergence line')
plt.title('Rewards per Episode')
plt.xlabel('Episodes')
plt.ylabel('Total Reward')
plt.legend()
# Plot steps per episode
plt.subplot(1, 2, 2)
plt.plot(range(episodes), steps_per_episode, color='r')
plt.axvline(x=train_episodes, color='k', linestyle='--', label='Convergence line')
plt.title('Steps per Episode')
plt.xlabel('Episodes')
plt.ylabel('Number of Steps')
plt.legend()
plt.tight_layout()
plt.show()

# Close the environment
env.close()
