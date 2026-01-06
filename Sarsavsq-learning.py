import numpy as np
import matplotlib.pyplot as plt

# ---------------- Environment Setup ----------------
ROWS, COLS = 4, 12
START, GOAL = (3, 0), (3, 11)
CLIFF = [(3, i) for i in range(1, 11)]
ACTIONS = {0: (-1, 0), 1: (1, 0), 2: (0, -1), 3: (0, 1)}  # up, down, left, right

def step(state, action):
    r, c = state
    dr, dc = ACTIONS[action]
    nr, nc = r + dr, c + dc
    nr = min(max(nr, 0), ROWS-1)
    nc = min(max(nc, 0), COLS-1)
    next_state = (nr, nc)
    if next_state in CLIFF:
        return START, -100, True
    if next_state == GOAL:
        return next_state, 0, True
    return next_state, -1, False

def epsilon_greedy(Q, state, epsilon):
    if np.random.rand() < epsilon:
        return np.random.choice(len(ACTIONS))
    r, c = state
    return np.argmax(Q[r, c])

# ---------------- Q-Learning ----------------
def q_learning_risk_rewards(episodes=500, alpha=0.5, gamma=1.0, epsilon=0.1):
    Q = np.zeros((ROWS, COLS, len(ACTIONS)))
    rewards, cliff_falls = [], []

    for _ in range(episodes):
        state = START
        total_reward = 0
        falls = 0
        while True:
            action = epsilon_greedy(Q, state, epsilon)
            next_state, reward, done = step(state, action)
            r, c = next_state
            best_next_action = np.argmax(Q[r, c])
            r0, c0 = state
            Q[r0, c0, action] += alpha * (reward + gamma * Q[r, c, best_next_action] - Q[r0, c0, action])
            total_reward += reward
            if reward == -100:
                falls += 1
            state = next_state
            if done:
                break
        rewards.append(total_reward)
        cliff_falls.append(falls)
    return Q, rewards, cliff_falls

# ---------------- SARSA ----------------
def sarsa_risk_rewards(episodes=500, alpha=0.5, gamma=1.0, epsilon=0.1):
    Q = np.zeros((ROWS, COLS, len(ACTIONS)))
    rewards, cliff_falls = [], []

    for _ in range(episodes):
        state = START
        action = epsilon_greedy(Q, state, epsilon)
        total_reward = 0
        falls = 0
        while True:
            next_state, reward, done = step(state, action)
            next_action = epsilon_greedy(Q, next_state, epsilon)
            r, c = state
            nr, nc = next_state
            Q[r, c, action] += alpha * (reward + gamma * Q[nr, nc, next_action] - Q[r, c, action])
            total_reward += reward
            if reward == -100:
                falls += 1
            state, action = next_state, next_action
            if done:
                break
        rewards.append(total_reward)
        cliff_falls.append(falls)
    return Q, rewards, cliff_falls

# ---------------- Run Both Algorithms ----------------
episodes = 500
q_Q, q_rewards, q_falls = q_learning_risk_rewards(episodes)
sarsa_Q, sarsa_rewards, sarsa_falls = sarsa_risk_rewards(episodes)

# ---------------- Q-Learning Scatter Plot ----------------
plt.figure(figsize=(12,5))
plt.scatter(range(episodes), q_rewards, c='red', s=10, label='Reward (Negative)')
plt.scatter(range(episodes), q_falls, c='black', s=10, label='Cliff Falls (Risk)')
plt.title("Q-Learning: Rewards vs Risk per Episode")
plt.xlabel("Episode")
plt.ylabel("Reward / Falls")
plt.legend()
plt.grid(True)
plt.show()

# ---------------- SARSA Scatter Plot ----------------
plt.figure(figsize=(12,5))
plt.scatter(range(episodes), sarsa_rewards, c='blue', s=10, label='Reward (Negative)')
plt.scatter(range(episodes), sarsa_falls, c='black', s=10, label='Cliff Falls (Risk)')
plt.title("SARSA: Rewards vs Risk per Episode")
plt.xlabel("Episode")
plt.ylabel("Reward / Falls")
plt.legend()
plt.grid(True)
plt.show()

# Print learned Q-table for Q-Learning
print("Learned Q-Table (Q-Learning):")
print(q_Q)

# Simulate Q-Learning policy
print("\nSimulation using Q-Learning policy:")
state = START
while state != GOAL:
    r, c = state
    action = np.argmax(q_Q[r, c])
    print(f"State: {state}, Action: {action}")
    state, reward, done = step(state, action)
