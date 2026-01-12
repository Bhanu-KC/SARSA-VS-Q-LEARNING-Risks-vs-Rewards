The article investigates how Q-learning and SARSA perform on the Cliff Walking environment, a grid-based reinforcement learning task where an agent must reach a goal while avoiding high-risk cliff cells. Q-learning uses an off-policy update rule that selects actions based on the maximum future reward, making it more aggressive and risk-taking. The environment provides −1 reward for each step and −100 for falling off the cliff. The key equation for Q-learning updates is:
Q(s,a) ← Q(s,a) + α [r + γ * max(Q(s′,:)) − Q(s,a)].
This research demonstrates that Q-learning learns faster but takes riskier paths, whereas SARSA is more stable and safer.

Done by Bhanu Prakash KC, Bistrit Pandey, Sushsant Poudel, Swadha Pandey

