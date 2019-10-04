import numpy as np

class BanditEnv:

    def __init__(self, actions):
        self.q_star = [np.random.randn() for i in range(actions)]

    def get_reward(self, action):
        return self.q_star[action] + np.random.randn()


class BasicAgent:

    def __init__(self, alpha, epsilon, actions):
        self.alpha = alpha
        self.epsilon = epsilon
        self.Q = [0 for i in range(actions)]

    def take_action(self):
        if np.random.rand() < self.epsilon:
            return np.random.choice(len(self.Q))

        return np.argmax(self.Q)

    def update(self, action, reward):
        self.Q[action] = self.Q[action] + self.alpha * (reward - self.Q[action])


def simulate(steps):
    actions = 10
    agent = BasicAgent(actions)
    environment = BanditEnv(actions)

    for i in range(1, steps + 1):
        action = agent.take_action()
        reward = environment.get_reward(action)
        agent.update(action, reward)

