
import matplotlib.pyplot as plt
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


class Config:

    def __init__(self, alpha, epsilon, times, steps):
        self.alpha = alpha
        self.epsilon = epsilon
        self.times = times
        self.steps = steps


def simulate(config):
    rewards = np.zeros((config.times, config.steps))

    for t in range(config.times):
        agent = BasicAgent(alpha=0.1, epsilon=0.05, actions=10)
        environment = BanditEnv(actions=10)

        for s in range(config.steps):
            action = agent.take_action()
            reward = environment.get_reward(action)
            agent.update(action, reward)

            rewards[t, s] = reward

    return rewards.mean(axis=0)


if __name__ == '__main__':
    configuration = Config(alpha=0.1, epsilon=0.05, times=100, steps=2000)

    mean_reward = simulate(configuration)
    plt.plot(range(len(mean_reward)), mean_reward)
    plt.show()