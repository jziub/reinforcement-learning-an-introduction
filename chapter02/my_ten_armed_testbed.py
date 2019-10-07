
import matplotlib.pyplot as plt
import numpy as np


class BanditEnv:

    def __init__(self, actions):
        self.q_star = [np.random.randn() for i in range(actions)]
        self.best_action = np.argmax(self.q_star)

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
    best_action_count = np.zeros((config.times, config.steps))

    for t in range(config.times):
        agent = BasicAgent(alpha=0.1, epsilon=config.epsilon, actions=10)
        environment = BanditEnv(actions=10)

        for s in range(config.steps):
            action = agent.take_action()
            reward = environment.get_reward(action)
            agent.update(action, reward)

            rewards[t, s] = reward
            best_action_count[t, s] = 1 if action == environment.best_action else 0

    return best_action_count.mean(axis=0), rewards.mean(axis=0)


def figure_2_2():
    epsilons = [0, 0.01, 0.1]
    results = []

    for epsilon_val in epsilons:
        configuration = Config(alpha=0.1, epsilon=epsilon_val, times=500, steps=2000)
        mean_best_action_count, mean_reward = simulate(configuration)
        results.append((epsilon_val, mean_best_action_count, mean_reward))

    plt.figure(figsize=(10, 20))
    plt.subplot(2, 1, 1)
    for item in results:
        plt.plot(range(len(item[2])), item[2], label='epsilon = %.02f' % item[0])
    plt.legend()

    plt.subplot(2, 1, 2)
    for item in results:
        plt.plot(range(len(item[1])), item[1], label='epsilon = %.02f' % item[0])
    plt.legend()

    plt.show()


if __name__ == '__main__':
    figure_2_2()
