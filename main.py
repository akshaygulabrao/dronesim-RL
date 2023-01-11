import gym
import numpy as np
from agent import Agent
from utils import plot_learning_curve, manage_memory
from dronesim import DroneSim
from gym.wrappers import FlattenObservation,TimeLimit

if __name__ == '__main__':
    manage_memory()
    env = TimeLimit(FlattenObservation(DroneSim(render_mode="human")),max_episode_steps=30)
    agent = Agent(input_dims=(6,), env=env,
                  n_actions=(2,),
                  alpha=0.0001, beta=0.001)
    n_games = 1000

    figure_file = 'plots/lunar_lander.png'

    best_score = env.reward_range[0]
    score_history = []
    load_checkpoint = False

    if load_checkpoint:
        agent.load_models()
        evaluate = True
    else:
        evaluate = False
    evaluate = False
    for i in range(n_games):
        observation, info = env.reset()
        done = False
        score = 0
        while not done:
            action = agent.choose_action(observation, evaluate)
            observation_, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            score += reward
            agent.store_transition(observation, action, reward,
                                   observation_, done)
            if not load_checkpoint:
                agent.learn()
            observation = observation_

        score_history.append(score)
        avg_score = np.mean(score_history[-100:])

        if avg_score > best_score:
            best_score = avg_score
            if not evaluate: agent.save_models()

        print('episode {} score {:.1f} avg score {:.1f}'.
              format(i, score, avg_score))

    if not load_checkpoint:
        x = [i+1 for i in range(n_games)]
        plot_learning_curve(x, score_history, figure_file)
