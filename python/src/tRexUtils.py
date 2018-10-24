import numpy as np

def linearly_decaying_epsilon(step, epsilon_init, decay_period, warmup_steps, epsilon_final):
  """Returns the current epsilon for the agent's epsilon-greedy policy.
  This follows the Nature DQN schedule of a linearly decaying epsilon (Mnih et
  al., 2015). The schedule is as follows:
    Begin at 1. until warmup_steps steps have been taken; then
    Linearly decay epsilon from 1. to epsilon in decay_period steps; and then
    Use epsilon from there on.
  Args:
    decay_period: float, the period over which epsilon is decayed.
    step: int, the number of training steps completed so far.
    warmup_steps: int, the number of steps taken before epsilon is decayed.
    epsilon: float, the final value to which to decay the epsilon parameter.
  Returns:
    A float, the current epsilon value computed according to the schedule.
  """
  assert epsilon_init > epsilon_final
  steps_left = decay_period + warmup_steps - step
  bonus = (epsilon_init - epsilon_final) * steps_left / decay_period
  bonus = np.clip(bonus, 0., epsilon_init - epsilon_final)
  return epsilon_final + bonus

def linearly_decaying_beta(step, decay_period, beta):
    beta_diff = 1 - beta
    steps_left = decay_period - step
    bonus = beta_diff / decay_period
    bonus = np.clip(bonus, 0., beta_diff)
    return beta + bonus
