# In this first task, you will implement the sampling algorithms: (1) epsilon-greedy, (2) UCB, (3) KL-UCB, and (4) Thompson Sampling. This task is straightforward based on the class lectures. The instructions below tell you about the code you are expected to prepare: much of it will generalise to the other tasks, as well.

# You will have to prepare a source file called bandit.py that is used for Task 1 as well as all the other tasks. You can decide for yourself how to modularise the code and name the internal functions. What we shall insist upon is the input-output behaviour of bandit.py, which we describe below.

# bandit.py must accept the following command line parameters.
# --instance in, where in is a path to the instance file.
# --algorithm al, where al is one of epsilon-greedy-t1, ucb-t1, kl-ucb-t1, thompson-sampling-t1, ucb-t2, alg-t3, alg-t4.
# --randomSeed rs, where rs is a non-negative integer.
# --epsilon ep, where ep is a number in [0, 1]. For everything except epsilon-greedy, pass 0.02.
# --scale c, where c is a positive real number. The parameter is only relevant for Task 2; for other tasks pass --scale 2.
# --threshold th, where th is a number in [0, 1]. The parameter is only relevant for Task 4; for other tasks pass --threshold 0.
# --horizon hz, where hz is a non-negative integer.

# Your first job is to simulate a multi-armed bandit. 
# You must read in the bandit instance and have a function to generate a random 0-1 reward with the corresponding probability when a particular arm is pulled. 
# A single random seed will be passed to your program; you must seed the random number generator in your simulation with this seed. 
# If any of your algorithms are randomised, they must also use the same seed for initialisation.

# Given a seed and keeping other input parameters fixed, your entire experiment must be deterministic: it should execute the same way and produce the same result. 
# Of course, the execution will be different for different random seeds; the point being made is that of repeatability for a given seed. 
# You should be able to implement this property by initialising all the random number generators in your program based on the seed provided as input: you should not leave them unseeded or use strategies such as seeding based on system time. 
# Make sure you understand this requirement; if the behaviour of your code does not get fixed by the input random seed (keeping other input parameters fixed), you will not receive any marks for the assignment.

# Having set up the code to pull arms and generate rewards, you must implement the following sampling algorithms: (1) epsilon-greedy, (2) UCB, (3) KL-UCB, and (4) Thompson Sampling. You are free to make assumptions on unspecified aspects such as how the first few pulls get made, how ties get broken, how any algorithm-specific parameters are set, and so on. 
# But you must list all such assumptions in your report. The two external parameters to the given set of algorithms are epsilon for epsilon-greedy sampling, and a threshold for Task 4, both of which will be passed from the command line. Recall that on every round, an algorithm can only use the sequence of pulls and rewards up to that round (or statistics derived from the sequence) to decide which arm to pull. 
# Specifically, it is illegal for an algorithm to have access to the bandit instance itself (although bandit.py has such access).

# Passed an instance, a random seed, an algorithm, epsilon, threshold, and a horizon, your code must run the algorithm on the instance for "horizon" number of pulls and note down the cumulative reward REW. 
# Subtracting REW from the maximum expected cumulative reward possible (the product of the maximum mean and the horizon) will give you REG, the cumulative regret for the particular run. Note that this number can be negative (and might especially turn up so on small horizons—why?). 
# When the algorithm terminates, bandit.py should output a single line with nine entries, separated by commas and terminated with a newline ('\n') character. The line must be in this format; outputFormat.txt contains a few such lines (in which REG and HIGHS are set to arbitrary values just for illustration).

# instance, algorithm, random seed, epsilon, scale, threshold, horizon, REG, HIGHS
# The last entry, HIGHS, is specific to Task 4, and is explained below. However, you must print out a value (say 0) for all the tasks. Similarly, note that epsilon only needs to be used by bandit.py if the algorithm passed is epsilon-greedy-t1; for other algorithms, it is a dummy parameter. Your output must still contain epsilon (either the value passed to it or any other value) to retain the nine-column format. Your REG value for Task 4 is similarly a dummy; up to you whether to print out the actual regret or to put a placeholder value such as 0.

# We will run your code on a subset of input parameters and validate the output with an automatic script. You will not receive any marks for the assignment if your code does not produce output in the format specified above.

# Once you have finished coding bandit.py, run check.sh to make sure that you correctly read in all the command line parameters, and print the output as we described above. While testing your code, we will use a different version of check.sh—with different parameters—and call it inside your submission directory.

# REG is the cumulative regret of the algorithm over the horizon. It is the difference between the maximum expected cumulative reward possible (the product of the maximum mean and the horizon) and the cumulative reward of the algorithm.

import random
import argparse
import numpy as np

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--instance', required=True, help='Path to the instance file')
    parser.add_argument('--algorithm', required=True, help='The algorithm to use')
    parser.add_argument('--randomSeed', required=True, help='The random seed')
    parser.add_argument('--epsilon', required=True, help='The epsilon value')
    parser.add_argument('--scale', required=True, help='The scale value')
    parser.add_argument('--threshold', required=True, help='The threshold value')
    parser.add_argument('--horizon', required=True, help='The horizon value')
    return parser.parse_args()

def seed_random(seed):
    """
    Seed the random number generator
    
    Parameters:
    seed (int): The seed for the random number generator
    """
    random.seed(seed)

def load_instance(filename):
    """ 
    Load instancea from file and return as a list
    
    Parameters:
    filename (str): The path to the file containing the instance
    
    Returns:
    instance_array (list): The instance as a list
    """
    instance_array = []
    with open(filename, 'r') as f:
        for line in f:
            instance_array.append(float(line))
    return instance_array

def pull_arm(arm_probability):
    """
    Pull an arm with given probability
    Note: This is a dummy function that returns a dummy reward based on the arm probability. 
    The function should be replaced with the actual function that pulls the arm and returns the reward.
    
    Parameters:
    arm_probability (float): The probability of the arm
    
    Returns:
    reward (int): The reward for pulling the arm
    """
    if random.random() < arm_probability:
        return 1
    return 0


def epsilon_greedy(instance, epsilon, horizon):
    """
    Implements the epsilon-greedy algorithm for a multi-armed bandit problem using the incremental update rule.
    
    Parameters:
    instance (list of floats): A list of probabilities representing the reward distributions for each arm.
    epsilon (float): The probability of choosing a random arm (exploration) over the best-performing arm (exploitation).
    horizon (int): The number of times the bandit is played, i.e., the total number of pulls.

    Returns:
    total_reward (int): The total cumulative reward accumulated over the horizon.
    """
    
    # Number of arms in the bandit (length of the instance array)
    num_arms = len(instance)
    
    # Initialize the Q-values (estimated average rewards) and number of pulls for each arm
    q_values = [0] * num_arms  # Q-values for each arm (initially 0)
    pulls = [0] * num_arms     # Number of times each arm has been pulled
    
    # Variable to store the total cumulative reward over all pulls
    total_reward = 0
    
    # Iterate through each round up to the given horizon (number of pulls)
    for t in range(horizon):
        
        # With probability epsilon, explore by choosing a random arm
        if random.random() < epsilon:
            # Exploration: Choose a random arm
            arm = random.randint(0, num_arms - 1)  # Select a random arm
        else:
            # Exploitation: Choose the arm with the highest Q-value (estimated average reward)
            arm = np.argmax(q_values)
        
        # Pull the selected arm and get the reward (0 or 1) using the pull_arm function
        reward = pull_arm(instance[arm])
        
        # Update the number of pulls for the selected arm
        pulls[arm] += 1
        
        # Update the Q-value (average reward estimate) for the selected arm using the incremental formula:
        # Q(n+1) = Qn + (1/n) * (Rn - Qn)
        q_values[arm] += (reward - q_values[arm]) / pulls[arm]
        
        # Update the cumulative total reward
        total_reward += reward
    
    # Return the total reward after all pulls have been made
    return total_reward

def calculate_regret(instance, total_reward, horizon):
    """
    Calculate the regret of the algorithm over the horizon.
    The regret is the difference between the maximum expected cumulative reward possible and the cumulative reward of the algorithm.
    
    Parameters:
    instance (list of floats): A list of probabilities representing the reward distributions for each arm. This is the actual expected reward for each arm.
    total_reward (int): The cumulative reward calculated by the algorithm
    horizon (int): The number of times the bandit is played, i.e., the total number of pulls.
    
    Retruns:
    regret (int) 
    """
    max_mean = max(instance)
    return max_mean * horizon - total_reward

def ucb(instance, horizon):
    """ 
    UCB 
    """
    num_arms = len(instance)  # Number of arms (each arm has an associated reward probability)
    rewards = [0] * num_arms  # List to store the cumulative rewards for each arm
    pulls = [0] * num_arms    # List to track how many times each arm has been pulled
    total_reward = 0          # Keeps track of the total reward accumulated

    # Step 1: Initial exploration - Pull each arm once to gather some initial data
    for t in range(min(num_arms, horizon)):  # Loop through all arms or until the horizon if less
        reward = pull_arm(instance[t])       # Pull each arm once
        rewards[t] += reward                 # Update the cumulative reward for arm t
        pulls[t] += 1                        # Increment the pull count for arm t
        total_reward += reward               # Add the reward to the total reward

    # Step 2: Main loop - Pull arms using the UCB formula for the remaining time steps
    for t in range(num_arms, horizon):  # Start from the first time step after initial exploration
        # Step 2.1: Calculate the UCB value for each arm
        ucb_values = [rewards[i] / pulls[i] + np.sqrt(2 * np.log(t + 1) / pulls[i]) for i in range(num_arms)]
        
        # Step 2.2: Select the arm with the highest UCB value
        arm = np.argmax(ucb_values)
        
        # Step 2.3: Pull the selected arm and update rewards and pulls
        reward = pull_arm(instance[arm])
        rewards[arm] += reward  # Add the new reward to the cumulative reward for the selected arm
        pulls[arm] += 1         # Increment the pull count for the selected arm
        total_reward += reward   # Add the reward to the total reward

    return total_reward  # Return the total reward accumulated over the horizon


def main():
    args = parse_args()
    seed_random(args.randomSeed)

    # Load the instance file (arm probabilities)
    instance = load_instance(args.instance)

    # Initialize variables
    if args.algorithm == 'epsilon-greedy-t1':
        total_reward = epsilon_greedy(instance, args.epsilon, args.horizon)
    elif args.algorithm == 'ucb-t1':
        total_reward = ucb(instance, args.horizon)
    elif args.algorithm == 'kl-ucb-t1':
        total_reward = kl_ucb(instance, args.horizon)
    elif args.algorithm == 'thompson-sampling-t1':
        total_reward = thompson_sampling(instance, args.horizon)
    else:
        print("Unknown algorithm")
        return

    # Calculate regret
    regret = calculate_regret(instance, total_reward, args.horizon)

    # Output the result in the required format
    print(f"{args.instance},{args.algorithm},{args.randomSeed},{args.epsilon},{args.scale},{args.threshold},{args.horizon},{regret},0")

if __name__ == '__main__':
    main()
