# What is value function
# The value function in the context of a Markov Decision Process (MDP) is a function that estimates the expected cumulative reward an agent can achieve starting from a particular state and following a particular policy.

# What is policy
# In the context of a Markov Decision Process (MDP), a policy is a function that maps states to actions. It defines the agent's behavior in the environment by specifying the action to take in each state.


def value_iteration(S, A, transitions, gamma, theta=1e-6):
    """
    Computes the optimal value function and policy using Value Iteration.

    Parameters:
    - S: The number of states.
    - A: The number of actions.
    - transitions: A list of dictionaries, each containing:
        - current_state: The current state
        - action: The action taken
        - next_state: The resulting state
        - reward: The reward received
        - transition_probability: The probability of this transition
    - gamma: Discount factor for future rewards, between 0 and 1.
    - theta: Convergence threshold for the value function.

    Returns:
    - value_function: A dictionary with optimal values for each state.
    - optimal_policy: A dictionary with the best action for each state.
    """
    
    # Step 1: Initialize the value function for all states to 0
    value_function = {}
    for s in range(S):
        value_function[s] = 0
        
    # Step 2: Perform value iteration until convergence
    while True:
        # Initialize max change to track convergence
        max_change = 0
        
        # Iterate over all states to update their value
        for state in range(S):
            # Save the current value for comparison later
            current_value = value_function[state]
            
            # Calculate the value for each action and store the highest
            best_value = float('-inf')
            for action in range(A):
                expected_value = 0
                
                # Calculate the expected value for taking this action
                for transition in transitions:
                    # Check if this transition matches the current state and action
                    if transition['current_state'] == state and transition['action'] == action:
                        next_state = transition['next_state']
                        reward = transition['reward']
                        transition_probability = transition['transition_probability']
                        
                        # Update the expected value for this action
                        expected_value += transition_probability * (reward + gamma * value_function[next_state])
                
                # Track the highest action value
                if expected_value > best_value:
                    best_value = expected_value
            
            # Update the value function for this state
            value_function[state] = best_value
            
            # Calculate the change and update max_change if needed
            max_change = max(max_change, abs(current_value - best_value))
        
        # Check for convergence using the threshold theta
        if max_change < theta:
            break

    # Step 3: Derive the optimal policy based on the optimal value function
    optimal_policy = {}
    for state in range(S):
        best_action = None
        best_action_value = float('-inf')
        
        # Find the action with the highest expected value for each state
        for action in range(A):
            expected_value = 0
            
            # Calculate the expected value for this action
            for transition in transitions:
                # Check if this transition matches the current state and action
                if transition['current_state'] == state and transition['action'] == action:
                    next_state = transition['next_state']
                    reward = transition['reward']
                    transition_probability = transition['transition_probability']
                    
                    # Update the expected value for this action
                    expected_value += transition_probability * (reward + gamma * value_function[next_state])
            
            # Track the best action for this state
            if expected_value > best_action_value:
                best_action_value = expected_value
                best_action = action
        
        # Assign the best action to the policy for this state
        optimal_policy[state] = best_action

    return value_function, optimal_policy
    
    
def read_mdp_file(file_path):
    """
    Reads an MDP file and extracts the number of states, number of actions, 
    transitions, and the discount factor.

    Parameters:
    - file_path: Path to the MDP input file.

    Returns:
    - S: Number of states.
    - A: Number of actions.
    - transitions: List of dictionaries for transitions.
    - gamma: Discount factor.
    """
    transitions = []
    S = 0
    A = 0
    gamma = None

    with open(file_path, 'r') as file:
        lines = file.readlines()
        for line in lines:
            line = line.strip()
            if line.startswith('numStates'):
                S = int(line.split()[1])
            elif line.startswith('numActions'):
                A = int(line.split()[1])
            elif line.startswith('transition'):
                parts = line.split()
                current_state = int(parts[1])
                action = int(parts[2])
                next_state = int(parts[3])
                reward = float(parts[4])
                transition_probability = float(parts[5])
                transitions.append({
                    "current_state": current_state,
                    "action": action,
                    "next_state": next_state,
                    "reward": reward,
                    "transition_probability": transition_probability
                })
            elif line.startswith('discount'):
                gamma = float(line.split()[1])
    
    return S, A, transitions, gamma

# Example usage
if __name__ == "__main__":
    # Specify the path to your MDP file
    mdp_file_path = "/workspaces/Reinforcement Learning/RL-2/data/mdp/episodic-mdp-10-5.txt"
    
    S, A, transitions, gamma = read_mdp_file(mdp_file_path)
    
    # Now you can call the value_iteration function with these parameters
    value_function, optimal_policy = value_iteration(S, A, transitions, gamma)
    
    # Print the results
    print("Optimal Value Function:")
    for state, value in value_function.items():
        print(f"State {state}: {value:.6f}")
    
    print("\nOptimal Policy:")
    for state, action in optimal_policy.items():
        print(f"State {state}: Take action {action}")
