# Active Sensing Policy Learning Approach

## Summary and Rationale

This document outlines an approach to develop a policy learning system that enables a robot to actively sense its environment and navigate obstacles using learned IID (Interaural Intensity Difference) and distance measurements. The key innovation is giving the robot control over its sensing strategy through strategic rotations before movement.

### Core Problem and Motivation

The current sonar system is fixed to the robot's orientation, limiting its ability to gather information about the environment. By allowing the robot to rotate, measure, rotate again, and then drive, we enable **active sensing** - the robot can choose where to "look" before deciding where to go.

### Biological Inspiration

This approach is bio-inspired by how bats might use internal models of echo acoustics to plan movement. The emulator network (trained in SCRIPT_TrainEmulator.py) serves as the robot's "imagination" of how the world would sound from different positions and orientations.

## System Architecture

```
Environment Profiles → Emulator (World Model)
                                      ↓
[Policy NN] ← (IID history, Distance history, Movement history)
                                      ↓
[Actions: rotate1, measure, rotate2, drive]
                                      ↓
Real/Simulated Environment → New state → Repeat
```

## Action Sequence

The robot's action cycle consists of:
1. **rotate1**: First rotation (can be 0°)
2. **measure**: Take IID and distance measurement
3. **rotate2**: Second rotation (can be 0° or cancel rotate1)
4. **drive**: Move forward

This gives the robot flexibility to:
- Measure in the same direction as driving (both rotations = 0°)
- "Look around" by rotating before measuring
- Measure in one direction but drive in another
- Make complex sensing decisions based on context

## Learning Frameworks

### Option 1: Genetic Algorithms

**Approach:** Evolve neural network weights through generations

**Advantages:**
- No need for reward engineering
- Can handle sparse rewards
- Naturally explores diverse strategies

**Challenges:**
- May require many evaluations
- Less sample-efficient than RL
- Harder to incorporate temporal structure

**Fitness Function:**
```python
def calculate_fitness(robot_trajectory):
    # Reward progress and speed
    distance_traveled = trajectory.total_distance()
    avg_speed = distance_traveled / trajectory.time
    
    # Penalize obstacle proximity
    obstacle_proximity = sum(1/d**2 for d in trajectory.obstacle_distances 
                           if d < safety_threshold)
    
    # Penalize excessive turning
    turning_cost = sum(abs(rotate1) + abs(rotate2) 
                      for rotate1, rotate2 in trajectory.rotations)
    
    # Penalize jerky movements
    smoothness = sum(abs(current_speed - prev_speed) 
                    for current, prev in zip(trajectory.speeds[1:], trajectory.speeds[:-1]))
    
    # Combined fitness score
    fitness = (distance_traveled * distance_weight +
               avg_speed * speed_weight -
               obstacle_proximity * obstacle_weight -
               turning_cost * turning_weight -
               smoothness * smoothness_weight)
    
    return fitness
```

### Option 2: Reinforcement Learning

**Approach:** Learn policy through trial-and-error with reward signals

**Advantages:**
- Can leverage temporal difference learning
- Potentially more sample-efficient
- Better at credit assignment

**Challenges:**
- Requires careful reward shaping
- May get stuck in local optima
- More complex implementation

**Reward Function:**
```python
def calculate_reward(state, action, next_state):
    # Reward progress
    progress_reward = next_state.distance - state.distance
    
    # Penalize collisions and close obstacles
    obstacle_reward = 0
    if next_state.min_obstacle_distance < safety_threshold:
        obstacle_reward = -1000  # Large collision penalty
    elif next_state.min_obstacle_distance < warning_threshold:
        obstacle_reward = -1/next_state.min_obstacle_distance
    
    # Penalize excessive turning
    smoothness_reward = -abs(action.rotate1 + action.rotate2)
    
    # Encourage movement
    speed_reward = next_state.speed * speed_factor
    
    total_reward = (progress_reward * progress_weight +
                   obstacle_reward * obstacle_weight +
                   smoothness_reward * smoothness_weight +
                   speed_reward * speed_weight)
    
    return total_reward
```

## Policy Network Architecture

```python
class ActiveSensingPolicy(nn.Module):
    def __init__(self, history_length=5, hidden_size=64):
        super().__init__()
        # Input: history of IID, distance, rotations, drives + current orientation
        input_size = history_length * 4 + 1
        
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.action_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 3)  # rotate1, rotate2, drive_distance
        )
        self.value_head = nn.Linear(hidden_size, 1)  # For RL
    
    def forward(self, x):
        # x shape: (batch, seq_len, input_size)
        lstm_out, _ = self.lstm(x)
        actions = self.action_head(lstm_out[:, -1, :])  # Use last timestep
        state_value = self.value_head(lstm_out[:, -1, :])
        
        # Scale actions to appropriate ranges
        rotate1 = torch.tanh(actions[:, 0]) * 180  # [-180°, 180°]
        rotate2 = torch.tanh(actions[:, 1]) * 180  # [-180°, 180°]
        drive = torch.sigmoid(actions[:, 2]) * max_drive_distance  # [0, max]
        
        return {
            'rotate1': rotate1,
            'rotate2': rotate2,
            'drive': drive,
            'state_value': state_value
        }
```

## Training Pipeline

### Phase 1: Emulator Validation
- Test emulator accuracy against real sonar data
- Establish confidence bounds and error characteristics
- Identify any systematic biases that need correction

### Phase 2: Simple Policy Prototyping
- Implement basic genetic algorithm version
- Test with hand-designed fitness function
- Visualize emergent behaviors
- Analyze learned rotation strategies

### Phase 3: Reinforcement Learning Implementation
- Implement PPO or A2C algorithm
- Experiment with reward shaping and hyperparameters
- Compare performance with GA approach
- Implement experience replay for sample efficiency

### Phase 4: Active Sensing Analysis
- Analyze learned measurement patterns
- Visualize sensing strategies in different environments
- Quantify benefits over passive sensing
- Compare with baseline policies (random, fixed patterns)

## Key Challenges

1. **Emulator Accuracy**: The emulator must be sufficiently accurate for meaningful learning
2. **Credit Assignment**: Long temporal dependencies in active sensing sequences
3. **Exploration vs Exploitation**: Balancing novel sensing strategies with known good behaviors
4. **Sim-to-Real Transfer**: Robustness if emulator isn't perfect
5. **Computational Efficiency**: Fast enough for practical training

## Expected Outcomes

1. **Emergent Active Sensing Behaviors**: Robot learns strategic measurement patterns
2. **Improved Navigation**: Better obstacle avoidance through informed sensing
3. **Efficient Paths**: Smooth trajectories with good progress
4. **Adaptive Strategies**: Different behaviors in different environment types
5. **Insights into Bio-inspired Navigation**: Understanding how internal models can guide active perception

## Implementation Considerations

- Start with genetic algorithms (simpler to implement)
- Use emulator for fast, parallel rollouts during training
- Implement comprehensive logging and visualization
- Design experiments to isolate active sensing benefits
- Plan for gradual complexity increase (simple → complex environments)

## Emulator Class

A new `Emulator` class has been created in `Control_code/Library/Emulator.py` that provides:

1. **Consistent Profile Parameters**: Automatically reads `profile_opening_angle` and `profile_steps` from EchoProcessor artifacts
2. **Clean Prediction Interface**: Simple `predict()` method that takes profiles and returns distance/IID predictions
3. **Batch Processing**: Supports both single and batch predictions
4. **Full Pipeline Reproduction**: Replicates the exact feature preprocessing and postprocessing from training

### Usage Example

```python
from Library.Emulator import Emulator

# Load the trained emulator
emulator = Emulator.load()

# Get profile parameters (ensures consistency with EchoProcessor)
params = emulator.get_profile_params()
print(f"Using profiles with {params['profile_steps']} steps, {params['profile_opening_angle']}° opening")

# Predict from a single profile
import numpy as np
profile = np.random.uniform(100, 2000, size=params['profile_steps'])  # Random profile
result = emulator.predict_single(profile)
print(f"Predicted: {result['distance_mm']:.1f}mm, {result['iid_db']:.1f}dB")

# Predict from batch of profiles
profiles = np.random.uniform(100, 2000, size=(10, params['profile_steps']))
batch_results = emulator.predict(profiles)
```

## Next Steps

1. **Test the Emulator**: Run `SCRIPT_TestEmulator.py` to verify it works
2. **Test the Simulator**: Run `SCRIPT_TestSimulator.py` to verify environment simulation
3. **Implement Policy Learning**: Start with genetic algorithms approach
4. **Develop Visualization**: Tools to analyze learned behaviors

## Environment Simulator

A comprehensive `EnvironmentSimulator` class has been created in `Control_code/Library/EnvironmentSimulator.py` that provides:

### Key Features

1. **Arena Layout Loading**: Loads real arena geometries from session data
2. **Profile Generation**: Computes distance profiles for arbitrary positions/orientations
3. **Sonar Emulation**: Uses the trained emulator to predict distance/IID measurements
4. **Robot Movement Simulation**: Simulates the full rotate-measure-rotate-drive action sequence
5. **Consistent Parameters**: Automatically uses profile parameters from EchoProcessor

### Core Components

```python
# ArenaLayout: Handles environment geometry
arena = ArenaLayout("sessionB01")
profile = arena.compute_profile(x, y, orientation, min_az, max_az, steps)

# EnvironmentSimulator: Full simulation
simulator = EnvironmentSimulator("sessionB01")
measurement = simulator.get_sonar_measurement(x, y, orientation)

# Movement simulation
actions = [
    {'rotate1_deg': 30, 'rotate2_deg': -30, 'drive_mm': 200},
    {'rotate1_deg': -45, 'rotate2_deg': 45, 'drive_mm': 150}
]
trajectory = simulator.simulate_robot_movement(start_x, start_y, start_orient, actions)
```

### Action Sequence Support

The simulator fully supports the active sensing action sequence:
1. **rotate1**: First rotation for measurement
2. **measure**: Get sonar prediction (distance + IID)
3. **rotate2**: Second rotation for final orientation
4. **drive**: Move forward

### Usage in Policy Learning

```python
# For policy learning, the main interface is:
measurement = simulator.get_sonar_measurement(x, y, orientation)
# Returns: {'distance_mm': float, 'iid_db': float}

# For full trajectory simulation:
trajectory = simulator.simulate_robot_movement(start_x, start_y, start_orient, actions)
# Returns list of states with positions, orientations, and measurements
```

## Policy Learning Implementation

With the simulator in place, we can now implement policy learning. The simulator provides:

- **State**: Current position, orientation, measurement history
- **Action Space**: rotate1, rotate2, drive_distance
- **Measurement Interface**: get_sonar_measurement() for observations
- **Movement Simulation**: simulate_robot_movement() for rollouts

### Genetic Algorithm Approach

```python
class PolicyNetwork(nn.Module):
    # Takes measurement history + position context
    # Outputs: rotate1, rotate2, drive_distance
    pass

class GeneticPolicyLearner:
    def __init__(self, simulator):
        self.simulator = simulator
        self.population = [PolicyNetwork() for _ in range(population_size)]
    
    def evaluate_fitness(self, policy):
        # Run policy in simulator
        trajectory = self._rollout_policy(policy)
        
        # Calculate fitness based on:
        # - Distance traveled
        # - Obstacle avoidance
        # - Path smoothness
        return fitness_score
    
    def _rollout_policy(self, policy):
        # Start at random position
        start_x, start_y, start_orient = self._get_random_start()
        
        # Run policy for N steps
        actions = []
        for step in range(max_steps):
            # Get current measurement
            measurement = self.simulator.get_sonar_measurement(x, y, orient)
            
            # Policy decides action
            action = policy.decide_action(measurement_history, position_history)
            actions.append(action)
            
            # Check for collisions/termination
            if self._check_collision(x, y, action):
                break
        
        # Simulate the trajectory
        return self.simulator.simulate_robot_movement(start_x, start_y, start_orient, actions)
```

### Reinforcement Learning Approach

```python
class RLPolicyLearner:
    def __init__(self, simulator):
        self.simulator = simulator
        self.env = GymEnvironment(simulator)
        self.agent = PPOAgent(state_space, action_space)
    
    def train(self):
        for episode in range(episodes):
            state = self.env.reset()
            done = False
            
            while not done:
                action = self.agent.choose_action(state)
                next_state, reward, done, info = self.env.step(action)
                
                # Calculate reward based on:
                # - Progress toward goal
                # - Obstacle proximity
                # - Movement efficiency
                
                self.agent.remember(state, action, reward, next_state, done)
                state = next_state
                
                if done:
                    self.agent.learn()
```

## Complete Pipeline Summary

```
Real Arena → DataProcessor → EchoProcessor Training → Emulator Training
                                      ↓
EnvironmentSimulator (uses Emulator) → Policy Learning → Robot Control
```

The simulator now provides a complete virtual environment for developing and testing navigation policies before deploying them on the real robot.

This approach represents a sophisticated integration of learned environment models with active perception strategies, potentially demonstrating how robots can use "imaginary" self-training to negotiate complex environments - much like bats might use internal models of echo acoustics.