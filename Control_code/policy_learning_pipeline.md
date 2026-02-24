# Active Sensing Policy Learning Pipeline

## Summary and Rationale

This document outlines the complete pipeline for developing a policy learning system that enables a robot to actively sense its environment and navigate obstacles using learned IID (Interaural Intensity Difference) and distance measurements.

### Core Problem and Motivation

The current sonar system is fixed to the robot's orientation, limiting its ability to gather information about the environment. By allowing the robot to rotate, measure, rotate again, and then drive, we enable **active sensing** - the robot can choose where to "look" before deciding where to go.

### Biological Inspiration

This approach is bio-inspired by how bats might use internal models of echo acoustics to plan movement. The emulator network serves as the robot's "imagination" of how the world would sound from different positions and orientations.

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

## Pipeline Stages

### Stage 1: EchoProcessor Training (SCRIPT_TrainEchoProcessor.py)

**Purpose:** Train a CNN to predict distance from sonar echoes and compute IID from predicted distance.

#### Key Components:
- **DistanceCNN Model**: 1D CNN with 3 conv layers
- **IID Computation**: Extracts IID from echo window around predicted distance
- **Training**: Huber loss, linear calibration, early stopping
- **Evaluation**: RMSE, MAE, correlation metrics, IID sign accuracy

#### Outputs:
- `best_model_pytorch.pth` - Trained distance prediction model
- `echoprocessor_artifacts.pth` - Portable inference artifact
- `training_params.json` - Configuration and metrics
- Visualizations: training curves, scatter plots

### Stage 2: Emulator Training (SCRIPT_TrainEmulator.py)

**Purpose:** Train an MLP to emulate EchoProcessor using profile data as input.

#### Key Components:
- **ProfileMLP Model**: Multi-layer perceptron with dual output heads
- **Feature Augmentation**: Raw bins, asymmetry, slope, center-of-mass
- **Training**: Multi-task learning with separate loss weights
- **Evaluation**: Distance and IID metrics, sign confusion analysis

#### Outputs:
- `best_model_pytorch.pth` - Environment emulator model
- `training_params.json` - Configuration and metrics
- Visualizations: training curves, test scatter plots

### Stage 3: Environment Simulation

**Purpose:** Create a virtual environment for policy learning using the trained emulator.

#### Key Components:
- **ArenaLayout**: Loads real arena geometries, computes distance profiles
- **EnvironmentSimulator**: Combines arena layout with sonar emulator
- **Action Sequence Support**: Full rotate-measure-rotate-drive simulation
- **Measurement Interface**: `get_sonar_measurement()` for policy observations

#### Capabilities:
- Virtual environment for safe policy testing
- Trajectory simulation with collision detection
- Measurement history tracking
- Arena boundary enforcement

### Stage 4: Policy Learning (Current Focus)

**Purpose:** Train neural network policies to control the robot's active sensing behavior.

#### Learning Frameworks:

##### Option 1: Genetic Algorithms

**Approach:** Evolve neural network weights through generations

**Advantages:**
- No need for reward engineering
- Can handle sparse rewards  
- Naturally explores diverse strategies

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

##### Option 2: Reinforcement Learning

**Approach:** Learn policy through trial-and-error with reward signals

**Advantages:**
- Can leverage temporal difference learning
- Potentially more sample-efficient
- Better at credit assignment

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

#### Policy Network Architecture:

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

#### Training Pipeline:
- Use EnvironmentSimulator for fast rollouts
- Parallel evaluation of multiple policies
- Comprehensive logging and visualization
- Gradual complexity increase (simple → complex environments)

#### Expected Outputs:
- Trained policy networks
- Performance metrics and visualizations
- Analysis of learned sensing strategies
- Comparison with baseline policies

## Complete Pipeline Flow

```
Real Arena → DataProcessor → EchoProcessor Training → Emulator Training
                                      ↓
                                 EnvironmentSimulator → Policy Learning → Robot Control
```

## Current Status

✅ **Completed:**
- EchoProcessor training and validation
- Emulator training and testing  
- Environment simulator implementation
- Basic simulator testing

🚧 **In Progress:**
- Policy learning implementation
- Genetic algorithm framework
- Reinforcement learning framework
- Policy network architecture

📋 **Next Steps:**
1. Implement genetic algorithm policy learner
2. Implement reinforcement learning policy learner
3. Develop fitness/reward functions
4. Create visualization tools for policy analysis
5. Test policies in simulation
6. Deploy best policies to real robot

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

## Configuration

All stages use consistent configuration patterns:
- Session selection and data splitting
- Normalization options (sonar/profile and target)
- Training hyperparameters (batch size, epochs, learning rate)
- Regularization (L2, dropout)
- Early stopping with patience
- Comprehensive logging of all parameters and metrics

The pipeline is designed for reproducibility with seeded random operations and gradual complexity increase.