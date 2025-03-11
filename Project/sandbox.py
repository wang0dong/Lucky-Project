"""
Author: WANG Dong
Date: 20250309
Email: wang0dong@gmail.com
"""
import pybullet as p
import pybullet_data
import time
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.vec_env import SubprocVecEnv
import gymnasium as gym
import numpy as np
from gymnasium import spaces
import os
import random

GREEN = '\033[92m'
ORANGE = '\033[38;5;208m'
RED = '\033[91m'
RESET = '\033[0m'

class RobotNavEnv(gym.Env):
    """
    Custom Gym environment for robot navigation using PyBullet.
    The robot moves within an environment, avoiding obstacles and reaching a target.
    """
    def __init__(self, render_mode = False):
        """
        Initialize the environment.
        Args:
            render_mode (bool): If True, runs PyBullet in GUI mode; otherwise, runs in headless mode.
        """        
        # self.total_reward = 0  # Cumulative reward for the episode
        # self.step_count = 0    # Number of steps taken in the episode
        self.max_steps = 3000  # Maximum step limit per episode
        self.num_envs = 1      # Number of parallel environments (default: 1)
        self.render_mode = render_mode
        # self.reward = 0.0  # Initialize step reward
        # self.done = False  # Episode not finished initially
        # self.truncated = False  # Flag for truncation        
        # self.rewards = []  # List to track rewards for the episode
        # self.dones = []  # List to track done flags
        # self.obs_list = []  # List for storing observations
        self.step_count = 0  # Counter to track steps taken in the episode

        # Initialize other necessary components (robot, environment, target, etc.)
        self.robot = None  # 
        self.target_position = [4.0, 4.0]  # Example goal position

        # Connect to PyBullet in GUI mode (if enabled) or direct mode (for performance)
        self.render_mode = render_mode
        self.physicsClient = p.connect(p.GUI if render_mode else p.DIRECT)  # GUI mode for testing

        # Debug, check if physicalClient is valid
        if self.physicsClient < 0:
            raise RuntimeError("Failed to connect to PyBullet physics server.")        

        # Set up physics properties
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setTimeStep(1./120.)  # increasing it to reduce computation. such as 60
        p.setGravity(0, 0, -9.81) # Apply gravity

        p.setPhysicsEngineParameter(enableFileCaching=0)  # Force reloading URDFs
        # p.configureDebugVisualizer(p.COV_ENABLE_WIREFRAME, 1)  # Enable wireframe visualization

        # Load robot & environment
        self.robot = self.load_robot() 
        self.plane = p.loadURDF("plane.urdf")
        self.load_walls()  

        # Enable collision filtering for the robot
        p.setCollisionFilterGroupMask(self.robot, -1, 1, 1)

        # Define action and observation spaces
        self.action_space = gym.spaces.Discrete(4)  # [Move Forward, Turn Left, Turn Right, No Action]
        self.observation_space = spaces.Box(low=-10, high=10, shape=(3,), dtype=np.float32)

    def get_observation(self):
        """
        Get the robot's current observation.
        Args:
            self: The instance of the class to which this method belongs.
        Returns:
            np.ndarray: A 1D array containing the robot's x, y position and yaw (heading).
        """ 
        if self.robot is None:
            raise AttributeError("Robot has not been initialized.")
        
        # Get the robot's position and orientation
        position, orientation = p.getBasePositionAndOrientation(self.robot)
        # Convert quaternion to euler angles for easier interpretation
        euler_angles = p.getEulerFromQuaternion(orientation)

        # For 2D navigation, we are interested in x, y, and yaw (rotation around the z-axis)
        x, y, z = position
        roll, pitch, yaw = euler_angles

        # Return the 2D position and yaw (heading) as the observation
        return np.array([x, y, yaw], dtype=np.float32)

    def reset_episode(self):
        """
        Reset episode-specific variables at the start of each new episode.
        This is typically called when starting a new episode.
        """
        # Initialize episode flags
        reward = 0.0  
        done = False  
        truncated = False  
        return reward, done, truncated 

        # # Reset other necessary variables like robot state or position, if needed
        # self.step_count = 0  # Reset step counter for the episode
        # self.total_reward = 0.0  # Reset total reward for the episode

    def step(self, action):
        """
        Takes a step in the environment by applying the given actions and calculating rewards, done flags, and observations.
        Args:
            actions (np.ndarray or scalar): The actions to apply to the robot. Can be a single action or a list of actions for each environment.
        Returns:
            obs_list (np.ndarray): A list of observations (robot's position and orientation) for each environment.
            rewards (np.ndarray): A list of rewards for each environment after the step.
            dones (np.ndarray): A list of done flags indicating whether each environment is finished.
            truncated (bool): Flag indicating whether the episode was truncated due to step count exceeding the max limit.
            info (dict): A dictionary with any additional information (currently empty).
        """
        reward = 0 # reset the reword when take action
        done = False
        truncated = False

        # # If starting a new episode, reset episode-specific variables
        # if self.done:  # If done, start a new episode
        #     self.reset_episode()

        # record the position prior the action
        prev_pos, _ = p.getBasePositionAndOrientation(self.robot)
        # Apply action (move forward, turn left, turn right)
        self.robot_control(action)
        p.stepSimulation()
        # time.sleep(0.01)  # Simulate real-time, only for visualizing  

        # Get robot position as observation
        pos, orn = p.getBasePositionAndOrientation(self.robot)
        # obs = self.get_observation()  # Get the robot's position and orientation
        obs = np.array(self.get_observation(), dtype=np.float32)


        # Check if the robot hit a wall or obstacle
        hit_penalty = self.check_collision()

        if hit_penalty:
            # print('Collision detected')
            # Apply a penalty for hitting an obstacle or wall and end episode
            reward -= 10
            done = True
        else:
            # Calculate distance to the goal
            goal = self.target_position
            distance_to_goal = np.linalg.norm(np.array([pos[0], pos[1]]) - np.array([goal[0], goal[1]]))
            # print(distance_to_goal)
            reward -= 0.1 * distance_to_goal  # negative reward for distance
            reward += 0.5  # Reward for just moving

            if distance_to_goal < 0.4:  # If robot is within 0.4 unit of the goal
                reward += 1000  # Give a large reward for reaching the goal
                done = True

            # Reward progress towards the goal
            previous_distance = np.linalg.norm(np.array([prev_pos[0], prev_pos[1]]) - np.array([goal[0], goal[1]]))
            if previous_distance > distance_to_goal:
                reward += 10  # Reward for moving closer to the goal
                
            # Reward exploration with a 10% chance
            # if np.random.rand() < 0.1:
            #     reward += 50  # Larger reward for exploration

            # # Update total reward and step count
            # self.total_reward += reward

            self.step_count += 1

            # End episode if max steps are reached (optional)
            if self.step_count >= self.max_steps:
                done = True
                truncated = True
                reward -= 50  # Optional penalty for exceeding step limit

            # print(reward)
            # self.rewards.append(self.reward)
            # self.dones.append(self.done)

        # # Ensure proper shapes for the output
        # rewards = np.array(self.rewards).squeeze() # Fix Reward Shape (1,)
        # obs_list = np.array(self.obs_list).squeeze() # Fix Observation Shape (3,)
        # dones = np.array(self.dones).squeeze() # Fix Done Shape (1,)
        # truncated = False
        
        # Debug
        # print("Observation shape:", np.shape(obs_list))
        # print("Reward shape:", np.shape(rewards))
        # print("Done shape:", np.shape(dones))  

        # return obs_list, rewards, dones, truncated, {}
        return obs, reward, done, truncated, {}

    def check_collision(self):
        """
        Checks if the robot has collided with walls or obstacles.
        Returns:
            bool: True if a collision is detected, False otherwise.
        """
        # Check for collisions with walls
        for wall_id in self.walls:
            contacts = p.getContactPoints(self.robot, wall_id)
            if contacts:
                # print(f"Collision detected with wall ID {wall_id}!")
                return True
        # Check for collisions with obstacles
        for obstacle_id in self.obstacles:
            if p.getContactPoints(self.robot, obstacle_id):
                # print(f"Collision detected with obstacle ID {obstacle_id}!")
                return True

        return False

    def reset(self, seed=None, return_info=False, options=None):
        """
        Resets the environment to its initial state. This includes resetting
        the robot's position, the target, obstacles, and other relevant state.
        Args:
            seed (int, optional): A seed for random number generation.
            return_info (bool, optional): If True, also returns additional information.
            options (dict, optional): Additional options for resetting (currently unused).
        Returns:
            np.ndarray: The initial observation of the environment after reset.
            dict: Additional information if `return_info` is True, else an empty dictionary.
        """

        # Set the random seed if provided
        if seed is not None:
            self.np_random = np.random.RandomState(seed)

        # Reset environment to its initial state at the start of a new episode
        self.total_reward = 0
        self.step_count = 0

        # Remove previous obstacles and target if they exist
        if hasattr(self, "obstacles"):
            for obs in self.obstacles:
                p.removeBody(obs)
        if hasattr(self, "target"):
            p.removeBody(self.target)

        self.obstacles = []  # Reset obstacle list

        # Reset the robot's position to the starting point
        self.reset_robot_position()

        # Create a new target
        self.target_position, self.target = self.create_target()

        # Create obstacles
        self.create_obstacles()

        # Enable Collision Filters
        for obs in self.obstacles:
            p.setCollisionFilterGroupMask(obs, -1, 1, 1)

        # Get the initial observation of the environment
        obs = self.get_observation()

        # Optionally return additional info
        if return_info:
            info = {"target_position": self.target_position, "obstacle_positions": self.obstacles}
            return obs, info  # Return both observation and info

        return obs, {} # Only return the observation if return_info is False

    def render(self, mode="human"):
        """
        Renders the environment for visualization.
        Args:
            mode (str, optional): The rendering mode. Default is "human". 
                - "human" mode is for GUI-based rendering, useful for manual testing.
                - Other modes can be added for different types of rendering if needed.        
        """
        pass  # Use GUI mode when testing manually

    def load_robot(self):
        """
        Loads the robot into the simulation and returns its object ID.
        Returns:
            int: The object ID of the loaded robot.
        """        
        # Load robot in the simulation and return the robot's object ID
        robot_id = p.loadURDF("r2d2.urdf", basePosition=[0, -2, 0.5])
        return robot_id
    
    def create_target(self):
        """
        Creates a target at a fixed location within the environment, away from the robot.
        Returns:
            tuple: A tuple containing:
                - target_position (list): The 3D position [x, y, z] of the target.
                - target (int): The object ID of the created target in the simulation.
        """
        # Random target
        # while True:
        #     x = random.uniform(-4.5, 4.5)
        #     y = random.uniform(-4.5, 4.5)
        #     if (abs(x) > 1 or abs(y) > 1):  # Avoid placing too close to the robot (0,0)
        #         break

        # Fixed target position
        target_position = [self.target_position[0], self.target_position[1], 0.5]  # Place target slightly above the ground

        # Define the visual and collision shapes for the target (sphere)
        target_radius = 0.2  # Radius of the target sphere
        visual_shape = p.createVisualShape(shapeType=p.GEOM_SPHERE, radius=target_radius, rgbaColor=[1, 215/255, 0, 1])  # Gold sphere
        collision_shape = p.createCollisionShape(shapeType=p.GEOM_SPHERE, radius=target_radius)
        
        # Create the target as a multi-body in the simulation
        target = p.createMultiBody(baseMass=0, baseVisualShapeIndex=visual_shape, baseCollisionShapeIndex=collision_shape, basePosition=target_position)

        # return target
        return target_position, target

    def robot_control(self, action):
        """
        Controls the robot's movement based on the given action.
        Args:
            action (int): The action to perform. The possible actions are:
                - 0: Move forward
                - 1: Turn left
                - 2: Turn right
                - 3: Stop
        """        
        # Define wheel groups based on your robot's joint indices:
        right_wheels = [2, 3] # Right wheels: indices 2 and 3
        left_wheels = [6, 7] # Left wheels: indices 6 and 7
        
        base_speed = 10.0  # Base forward speed (tune this value as needed)
        
        # Debug
        # if p.getConnectionInfo()['isConnected'] == 0:
        #     raise RuntimeError("PyBullet is not connected to the physics server. Ensure p.connect() is called first.")
        # for joint_index in left_wheels + right_wheels:
        #     joint_info = p.getJointInfo(self.robot, joint_index)
        #     print(f"Joint {joint_index} - Axis: {joint_info[13]}")

        # Determine the speed for the wheels based on the action
        if action == 0:  # Move forward
            left_speed = -base_speed
            right_speed = -base_speed
        elif action == 1:  # Turn left
            # Slow down left wheels, speed up right wheels
            left_speed = -base_speed * 0.5
            right_speed = -base_speed * 1.5
        elif action == 2:  # Turn right
            # Speed up left wheels, slow down right wheels
            left_speed = -base_speed * 1.5
            right_speed = -base_speed * 0.5
        else: # Stop the robot
            left_speed = 0.0
            right_speed = 0.0

        # Debug: Print speeds
        # print(f"Action: {action}, Left Speed: {left_speed}, Right Speed: {right_speed}")

        # Apply velocity control to left wheels
        for joint_index in left_wheels:
            p.setJointMotorControl2(self.robot, joint_index, controlMode=p.VELOCITY_CONTROL, targetVelocity=left_speed, force= 200.0)
        # Apply velocity control to right wheels
        for joint_index in right_wheels:
            p.setJointMotorControl2(self.robot, joint_index, controlMode=p.VELOCITY_CONTROL, targetVelocity=right_speed, force= 200.0)

        # Debug: Print actual joint vel
        # for joint_index in left_wheels + right_wheels:
        #     joint_state = p.getJointState(self.robot, joint_index)
        #     print(f"Joint {joint_index} - Velocity: {joint_state[1]}")
            
        # Step the simulation so the changes take effect
        p.stepSimulation()

    def create_obstacles(self):
        """
        Creates fixed obstacles in the environment while avoiding the robot and target positions.
        The obstacles are placed at predefined positions with different sizes.
        """
        self.obstacles = []  # Store obstacle IDs
        # Radom obstacles
        # for _ in range(5):  # You can adjust the number of obstacles
        #     while True:
        #         x = random.uniform(-4.5, 4.5)
        #         y = random.uniform(-4.5, 4.5)
        #         # Check if the obstacle is not placed too close to robot (0, 0) or target position
        #         target_position = [4, 4, 0.5]  # Example fixed target position
        #         robot_position = [0, 0, 0.5]  # Robot's initial position
        #         if (abs(x) > 1 or abs(y) > 1) and np.linalg.norm(np.array([x, y]) - np.array(robot_position[:2])) > 1.0 and np.linalg.norm(np.array([x, y]) - np.array(target_position[:2])) > 1.0:
        #             break
        # Define fixed obstacle positions (within the range of -4.5 to 4.5 for both x and y)
        obstacle_positions = [
            (2, 2),   # Position 1
            (3.5, -3),  # Position 2
            (-2, 3),  # Position 3
            (-3.5, 4),  # Position 4
            (4, -1.5),   # Position 5
            (3, 1) # Position 6
        ]
        # Define corresponding obstacle sizes (radius for spherical obstacles)
        obstacle_sizes = [
            0.4,  # Size for Position 1
            0.3,  # Size for Position 2
            0.3,  # Size for Position 3
            0.3,  # Size for Position 4
            0.3,  # Size for Position 5
            0.5   # Size for Position 6
        ]

        # Loop over each position and size to create obstacles
        for pos, size in zip(obstacle_positions, obstacle_sizes):
            x, y = pos  # Unpack the position
            # Place the obstacle in the simulation at the position (x, y) with the specified size
            obstacle_id = self.place_obstacle(x, y, size)
            self.obstacles.append(obstacle_id)  # Store the obstacle's ID for reference

    def place_obstacle(self, x, y, size):
        """
        Creates an obstacle at the specified position with the given size and returns its ID.
        Args:
            x (float): The x-coordinate for the obstacle's position.
            y (float): The y-coordinate for the obstacle's position.
            size (float): The size (radius or dimension) of the obstacle.
        Returns:
            int: The ID of the created obstacle.
        """        
        # size = random.choice([0.2, 0.3, 0.4])  # Random size for the obstacle
        # Create a visual representation of the obstacle (green cube)
        visual_shape = p.createVisualShape(shapeType=p.GEOM_BOX, halfExtents=[size, size, size], rgbaColor=[1, 0, 0, 1])  # Red cube
        # Create the collision shape for the obstacle (box with the same dimensions)
        collision_shape = p.createCollisionShape(shapeType=p.GEOM_BOX, halfExtents=[size, size, size])
        # Create the obstacle in the simulation world with the specified position and shape
        obstacle_id = p.createMultiBody(baseMass=1, baseVisualShapeIndex=visual_shape, baseCollisionShapeIndex=collision_shape, basePosition=[x, y, 0.5])

        return obstacle_id  # Return the obstacle ID to store

    def reset_robot_position(self):
        """
        Resets the robot's position to the initial starting position at (0, 0, 0.5).
        The robot's orientation is reset to a neutral orientation (no rotation).
        """        
        # Initial position and orientation for the robot
        robot_position = [0, -2, 0.5] # Set robot at the starting point (x=0, y=-2, z=0.5)
        p.resetBasePositionAndOrientation(self.robot, robot_position, [0, 0, 0, 1]) # No rotation (identity quaternion)

    def load_walls(self):
        """
        Loads walls from the `wall.urdf` file at the specified positions and orientations in the environment.
        The walls are placed at predefined positions and with predefined orientations to create boundaries.
        """
         # Define wall positions in the environment (x, y, z)
        wall_positions = [
            (0, 5, 1),   # Right wall
            (0, -5, 1),  # Left wall
            (5, 0, 1),   # Front wall
            (-5, 0, 1)   # Back wall
        ]
        # Define wall orientations as quaternions (x, y, z, w)
        wall_orientations = [
            (0, 0, 0.707, 0.707),   # No rotation (right wall)
            (0, 0,0.707, 0.707),   # No rotation (left wall)
            (0, 0, 0.707, 0.707),  # 90-degree rotation (front wall)
            (0, 0, 0.707, 0.707)   # 90-degree rotation (back wall)

        ]

        # Define wall dimensions (length, width, height)
        wall_dimensions = [
            (0.5, 10, 2),  # Right wall (length: 10, height: 2)
            (0.5, 10, 2),  # Left wall (length: 10, height: 2)
            (10, 0.5, 2),  # Front wall (length: 10, height: 2)
            (10, 0.5, 2)   # Back wall (length: 10, height: 2)
        ]

         # Initialize the list to store wall IDs
        self.walls = []

        # Loop through the positions, orientations, and dimensions to create walls
        for pos, orn, dim in zip(wall_positions, wall_orientations, wall_dimensions):
            # Create a collision shape (box) for the wall
            collision_shape = p.createCollisionShape(
                shapeType=p.GEOM_BOX,
                halfExtents=[dim[0] / 2, dim[1] / 2, dim[2] / 2]  # Half of the size for the box
            )

            # Create a visual shape (box) for the wall, with red color for visualization
            visual_shape = p.createVisualShape(
                shapeType=p.GEOM_BOX,
                halfExtents=[dim[0] / 2, dim[1] / 2, dim[2] / 2],
                rgbaColor=[0.5, 0.5, 0.5, 1]  # Grey color for visualization
            )

            # Create the wall in the simulation, using the collision and visual shapes
            wall_id = p.createMultiBody(
                baseMass=0,  # Mass = 0 to make it static
                baseVisualShapeIndex=visual_shape,
                baseCollisionShapeIndex=collision_shape,
                basePosition=pos,
                baseOrientation=orn
            )
            # Debugging
            # print(f"Loaded wall with ID: {wall_id}")  
            # Append the wall ID to the walls list
            self.walls.append(wall_id)

    def close(self):
        """
        Disconnect from the PyBullet simulation.
        This function should be called when the simulation is done to release resources.
        """        
        p.disconnect() # Disconnects from the PyBullet physics server

def make_env():
    """
    Factory function to create and return an instance of the RobotNavEnv environment.
    Returns:
        RobotNavEnv: A new instance of the RobotNavEnv environment.
    """    
    return RobotNavEnv() 

def train():
    """
    Trains a reinforcement learning agent to navigate in a robot navigation environment using the PPO algorithm.

    The function sets up the environment, initializes the PPO model, and runs the training loop. It trains the agent 
    for a set number of timesteps and tracks key metrics, including average reward, episode length, success rate, 
    and collision rate. The model is saved after each batch of timesteps.

    The following steps are performed:
    1. Initialize the environment and PPO model.
    2. For each training batch:
        - Train the model for 2000 timesteps.
        - Collect episode metrics such as rewards, episode length, success, and collisions.
        - Print and log metrics at specified intervals (every 10,000 timesteps).
        - Save the model checkpoint after each batch.
    """    
    # Set up the environment and PPO model
    env = RobotNavEnv(render_mode=True)  # Disable rendering for training
    model = PPO("MlpPolicy", env, policy_kwargs={"net_arch": [256, 256]}, verbose=1, batch_size=1024, n_steps=1024, ent_coef=0.01) # PPO model setup


    log_interval = 4096  # Print metrics every 4096 timesteps
    total_timesteps = 100000

    # Training loop with episode restarts
    timesteps_done = 0
    batch_size = 1024  # Number of steps per training batch
    while timesteps_done < total_timesteps:
        obs, _ = env.reset()  # Reset environment at the start of each episode
        done = False
        truncated = False
        episode_reward = 0
        episode_length = 0

        while not done:
            action, _ = model.predict(obs, deterministic=False)  # Choose action
            obs, reward, done, truncated, info = env.step(action)  # Take a step
            episode_reward += reward
            timesteps_done += 1

            # Restart episode when the robot reaches the goal
            if done or truncated:
                print(f"episode_reward:{episode_reward}" )
                print(f"timesteps_done:{timesteps_done}" )
                obs, _ = env.reset()
                episode_length += 1
                break  # Move to the next episode

            # Train PPO every batch_size steps
            if timesteps_done % batch_size == 0:
                model.learn(total_timesteps=batch_size)

            # Log progress
            if timesteps_done % log_interval == 0:
                print(f"Timesteps: {timesteps_done}/{total_timesteps} | "
                    f"Episode Reward: {episode_reward:.2f} | Episode Length: {episode_length}")

            # Save model periodically
            if timesteps_done % (log_interval * 5) == 0:  # Save every 10,000 steps
                model.save(f"robot_nav_ppo_checkpoint_{timesteps_done}")
                print(f"Checkpoint saved at {timesteps_done} steps")

    model.save("robot_nav_ppo_final")  # Save the final trained model
    print("Training completed!")

def test(model):
    """
    Tests a trained PPO model in the robot navigation environment.
    This function:
    1. Loads the trained PPO model from a .zip file.
    2. Initializes the robot navigation environment with GUI rendering enabled.
    3. Runs the simulation by taking actions predicted by the model.
    4. Tracks rewards and step count throughout the test.
    5. Provides debugging information such as reward at each step.
    6. Optionally handles collision and resets the environment if a collision is detected.
    7. Stops the simulation after a specified maximum number of steps to avoid infinite loops.
    Args:
        model (str): The path to the saved model file (without extension).
    """    
    if not os.path.exists(model+'.zip'):
        print("Model not found. Train the model first!")
        return

    print("Loading model...")
    model = PPO.load(model)
    print("Launching simulation...")
    env = RobotNavEnv(render_mode=True) # Enable GUI mode

    # Debug
    # for i in range(p.getNumJoints(env.robot)):
    #     joint_info = p.getJointInfo(env.robot, i)
    #     print(f"Joint {i}: {joint_info[1].decode('utf-8')}")

    obs, _ = env.reset() # Reset environment to start a new episode
    done = False
    reward = 0
    truncated = False
    total_reward = 0    
    step_count = 0  # Track number of steps for debugging

    while not done:
        # Predict the next action using the trained model
        action, _states = model.predict(obs, deterministic=False)

        # Step the environment and unpack results
        obs, reward, done, truncated, info = env.step(action)

        time.sleep(0.01)  # Simulate real-time, only for visualization purposes 
        step_count += 1
        # Debugging information for each step
        # print(f"Step {step_count} - Reward: {reward}, Done: {done}, Truncated: {truncated}")
        total_reward += reward

        # Render the environment every 10th step to reduce overhead
        if step_count % 10 == 0: 
            env.render()  # Visualize the environment (GUI will be visible)

        # Optionally add a break condition after collision
        if env.check_collision():
            print("Breaking after collision.")
            obs, _ = env.reset()  # Reset the environment and get the new observation
            total_reward = 0  # Optionally reset the total reward for the new episode
            done = False  # Reset 'done' flag to continue testing


        # Break the loop after a certain number of steps (to avoid infinite loops)
        if step_count > 30000:  # Example: break if it exceeds 30000 steps
            print("Breaking after 30000 steps.")
            done = True
            break

    print(f"Total reward: {total_reward}") # Print the total reward for the test
    env.close() # Close the environment when done

def show_world():
    """
    Initializes the Robot Navigation Environment, resets it, and renders the world for visualization.
    This function:
    1. Initializes the robot navigation environment with rendering enabled.
    2. Resets the environment to its initial state and prepares the simulation.
    3. Renders the environment (visualizing the robot, obstacles, and walls).
    4. Steps the simulation continuously to allow for real-time visualization.
    5. Runs the simulation at a rate of 240 Hz to maintain real-time physics behavior.
    This is useful for visualizing the robot's initial setup and environment in the simulation.
    """    
    # Initialize environment
    env = RobotNavEnv(True)  # Set render_mode to True to enable visualization

    # Reset the environment and visualize the world
    obs, info = env.reset()

    # Visualization step: Render the environment
    env.render()

    # Allow for some time to visualize the initial setup
    while True:
        p.stepSimulation() # Step the simulation to simulate the world
        time.sleep(1. / 240.)  # Simulate physics at 240 Hz for real-time rendering   

def main():
    """
    Main function to run the simulation. This function loads a pre-trained model and tests it in the environment.
    - It is set to test the robot navigation model, `robot_nav_ppo_checkpoint_50000`.
    - The environment will be reset, and the agent will perform actions based on the loaded model.
    """
    # Uncomment to visualize the world setup (optional)
    # show_world()

    # Uncomment to train the model (optional)
    # train()

    # Define the model checkpoint to be loaded for testing    
    model = "robot_nav_ppo_final"
 
    # Uncomment to test the model in the environment
    test(model)

if __name__ == "__main__":
    main()
