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

class RobotNavEnv(gym.Env):
    def __init__(self, render_mode = False):
        # Initialize environment variables
        self.total_reward = 0  # Initialize cumulative reward for the episode
        self.step_count = 0    # To track the number of steps taken in the episode
        self.max_steps = 3000  # Set a maximum step limit for each episode (optional)
        self.num_envs = 1
        # super(RobotNavEnv, self).__init__()
        self.render_mode = render_mode
        self.physicsClient = p.connect(p.GUI if render_mode else p.DIRECT)  # GUI mode for testing

        # Debug, check if physicalClient is valid
        if self.physicsClient < 0:
            raise RuntimeError("Failed to connect to PyBullet physics server.")        

        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setTimeStep(1./120.)  # increasing it to reduce computation. Baseline
        # p.setTimeStep(1./60.)  # increasing it to reduce computation.
        p.setGravity(0, 0, -9.81)

        p.setPhysicsEngineParameter(enableFileCaching=0)  # Force reloading URDFs

        # p.configureDebugVisualizer(p.COV_ENABLE_WIREFRAME, 1)  # Enable wireframe visualization

        # Load robot
        self.robot = self.load_robot()
        # Enable Collision Filters
        p.setCollisionFilterGroupMask(self.robot, -1, 1, 1)


        # Debug, check joint index
        '''
        num_joints = p.getNumJoints(self.robot)
        print(f"Number of joints: {num_joints}")

        for joint_index in range(num_joints):
            joint_info = p.getJointInfo(self.robot, joint_index)
            joint_name = joint_info[1].decode("utf-8")  # Convert byte string to regular string
            print(f"Joint Index: {joint_index}, Joint Name: {joint_name}")
        '''
        # Load environment
        self.plane = p.loadURDF("plane.urdf")
        self.load_walls()        

        # Action & Observation Space
        # self.action_space = spaces.Discrete(3)  # [Forward, Left, Right]
        self.action_space = gym.spaces.Discrete(4)  # 4 possible actions,
        self.observation_space = spaces.Box(low=-10, high=10, shape=(3,), dtype=np.float32)

    def get_observation(self):
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

    def step(self, actions):
        # Initialize reward and done
        # reward = 0.0
        # done = False

        # Initialize rewards and done flags for each environment
        rewards = []
        dones = []
        obs_list = []  # This will hold the observations for each environment

        if np.ndim(actions) == 0:  # Check if actions is a scalar
            actions = [actions]  # Convert it into an iterable

        # Process each action for the respective environment
        for idx, action in enumerate(actions):
            # Apply action (move forward, turn left, turn right)
            self.robot_control(action)

            p.stepSimulation()
            # time.sleep(0.01)  # Simulate real-time, only for visualizing  

            # Get robot position as observation
            pos, orn = p.getBasePositionAndOrientation(self.robot)
            obs = self.get_observation()  # Get the robot's position and orientation
            obs_list.append(obs)

            # Check if the robot hit a wall or obstacle
            hit_penalty = self.check_collision()  # Check for collisions with walls or obstacles

            # Initialize reward and done for this environment
            reward = 0.0
            done = False
            truncated = False

            if hit_penalty:
                # print('Collision detected')
                reward -= 1  # Apply a penalty for hitting an obstacle or wall
                # End episode
                done = True
            else:
                # Reward
                goal = self.target_position
                
                distance_to_goal = np.linalg.norm(np.array([pos[0], pos[1]]) - np.array([goal[0], goal[1]]))
                print(distance_to_goal)
                reward -= 1 * distance_to_goal  # negative reward for distance

                if distance_to_goal < 0.2:  # If robot is within 0.2 unit of the goal
                    reward += 1000  # Give a large reward for reaching the goal
                    # done = True  # Mark episode as done

                reward -= 1  # Small penalty for each step taken

                if np.random.rand() < 0.1:
                    reward += 50  # Larger reward for exploration

                # Update the total reward with the reward for this step
                self.total_reward += reward
                # Increment step count
                self.step_count += 1

                # End episode if max steps are reached (optional)
                if self.step_count >= self.max_steps:
                    done = True
                    truncated = True
                    reward -= 50  # Optional penalty for exceeding step limit

            rewards.append(reward)
            dones.append(done)

        # Fix Reward Shape (1,)
        rewards = np.array(rewards).squeeze()
        # Fix Observation Shape (3,)
        obs_list = np.array(obs_list).squeeze()        
        # Fix Done Shape (1,)
        dones = np.array(dones).squeeze()
        
        # Debug
        # print("Observation shape:", np.shape(obs_list))
        # print("Reward shape:", np.shape(rewards))
        # print("Done shape:", np.shape(dones))  

        return obs_list, rewards, dones, truncated, {}

    def check_collision(self):
        """
        Checks for collisions with walls or obstacles. 
        Returns True if a collision is detected, else False.
        """
        # Debug, get all contact points involving the robot
        for wall_id in self.walls:
            contacts = p.getContactPoints(self.robot, wall_id)
            if contacts:  # If there are contact points, collision is detected
                print(f"Collision detected with wall ID {wall_id}!")
                return True

        for obstacle_id in self.obstacles:  # Loop through stored obstacles
            if p.getContactPoints(self.robot, obstacle_id):
                print(f"Collision detected with obstacle ID {obstacle_id}!")
                return True

        return False

    def reset(self, seed=None, return_info=False, options=None):
        """
        Reset the environment to its initial state. This includes resetting
        the robot's position, the target, and any other relevant state.
        """
        # Set the random seed if provided
        if seed is not None:
            self.np_random = np.random.RandomState(seed)

        # Reset environment to its initial state at the start of a new episode
        self.total_reward = 0  # Reset cumulative reward at the start of a new episode
        self.step_count = 0    # Reset step count

        # Remove previous obstacles and target if they exist
        if hasattr(self, "obstacles"):
            for obs in self.obstacles:
                p.removeBody(obs)
        if hasattr(self, "target"):
            p.removeBody(self.target)

        self.obstacles = []  # Reset obstacle list

        # Reset the robot's position to the starting point
        self.reset_robot_position()

        # Create a new target at a random position (ensure it's not near the robot or previous target)
        self.target_position, self.target = self.create_target()

        # Create obstacles (ensure they don't conflict with robot or target position)
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
        pass  # Use GUI mode when testing manually

    def load_robot(self):
        # Load robot in the simulation and return the robot's object ID
        robot_id = p.loadURDF("r2d2.urdf", basePosition=[0, 0, 0.5])
        return robot_id
    
    def create_target(self):
        """
        Creates a target at a random location away from the robot and within the boundary.
        """
        # Random target
        # while True:
        #     x = random.uniform(-4.5, 4.5)
        #     y = random.uniform(-4.5, 4.5)
        #     if (abs(x) > 1 or abs(y) > 1):  # Avoid placing too close to the robot (0,0)
        #         break
        x, y = 4, 4 # fix target
        target_position = [x, y, 0.5]  # Target position

        # Define the visual and collision shapes for the target (sphere)
        target_radius = 0.2  # Radius of the target sphere
        visual_shape = p.createVisualShape(shapeType=p.GEOM_SPHERE, radius=target_radius, rgbaColor=[1, 0, 0, 1])  # Red sphere
        collision_shape = p.createCollisionShape(shapeType=p.GEOM_SPHERE, radius=target_radius)
        
        # Add the target to the simulation world
        target = p.createMultiBody(baseMass=0, baseVisualShapeIndex=visual_shape, baseCollisionShapeIndex=collision_shape, basePosition=target_position)

        # return target
        return target_position, target

    def robot_control(self, action):
        # Define wheel groups based on your robot's joint indices:
        # Right wheels: indices 2 and 3
        # Left wheels: indices 6 and 7
        right_wheels = [2, 3]
        left_wheels = [6, 7]
        
        base_speed = 10.0  # Base forward speed (tune this value as needed)
        
        # Debug
        # if p.getConnectionInfo()['isConnected'] == 0:
        #     raise RuntimeError("PyBullet is not connected to the physics server. Ensure p.connect() is called first.")
        # for joint_index in left_wheels + right_wheels:
        #     joint_info = p.getJointInfo(self.robot, joint_index)
        #     print(f"Joint {joint_index} - Axis: {joint_info[13]}")


        if action == 0:  # Move forward
            left_speed = -base_speed
            right_speed = -base_speed
        elif action == 1:  # Turn left
            # Slow down left wheels, speed up right wheels
            left_speed = -base_speed * 1.5
            right_speed = -base_speed * 0.5
        elif action == 2:  # Turn right
            # Speed up left wheels, slow down right wheels
            left_speed = -base_speed * 0.5
            right_speed = -base_speed * 1.5
        else:
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
        Creates random obstacles in the environment while avoiding the robot and target positions.
        """
        self.obstacles = []  # Store obstacle IDs
        # Radom obstacles
        '''
        for _ in range(5):  # You can adjust the number of obstacles
            while True:
                x = random.uniform(-4.5, 4.5)
                y = random.uniform(-4.5, 4.5)
                # Check if the obstacle is not placed too close to robot (0, 0) or target position
                target_position = [4, 4, 0.5]  # Example fixed target position
                robot_position = [0, 0, 0.5]  # Robot's initial position
                if (abs(x) > 1 or abs(y) > 1) and np.linalg.norm(np.array([x, y]) - np.array(robot_position[:2])) > 1.0 and np.linalg.norm(np.array([x, y]) - np.array(target_position[:2])) > 1.0:
                    break
           '''
        # Creates fixed obstacles in the environment at predefined positions
        # Define fixed obstacle positions (within the range of -4.5 to 4.5 for both x and y)
        obstacle_positions = [
            (2, 2),   # Position 1
            (3.5, -3),  # Position 2
            (-2, 3),  # Position 3
            (-3.5, 4),  # Position 4
            (4, -1.5),   # Position 5
            (3, 1) # Position 5
        ]
        obstacle_sizes = [
            0.4,   # Position 1
            0.3,  # Position 2
            0.3,  # Position 3
            0.3,  # Position 4
            0.3,   # Position 5
            0.5    # Position 6
        ]

        for pos, size in zip(obstacle_positions, obstacle_sizes):
            x, y = pos  # Unpack the position
            # Place obstacle at position (x, y, z)
            obstacle_id = self.place_obstacle(x, y, size)
            self.obstacles.append(obstacle_id)  # Store obstacle ID


    def place_obstacle(self, x, y, size):
        # Define obstacle shape and add to the world (cube or cylinder)
        # size = random.choice([0.2, 0.3, 0.4])  # Random size for the obstacle
        visual_shape = p.createVisualShape(shapeType=p.GEOM_BOX, halfExtents=[size, size, size], rgbaColor=[0, 1, 0, 1])  # Green cube
        collision_shape = p.createCollisionShape(shapeType=p.GEOM_BOX, halfExtents=[size, size, size])
        obstacle_id = p.createMultiBody(baseMass=1, baseVisualShapeIndex=visual_shape, baseCollisionShapeIndex=collision_shape, basePosition=[x, y, 0.5])

        return obstacle_id  # Return ID to store

    def reset_robot_position(self):
        # Reset robot to initial position (0, 0, 0.5)
        robot_position = [0, 0, 0.5]
        p.resetBasePositionAndOrientation(self.robot, robot_position, [0, 0, 0, 1])

    def load_walls(self):
        """
        Loads walls from the `wall.urdf` file at the specified positions and orientations.
        """
        # Define wall positions and orientations
        wall_positions = [
            (0, 5, 1),   # Right wall
            (0, -5, 1),  # Left wall
            (5, 0, 1),   # Front wall
            (-5, 0, 1)   # Back wall
        ]
        
        wall_orientations = [
            (0, 0, 0.707, 0.707),   # No rotation (right wall)
            (0, 0,0.707, 0.707),   # No rotation (left wall)
            (0, 0, 0.707, 0.707),  # 90-degree rotation (front wall)
            (0, 0, 0.707, 0.707)   # 90-degree rotation (back wall)

        ]

        # Wall dimensions (length, width, height)
        wall_dimensions = [
            (0.5, 10, 2),  # Right wall (length: 10, height: 2)
            (0.5, 10, 2),  # Left wall (length: 10, height: 2)
            (10, 0.5, 2),  # Front wall (length: 10, height: 2)
            (10, 0.5, 2)   # Back wall (length: 10, height: 2)
        ]

        # Load walls
        self.walls = []
        for pos, orn, dim in zip(wall_positions, wall_orientations, wall_dimensions):
            # Create a collision shape (box) for the wall
            collision_shape = p.createCollisionShape(
                shapeType=p.GEOM_BOX,
                halfExtents=[dim[0] / 2, dim[1] / 2, dim[2] / 2]  # Half of the size for the box
            )

            # Load the wall with the visual shape (optional) and collision shape
            visual_shape = p.createVisualShape(
                shapeType=p.GEOM_BOX,
                halfExtents=[dim[0] / 2, dim[1] / 2, dim[2] / 2],
                rgbaColor=[1, 0, 0, 1]  # Red color for visualization
            )

            # Load the wall with collision shape and visual shape, fixed base
            wall_id = p.createMultiBody(
                baseMass=0,  # Mass = 0 to make it static
                baseVisualShapeIndex=visual_shape,
                baseCollisionShapeIndex=collision_shape,
                basePosition=pos,
                baseOrientation=orn
            )

            # print(f"Loaded wall with ID: {wall_id}")  # Debugging
            # Append the wall ID to the walls list
            self.walls.append(wall_id)

    def close(self):
        p.disconnect()

def make_env():
    return RobotNavEnv() 

'''
def train():
    # env = RobotNavEnv()
    # Use 4 environments in parallel
    def make_env():
        return RobotNavEnv(render_mode=False)  # Disable rendering in each env
    num_envs = 1
    env = SubprocVecEnv([make_env for _ in range(num_envs)])

    total_rewards = []  # List to track total rewards for each episode
    episode_lengths = []  # List to track episode lengths
    success_count = 0  # To track how many times the agent reaches the goal
    collision_count = 0  # To track collisions with obstacles
    action_counts = {action: 0 for action in range(env.action_space.n)}  # For discrete action spaces


    # PPO model setup
    model = PPO("MlpPolicy", env, policy_kwargs={"net_arch": [256, 256]}, verbose=1,  batch_size=64, n_steps=1024, ent_coef=0.01) # Higher entropy coefficient to encourage exploration
    # model = PPO("MlpPolicy", env, verbose=1,  batch_size=32) # Decrease batch size # baseline
    # model = PPO("MlpPolicy", env, policy_kwargs={"net_arch": [64, 64]}, verbose=1) # smaller Neural Network
 
    total_timesteps = 20000
    log_interval = 1000  # Print metrics every 1000 timesteps

    # Epsilon decay parameters
    epsilon_start = 1.0  # Start with 100% exploration
    epsilon_end = 0.1  # End with 10% exploration
    epsilon_decay = 0.995  # Decay rate per step
    epsilon = epsilon_start  # Initialize epsilon

    for i in range(0, total_timesteps, 2000):
        model.learn(total_timesteps=2000)  # Train for 2000 timesteps

        # Collect episode metrics for the current batch of steps
        rewards, lengths, successes, collisions = [], [], 0, 0
        obs = env.reset()  # Reset environment at the beginning of each batch
        # print(f"Observation shape after reset: {obs[0].shape}")
        print(f"Observation after reset: {obs}")
        # done = [False] * num_envs  # Track done status for each environment
        done = np.array([False] * num_envs)  # Ensuring it's a numpy array

        while not np.all(done):  # Continue until all environments are done
            # actions = model.predict(obs, deterministic=False)[0]  # Get actions for all environments

            # Epsilon-greedy action selection
            if np.random.rand() < epsilon:  # Exploration
                actions = np.random.choice(env.action_space.n, num_envs)  # Select random actions
            else:  # Exploitation
                actions = model.predict(obs, deterministic=False)[0]  # Get actions using the model            

            # # Print actions and their shape
            # print("Actions:", actions)
            # print("Actions type:", type(actions))
            # print("Actions shape:", np.shape(actions))

            # If actions are scalar, wrap them in a list/array to match the number of environments
            if isinstance(actions, (np.int64, np.int32)):
                actions = np.array([actions] * num_envs)
            assert len(actions) == num_envs, f"Expected {num_envs} actions, got {len(actions)}"


            obs, rewards_batch, done, info = env.step(actions)  # Step through the environments

            # Track rewards, lengths, success, and collisions
            for idx in range(num_envs):
                # Each environment has its own reward and done flag
                reward = rewards_batch[idx]
                length = 1  # Each step is counted as part of the episode length

                # rewards.append(reward[idx])
                # lengths.append(1)  # Each step is counted as part of the episode length
                # if reward[idx] > 0:  # Assuming a positive reward means goal reached
                #     successes += 1

                # Track success
                if reward > 0:  # Assuming a positive reward means goal reached
                    successes += 1                    

                # Track collisions
                if "collision" in info[idx] and info[idx]["collision"]:  # Assuming collision info is in the environment info
                    collisions += 1

                # Accumulate rewards and lengths
                rewards.append(reward)
                lengths.append(length)

        # Update metrics after the batch
        total_rewards.append(np.sum(rewards))
        episode_lengths.append(np.sum(lengths))

        # Update epsilon after each batch (decay it)
        epsilon = max(epsilon_end, epsilon * epsilon_decay)

        # Print metrics at specified intervals
        if i % log_interval == 0:
            avg_reward = np.mean(total_rewards[-log_interval:])  # Average reward in the last 'log_interval' episodes
            avg_episode_length = np.mean(episode_lengths[-log_interval:])
            success_rate = successes / (i + 1)
            avg_collision_rate = collisions / (i + 1)

            print(f"Episode {i+2000}/{total_timesteps} - "
                  f"Avg Reward: {avg_reward:.2f}, "
                  f"Avg Episode Length: {avg_episode_length:.2f}, "
                  f"Success Rate: {success_rate:.2f}, "
                  f"Collision Rate: {avg_collision_rate:.2f}")

        # Save model checkpoint after each batch
        model.save(f"robot_nav_ppo_checkpoint_{i + 2000}")  # Save model checkpoint

    print("Training completed!")
'''

def train():
    # Use a single environment instead of multiple
    env = RobotNavEnv(render_mode=True)  # Disable rendering for training

    total_rewards = []  # List to track total rewards for each episode
    episode_lengths = []  # List to track episode lengths
    success_count = 0  # To track how many times the agent reaches the goal
    collision_count = 0  # To track collisions with obstacles
    action_counts = {action: 0 for action in range(env.action_space.n)}  # For discrete action spaces

    # PPO model setup
    model = PPO("MlpPolicy", env, policy_kwargs={"net_arch": [256, 256]}, verbose=1, batch_size=64, n_steps=1024, ent_coef=0.01)

    total_timesteps = 200000
    log_interval = 10000  # Print metrics every 1000 timesteps

    for i in range(0, total_timesteps, 2000):
        model.learn(total_timesteps=2000)  # Train for 2000 timesteps

        # Collect episode metrics for the current batch of steps
        rewards, lengths, successes, collisions = [], [], 0, 0
        obs, _ = env.reset()  # Reset environment at the beginning of each batch
        print(f"Observation shape after reset: {obs[0].shape}")

        done = False  # Track done status for the environment

        while not done:  # Continue until the environment is done
            actions = model.predict(obs, deterministic=False)[0]  # Get action for the environment

            obs, rewards_batch, done, truncated, info = env.step(actions)  # Step through the environment

            # Track rewards, lengths, success, and collisions
            if rewards_batch > 0:  # Assuming a positive reward means goal reached
                successes += 1

            # Track collisions
            if "collision" in info and info["collision"]:  # Assuming collision info is in the environment info
                collisions += 1

            # Accumulate rewards and lengths
            rewards.append(rewards_batch)
            lengths.append(1)  # Each step is counted as part of the episode length

        # Update metrics after the batch
        total_rewards.append(np.sum(rewards))
        episode_lengths.append(np.sum(lengths))

        # Print metrics at specified intervals
        if i % log_interval == 0:
            avg_reward = np.mean(total_rewards[-log_interval:])  # Average reward in the last 'log_interval' episodes
            avg_episode_length = np.mean(episode_lengths[-log_interval:])
            success_rate = successes / (i + 1)
            avg_collision_rate = collisions / (i + 1)

            print(f"Episode {i + 2000}/{total_timesteps} - "
                  f"Avg Reward: {avg_reward:.2f}, "
                  f"Avg Episode Length: {avg_episode_length:.2f}, "
                  f"Success Rate: {success_rate:.2f}, "
                  f"Collision Rate: {avg_collision_rate:.2f}")

        # Save model checkpoint after each batch
        model.save(f"robot_nav_ppo_checkpoint_{i + 2000}")  # Save model checkpoint

    print("Training completed!")

def test(model):
    if not os.path.exists(model+'.zip'):
        print("Model not found. Train the model first!")
        return

    print("Loading model...")
    model = PPO.load(model)
    print("Launching simulation...")
    env = RobotNavEnv(render_mode=True) # Enable GUI mode

    # for i in range(p.getNumJoints(env.robot)):
    #     joint_info = p.getJointInfo(env.robot, i)
    #     print(f"Joint {i}: {joint_info[1].decode('utf-8')}")

    obs, _ = env.reset()
    done = False
    total_reward = 0    
    step_count = 0  # Track number of steps for debugging
    while not done:
        action, _states = model.predict(obs, deterministic=False)  # Allow exploration by setting deterministic=False
        # Step the environment and unpack
        obs, reward, done, truncated, info = env.step(action)
        time.sleep(0.01)  # Simulate real-time, only for visualizing  
        step_count += 1
        print(f"Step {step_count} - Reward: {reward}, Done: {done}, Truncated: {truncated}")  # Debugging line
        total_reward += reward
        if step_count % 10 == 0:  # Render every 10th step to reduce overhead (example)
            env.render()  # Make sure to render the environment (GUI will be visible)

        # Optionally add a break condition after collision
        if env.check_collision():
            print("Breaking after collision.")
            obs, _ = env.reset()  # Reset the environment and get the new observation
            total_reward = 0  # Optionally reset the total reward for the new episode
            done = False  # Reset 'done' flag to continue testing


        # Optionally add a break condition to avoid infinite loops
        if step_count > 3000:  # Example: break if it exceeds 1000 steps
            print("Breaking after 3000 steps.")
            done = True
            break

    print(f"Total reward: {total_reward}")
    env.close()

def show_world():
    # Initialize environment
    env = RobotNavEnv(True)

    # Reset the environment and visualize the world
    obs = env.reset()

    # Visualization step: Render the environment
    env.render()

    # Debugging
    # for wall_id in env.walls:
    #     pos, orn = p.getBasePositionAndOrientation(wall_id)
    #     print(f"Wall {wall_id} Position: {pos}, Orientation: {orn}")

    # Allow for some time to visualize the initial setup
    while True:
        p.stepSimulation()
        time.sleep(1. / 240.)  # Simulate physics at 240 Hz   
    '''
    # Take random actions for a few steps
    for _ in range(100):  # Step for 100 iterations
        action = env.action_space.sample()  # Random action
        obs, reward, done, truncated, info = env.step(action)
        env.render()  # Continuously render as the agent moves

    env.close()
    '''
def main():
    # show_world()
    # train()
    model = "robot_nav_ppo_checkpoint_50000"
    test(model)

    # Debug
    '''
    env = RobotNavEnv(use_gui=True) # Enable GUI mode
    obs, _ = env.reset()
    # Set forward velocity for all wheels
    for _ in range(2000):
        for joint_index in [2, 3, 6, 7]:
            p.setJointMotorControl2(env.robot, joint_index, controlMode=p.VELOCITY_CONTROL, targetVelocity=10.0)
        p.stepSimulation()    
        env.render()
        time.sleep(0.05)    
    env.close()
    '''
    # env = RobotNavEnv(use_gui=True) # Enable GUI mode
    # robot_collision_shape = p.getCollisionShapeData(env.robot, -1)
    # print(f"Robot collision shape: {robot_collision_shape}")
    # env.close()

if __name__ == "__main__":
    main()
