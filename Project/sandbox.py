import pybullet as p
import pybullet_data
import time
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
import gymnasium as gym
import numpy as np
from gymnasium import spaces
import os
import random

class RobotNavEnv(gym.Env):
    def __init__(self, use_gui = False):
        super(RobotNavEnv, self).__init__()
        
        self.physicsClient = p.connect(p.GUI if use_gui else p.DIRECT)  # GUI mode for testing
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setTimeStep(1./120.)  # increasing it to reduce computation.
        p.setGravity(0, 0, -9.81)

        # Load robot
        self.robot = self.load_robot()
        # Load environment
        self.plane = p.loadURDF("plane.urdf")
        # self.robot = p.loadURDF("r2d2.urdf", [0, 0, 0.5])
        # self.wall = None # Initialize wall object
        # Load walls
        self.load_walls()        

        # Action & Observation Space
        self.action_space = spaces.Discrete(3)  # [Forward, Left, Right]
        self.observation_space = spaces.Box(low=-10, high=10, shape=(3,), dtype=np.float32)

    def step(self, action):
        if action == 0:  # Move forward
            p.applyExternalForce(self.robot, -1, [10, 0, 0], [0, 0, 0], p.WORLD_FRAME)
        elif action == 1:  # Turn left
            p.applyExternalTorque(self.robot, -1, [0, 0, 5], p.WORLD_FRAME)
        elif action == 2:  # Turn right
            p.applyExternalTorque(self.robot, -1, [0, 0, -5], p.WORLD_FRAME)

        p.stepSimulation()
        time.sleep(0.05)  # Simulate real-time

        # Get robot position as observation
        pos, _ = p.getBasePositionAndOrientation(self.robot)
        obs = np.array([pos[0], pos[1], 0], dtype=np.float32)

        reward = 1.0  # Reward for moving
        done = False  # End condition

        return obs, reward, done, {}

    '''
    def reset(self, seed=None, options=None):
        p.resetSimulation()

        # Reload environment, walls, and robot
        self.plane = p.loadURDF("plane.urdf")
        self.robot = p.loadURDF("r2d2.urdf", [0, 0, 0.5])

        # Create a new target
        self.target = self.create_target()

        # Load walls again in reset
        if self.wall is None:  # Load wall only once
            # Define wall positions and orientations
            wall_positions = [
                (0, 5, 1),   # Right wall
                (0, -5, 1),  # Left wall
                (5, 0, 1),   # Front wall
                (-5, 0, 1)   # Back wall
            ]
            
            wall_orientations = [
                (0, 0, 0, 1),   # No rotation (right wall)
                (0, 0, 0, 1),   # No rotation (left wall)
                (0, 0, 0.707, 0.707),  # 90-degree rotation (front wall)
                (0, 0, 0.707, 0.707)   # 90-degree rotation (back wall)
            ]
            
            # Load walls
            self.walls = []
            for pos, orn in zip(wall_positions, wall_orientations):
                wall_id = p.loadURDF("wall.urdf", pos, orn)
                self.walls.append(wall_id)

            # Load random objects inside the boundary
            num_objects = 5  # Change this to add more or fewer objects
            self.objects = []
            
            for _ in range(num_objects):
                obj_type = random.choice(["cube", "cylinder"])

                # Ensure objects do not spawn too close to (0,0)
                while True:
                    x = random.uniform(-4.5, 4.5)
                    y = random.uniform(-4.5, 4.5)
                    if abs(x) > 1 or abs(y) > 1:  # Keep a safe zone around (0,0) robot inital pos and target pos
                        break

                z = 0.5  # Object height

                if obj_type == "cube":
                    obj_id = p.loadURDF("cube.urdf", [x, y, z])
                else:
                    obj_id = p.loadURDF("cylinder.urdf", [x, y, z])

                self.objects.append(obj_id)

        pos, _ = p.getBasePositionAndOrientation(self.robot)
        return np.array([pos[0], pos[1], 0], dtype=np.float32), {}
    '''
    def reset(self):
        """
        Reset the environment to its initial state. This includes resetting
        the robot's position, the target, and any other relevant state.
        """
        # Reset the robot's position to the starting point
        self.reset_robot_position()

        # Create a new target at a random position (ensure it's not near the robot or previous target)
        self.target = self.create_target()

        # Create obstacles (ensure they don't conflict with robot or target position)
        self.create_obstacles()

        # Return the initial observation
        return self.get_observation()

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
        while True:
            x = random.uniform(-4.5, 4.5)
            y = random.uniform(-4.5, 4.5)
            if (abs(x) > 1 or abs(y) > 1):  # Avoid placing too close to the robot (0,0)
                break
        
        target_position = [x, y, 0.5]  # Target position

        # Define the visual and collision shapes for the target (sphere)
        target_radius = 0.2  # Radius of the target sphere
        visual_shape = p.createVisualShape(shapeType=p.GEOM_SPHERE, radius=target_radius, rgbaColor=[1, 0, 0, 1])  # Red sphere
        collision_shape = p.createCollisionShape(shapeType=p.GEOM_SPHERE, radius=target_radius)
        
        # Add the target to the simulation world
        target = p.createMultiBody(baseMass=0, baseVisualShapeIndex=visual_shape, baseCollisionShapeIndex=collision_shape, basePosition=target_position)

        return target
    '''
    def create_target(self):
        # Set the target position in the world
        target_position = [4, 4, 0.5]  # position for the target (x, y, z), fix target
        
        # Load the target object as a URDF (can use a simple sphere)
        target_radius = 0.2  # Radius of the target sphere
        visual_shape = p.createVisualShape(shapeType=p.GEOM_SPHERE, radius=target_radius, rgbaColor=[1, 0, 0, 1])  # Red sphere
        collision_shape = p.createCollisionShape(shapeType=p.GEOM_SPHERE, radius=target_radius)
        
        # Add the target to the simulation world
        target = p.createMultiBody(baseMass=0, baseVisualShapeIndex=visual_shape, baseCollisionShapeIndex=collision_shape, basePosition=target_position)
        
        return target
    '''    
    def create_obstacles(self):
        """
        Creates random obstacles in the environment while avoiding the robot and target positions.
        """
        for _ in range(5):  # You can adjust the number of obstacles
            while True:
                x = random.uniform(-4.5, 4.5)
                y = random.uniform(-4.5, 4.5)
                # Check if the obstacle is not placed too close to robot (0, 0) or target position
                target_position = [4, 4, 0.5]  # Example fixed target position
                robot_position = [0, 0, 0.5]  # Robot's initial position
                if (abs(x) > 1 or abs(y) > 1) and np.linalg.norm(np.array([x, y]) - np.array(robot_position[:2])) > 1.0 and np.linalg.norm(np.array([x, y]) - np.array(target_position[:2])) > 1.0:
                    break
            
            # Place obstacle at position (x, y, z)
            self.place_obstacle(x, y)

    def place_obstacle(self, x, y):
        # Define obstacle shape and add to the world (cube or cylinder)
        size = random.choice([0.2, 0.3, 0.4])  # Random size for the obstacle
        visual_shape = p.createVisualShape(shapeType=p.GEOM_BOX, halfExtents=[size, size, size], rgbaColor=[0, 1, 0, 1])  # Green cube
        collision_shape = p.createCollisionShape(shapeType=p.GEOM_BOX, halfExtents=[size, size, size])
        p.createMultiBody(baseMass=0, baseVisualShapeIndex=visual_shape, baseCollisionShapeIndex=collision_shape, basePosition=[x, y, 0.5])

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
            (0, 0, 0, 1),   # No rotation (right wall)
            (0, 0, 0, 1),   # No rotation (left wall)
            (0, 0, 0.707, 0.707),  # 90-degree rotation (front wall)
            (0, 0, 0.707, 0.707)   # 90-degree rotation (back wall)
        ]
        
        # Load walls
        self.walls = []
        for pos, orn in zip(wall_positions, wall_orientations):
            wall_id = p.loadURDF("wall.urdf", pos, orn)
            self.walls.append(wall_id)

    def close(self):
        p.disconnect()

def make_env():
    return RobotNavEnv() 

def train():
    env = RobotNavEnv()
    # Use 4 environments in parallel
    # env = DummyVecEnv([make_env for _ in range(4)])
    model = PPO("MlpPolicy", env, verbose=1,  batch_size=32) # Decrease batch size
 
    total_timesteps = 100000
    for i in range(0, total_timesteps, 20000):
        model.learn(total_timesteps=20000)  # Train for 20,000 timesteps
        model.save(f"robot_nav_ppo_checkpoint_{i + 20000}")  # Save model checkpoint

    model.save("robot_nav_ppo")
    print("Training completed!")

def test():
    if not os.path.exists("robot_nav_ppo.zip"):
        print("Model not found. Train the model first!")
        return

    print("Loading model...")
    model = PPO.load("robot_nav_ppo")
    print("Launching simulation...")
    env = RobotNavEnv(use_gui=True) # Enable GUI mode

    obs, _ = env.reset()
    for _ in range(200):
        action, _ = model.predict(obs)
        obs, _, done, _, _ = env.step(action)
        env.render()
        time.sleep(0.05)

    print("Test completed.")
    env.close()

def show_world():
    # Initialize environment
    env = RobotNavEnv(True)

    # Reset the environment and visualize the world
    obs = env.reset()
    
    # Visualization step: Render the environment
    env.render()

    # Allow for some time to visualize the initial setup
    time.sleep(20)  # You can adjust the sleep time based on how long you want to visualize the world
    
    # Take random actions for a few steps
    for _ in range(100):  # Step for 100 iterations
        action = env.action_space.sample()  # Random action
        obs, reward, done, info = env.step(action)
        env.render()  # Continuously render as the agent moves

    env.close()

def main():
    show_world()
    # train()
    # test()

if __name__ == "__main__":
    main()
