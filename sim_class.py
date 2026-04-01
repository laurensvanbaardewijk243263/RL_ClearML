"""
Simulation module (PyBullet-based OT-2 twin)

Purpose
-------
This file defines the simulation environment used for the pipette system.

It:
1. Initializes PyBullet physics
2. Loads OT-2 robot(s)
3. Loads a specimen (plate) with a texture image
4. Applies velocity control to the robot joints
5. Simulates droplet release and landing
6. Provides state information (pipette position, etc.)

Key idea
--------
The simulation acts as a "digital twin" of the OT-2 system:
- Robot motion is controlled via velocities
- The plate has known geometry (used for calibration)
- Droplets are simulated as small spheres

Used by:
- pid.py (via SimAdapter)
- pipeline.py (for calibration + control)
"""

import pybullet as p
import time
import pybullet_data
import math
import logging
import os
from typing import Optional, Tuple

# =========================================================
# MAIN SIMULATION CLASS
# =========================================================
class Simulation:
    def __init__(self, num_agents, render=True, rgb_array=False, texture_path: Optional[str] = None):
        """
        Initialize the simulation environment.

        Parameters
        ----------
        num_agents : int
            Number of robots to spawn (usually 1)
        render : bool
            Whether to show GUI (True) or run headless (False)
        texture_path : str (optional)
            Path to image used as plate texture
        """

        self.render = render
        self.rgb_array = rgb_array

        # Plate dimensions (must match custom.urdf!)
        self.specimen_size = (0.15, 0.15, 0.015)  # meters (x, y, z)

        # Offset of plate relative to robot base
        self.specimen_offset = [0.18275 - 0.00005, 0.163 - 0.026, 0.057]

        # =========================================================
        # START PHYSICS ENGINE
        # =========================================================
        if render:
            mode = p.GUI
        else:
            mode = p.DIRECT

        self.physicsClient = p.connect(mode)
        p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -10)

        # =========================================================
        # LOAD TEXTURE (PLATE IMAGE)
        # =========================================================
        self.texture_path = self._resolve_texture_path(texture_path)
        self.textureId = p.loadTexture(self.texture_path)

        # =========================================================
        # CAMERA SETTINGS
        # =========================================================
        cameraDistance = 1.1 * (math.ceil((num_agents) ** 0.3))
        cameraYaw = 90
        cameraPitch = -35
        cameraTargetPosition = [-0.2, -(math.ceil(num_agents ** 0.5) / 2) + 0.5, 0.1]
        p.resetDebugVisualizerCamera(cameraDistance, cameraYaw, cameraPitch, cameraTargetPosition)

        # =========================================================
        # LOAD BASE PLANE
        # =========================================================
        self.baseplaneId = p.loadURDF("plane.urdf")

        # Offset from robot base to pipette tip
        self.pipette_offset = [0.073, 0.0895, 0.0895]

        # Storage
        self.pipette_positions = {}

        # Create robot(s) + specimen(s)
        self.create_robots(num_agents)
        self.sphereIds = []
        self.droplet_positions = {}

    # =========================================================
    # TEXTURE HANDLING
    # =========================================================
    def _resolve_texture_path(self, texture_path: Optional[str]) -> str:
        """
        Determine which texture image to use.

        Priority:
        1. User-provided path
        2. First image in ./textures folder
        """

        if texture_path:
            return texture_path
        texture_dir = "textures"
        candidates = []
        if os.path.isdir(texture_dir):
            for name in sorted(os.listdir(texture_dir)):
                full = os.path.join(texture_dir, name)
                if os.path.isfile(full) and name.lower().endswith((".png", ".jpg", ".jpeg", ".bmp")):
                    candidates.append(full)
        if not candidates:
            raise FileNotFoundError(
                "No texture image found. Pass texture_path explicitly or place an image in ./textures"
            )
        return candidates[0]

    # =========================================================
    # ROBOT + PLATE CREATION
    # =========================================================
    def create_robots(self, num_agents):
        """
        Create robots and their corresponding specimen plates.
        """

        spacing = 1
        grid_size = math.ceil(num_agents ** 0.5)

        self.robotIds = []
        self.specimenIds = []
        agent_count = 0

        for i in range(grid_size):
            for j in range(grid_size):
                if agent_count < num_agents:
                    # --- Robot placement ---

                    position = [-spacing * i, -spacing * j, 0.03]
                    robotId = p.loadURDF(
                        "ot_2_simulation_v6.urdf",
                        position,
                        [0, 0, 0, 1],
                        flags=p.URDF_USE_INERTIA_FROM_FILE,
                    )

                    # Fix robot in space
                    start_position, start_orientation = p.getBasePositionAndOrientation(robotId)
                    p.createConstraint(
                        parentBodyUniqueId=robotId,
                        parentLinkIndex=-1,
                        childBodyUniqueId=-1,
                        childLinkIndex=-1,
                        jointType=p.JOINT_FIXED,
                        jointAxis=[0, 0, 0],
                        parentFramePosition=[0, 0, 0],
                        childFramePosition=start_position,
                        childFrameOrientation=start_orientation,
                    )

                    # --- Plate placement ---
                    offset = self.specimen_offset
                    position_with_offset = [
                        position[0] + offset[0],
                        position[1] + offset[1],
                        position[2] + offset[2],
                    ]
                    rotate_90 = p.getQuaternionFromEuler([0, 0, -math.pi / 2])
                    planeId = p.loadURDF("custom.urdf", position_with_offset, rotate_90)

                    # Disable collision between robot and plate
                    p.setCollisionFilterPair(robotId, planeId, -1, -1, enableCollision=0)

                    # Fix plate in space
                    spec_position, spec_orientation = p.getBasePositionAndOrientation(planeId)
                    p.createConstraint(
                        parentBodyUniqueId=planeId,
                        parentLinkIndex=-1,
                        childBodyUniqueId=-1,
                        childLinkIndex=-1,
                        jointType=p.JOINT_FIXED,
                        jointAxis=[0, 0, 0],
                        parentFramePosition=[0, 0, 0],
                        childFramePosition=spec_position,
                        childFrameOrientation=spec_orientation,
                    )
                    # Apply texture image
                    p.changeVisualShape(planeId, -1, textureUniqueId=self.textureId)

                    self.robotIds.append(robotId)
                    self.specimenIds.append(planeId)
                    agent_count += 1

                    pipette_position = self.get_pipette_position(robotId)
                    self.pipette_positions[f"robotId_{robotId}"] = pipette_position

    # =========================================================
    # POSITION COMPUTATION
    # =========================================================
    def get_pipette_position(self, robotId):
        """
        Compute actual pipette tip position from:
        - robot base position
        - joint states
        - fixed offset
        """

        robot_position = list(p.getBasePositionAndOrientation(robotId)[0])
        joint_states = p.getJointStates(robotId, [0, 1, 2])
        robot_position[0] -= joint_states[0][0]
        robot_position[1] -= joint_states[1][0]
        robot_position[2] += joint_states[2][0]
        x_offset, y_offset, z_offset = self.pipette_offset
        return [
            robot_position[0] + x_offset,
            robot_position[1] + y_offset,
            robot_position[2] + z_offset,
        ]

    def get_plate_bounds_xy(self, specimen_index: int = 0) -> Tuple[float, float, float, float]:
        specimen_id = self.specimenIds[specimen_index]
        cx, cy, _ = p.getBasePositionAndOrientation(specimen_id)[0]
        sx, sy, _ = self.specimen_size
        xmin = cx - sx / 2.0
        xmax = cx + sx / 2.0
        ymin = cy - sy / 2.0
        ymax = cy + sy / 2.0
        return xmin, xmax, ymin, ymax

    def get_plate_top_z(self, specimen_index: int = 0) -> float:
        specimen_id = self.specimenIds[specimen_index]
        _, _, cz = p.getBasePositionAndOrientation(specimen_id)[0]
        return cz + self.specimen_size[2] / 2.0

    def get_texture_path(self) -> str:
        return self.texture_path

    def reset(self, num_agents=1):
        for specimenId in list(self.specimenIds):
            p.changeVisualShape(specimenId, -1, textureUniqueId=-1)
        for robotId in list(self.robotIds):
            p.removeBody(robotId)
            self.robotIds.remove(robotId)
        for specimenId in list(self.specimenIds):
            p.removeBody(specimenId)
            self.specimenIds.remove(specimenId)
        for sphereId in list(self.sphereIds):
            p.removeBody(sphereId)
            self.sphereIds.remove(sphereId)

        self.pipette_positions = {}
        self.sphereIds = []
        self.droplet_positions = {}
        self.create_robots(num_agents)
        return self.get_states()

    # =========================================================
    # SIMULATION STEP
    # =========================================================
    def run(self, actions, num_steps=1):
        """
        Run simulation forward.

        actions = [[vx, vy, vz, drop_flag]]
        """

        for _ in range(num_steps):
            self.apply_actions(actions)
            p.stepSimulation()

            # Check droplet contact
            for specimenId, robotId in zip(self.specimenIds, self.robotIds):
                self.check_contact(robotId, specimenId)
            if self.render:
                time.sleep(1.0 / 240.0)
        return self.get_states()

    # =========================================================
    # APPLY CONTROL INPUT
    # =========================================================
    def apply_actions(self, actions):
        """
        Apply velocity commands to robot joints.
        """

        for i in range(len(self.robotIds)):
            p.setJointMotorControl2(self.robotIds[i], 0, p.VELOCITY_CONTROL, targetVelocity=-actions[i][0], force=500)
            p.setJointMotorControl2(self.robotIds[i], 1, p.VELOCITY_CONTROL, targetVelocity=-actions[i][1], force=500)
            p.setJointMotorControl2(self.robotIds[i], 2, p.VELOCITY_CONTROL, targetVelocity=actions[i][2], force=800)
            if actions[i][3] == 1:
                self.drop(robotId=self.robotIds[i])

    # =========================================================
    # DROPLET SIMULATION
    # =========================================================
    def drop(self, robotId):
        """
        Spawn a droplet at pipette tip.
        """

        robot_position = list(p.getBasePositionAndOrientation(robotId)[0])
        joint_states = p.getJointStates(robotId, [0, 1, 2])
        robot_position[0] -= joint_states[0][0]
        robot_position[1] -= joint_states[1][0]
        robot_position[2] += joint_states[2][0]
        x_offset = self.pipette_offset[0]
        y_offset = self.pipette_offset[1]
        z_offset = self.pipette_offset[2] - 0.0015

        sphereRadius = 0.003
        sphereColor = [1, 0, 0, 0.5]
        visualShapeId = p.createVisualShape(shapeType=p.GEOM_SPHERE, radius=sphereRadius, rgbaColor=sphereColor)
        collision = p.createCollisionShape(shapeType=p.GEOM_SPHERE, radius=sphereRadius)
        sphereBody = p.createMultiBody(baseMass=0.1, baseVisualShapeIndex=visualShapeId, baseCollisionShapeIndex=collision)
        droplet_position = [
            robot_position[0] + x_offset,
            robot_position[1] + y_offset,
            robot_position[2] + z_offset,
        ]
        p.resetBasePositionAndOrientation(sphereBody, droplet_position, [0, 0, 0, 1])
        self.sphereIds.append(sphereBody)
        self.dropped = True
        return droplet_position

    # =========================================================
    # STATE RETRIEVAL
    # =========================================================
    def get_states(self):
        """
        Return current robot + pipette states.
        """

        states = {}
        for robotId in self.robotIds:
            raw_joint_states = p.getJointStates(robotId, [0, 1, 2])
            joint_states = {}
            for i, joint_state in enumerate(raw_joint_states):
                joint_states[f"joint_{i}"] = {
                    "position": joint_state[0],
                    "velocity": joint_state[1],
                    "reaction_forces": joint_state[2],
                    "motor_torque": joint_state[3],
                }
            robot_position = list(p.getBasePositionAndOrientation(robotId)[0])
            robot_position[0] -= raw_joint_states[0][0]
            robot_position[1] -= raw_joint_states[1][0]
            robot_position[2] += raw_joint_states[2][0]
            pipette_position = [
                round(robot_position[0] + self.pipette_offset[0], 4),
                round(robot_position[1] + self.pipette_offset[1], 4),
                round(robot_position[2] + self.pipette_offset[2], 4),
            ]
            states[f"robotId_{robotId}"] = {
                "joint_states": joint_states,
                "robot_position": robot_position,
                "pipette_position": pipette_position,
            }
        return states

    def check_contact(self, robotId, specimenId):
        for sphereId in list(self.sphereIds):
            contact_points_specimen = p.getContactPoints(sphereId, specimenId)
            contact_points_robot = p.getContactPoints(sphereId, robotId)
            if contact_points_specimen:
                p.setCollisionFilterPair(sphereId, specimenId, -1, -1, enableCollision=0)
                sphere_position, sphere_orientation = p.getBasePositionAndOrientation(sphereId)
                p.createConstraint(
                    parentBodyUniqueId=sphereId,
                    parentLinkIndex=-1,
                    childBodyUniqueId=-1,
                    childLinkIndex=-1,
                    jointType=p.JOINT_FIXED,
                    jointAxis=[0, 0, 0],
                    parentFramePosition=[0, 0, 0],
                    childFramePosition=sphere_position,
                    childFrameOrientation=sphere_orientation,
                )
                key = f"specimenId_{specimenId}"
                self.droplet_positions.setdefault(key, []).append(sphere_position)
            if contact_points_robot:
                p.removeBody(sphereId)
                if sphereId in self.sphereIds:
                    self.sphereIds.remove(sphereId)

    def set_start_position(self, x, y, z):
        for robotId in self.robotIds:
            robot_position = p.getBasePositionAndOrientation(robotId)[0]
            adjusted_x = x - robot_position[0] - self.pipette_offset[0]
            adjusted_y = y - robot_position[1] - self.pipette_offset[1]
            adjusted_z = z - robot_position[2] - self.pipette_offset[2]
            p.resetJointState(robotId, 0, targetValue=adjusted_x)
            p.resetJointState(robotId, 1, targetValue=adjusted_y)
            p.resetJointState(robotId, 2, targetValue=adjusted_z)

    def get_plate_image(self):
        return self.texture_path

    # =========================================================
    # CLEANUP
    # =========================================================
    def close(self):
        p.disconnect()
