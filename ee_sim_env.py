import numpy as np
import collections
import os

from constants import DT, XML_DIR, START_ARM_POSE
from constants import PUPPET_GRIPPER_POSITION_CLOSE
from constants import PUPPET_GRIPPER_POSITION_UNNORMALIZE_FN
from constants import PUPPET_GRIPPER_POSITION_NORMALIZE_FN
from constants import PUPPET_GRIPPER_VELOCITY_NORMALIZE_FN

from utils import sample_2_box_poses
from dm_control import mujoco
from dm_control.rl import control
from dm_control.suite import base

import IPython
e = IPython.embed


def make_ee_sim_env(task_cls):
    xml_path = os.path.join(XML_DIR, f'bimanual_viperx_ee_multitask.xml')
    physics = mujoco.Physics.from_xml_path(xml_path)
    task = task_cls(random=False)
    env = control.Environment(physics, task, time_limit=20, control_timestep=DT,
                                n_sub_steps=None, flat_observation=False)
    
    return env

class BimanualViperXEETask(base.Task):
    def __init__(self, random=None):
        super().__init__(random=random)

    def before_step(self, action, physics):
        a_len = len(action) // 2
        action_left = action[:a_len]
        action_right = action[a_len:]

        # set mocap position and quat
        # left
        np.copyto(physics.data.mocap_pos[0], action_left[:3])
        np.copyto(physics.data.mocap_quat[0], action_left[3:7])
        # right
        np.copyto(physics.data.mocap_pos[1], action_right[:3])
        np.copyto(physics.data.mocap_quat[1], action_right[3:7])

        # set gripper
        g_left_ctrl = PUPPET_GRIPPER_POSITION_UNNORMALIZE_FN(action_left[7])
        g_right_ctrl = PUPPET_GRIPPER_POSITION_UNNORMALIZE_FN(action_right[7])
        np.copyto(physics.data.ctrl, np.array([g_left_ctrl, -g_left_ctrl, g_right_ctrl, -g_right_ctrl]))

    def initialize_robots(self, physics):
        # reset joint position
        physics.named.data.qpos[:16] = START_ARM_POSE

        # reset mocap to align with end effector
        # to obtain these numbers:
        # (1) make an ee_sim env and reset to the same start_pose
        # (2) get env._physics.named.data.xpos['vx300s_left/gripper_link']
        #     get env._physics.named.data.xquat['vx300s_left/gripper_link']
        #     repeat the same for right side
        np.copyto(physics.data.mocap_pos[0], [-0.31718881, 0.5, 0.29525084])
        np.copyto(physics.data.mocap_quat[0], [1, 0, 0, 0])
        # right
        np.copyto(physics.data.mocap_pos[1], np.array([0.31718881, 0.49999888, 0.29525084]))
        np.copyto(physics.data.mocap_quat[1],  [1, 0, 0, 0])

        # reset gripper control
        close_gripper_control = np.array([
            PUPPET_GRIPPER_POSITION_CLOSE,
            -PUPPET_GRIPPER_POSITION_CLOSE,
            PUPPET_GRIPPER_POSITION_CLOSE,
            -PUPPET_GRIPPER_POSITION_CLOSE,
        ])
        np.copyto(physics.data.ctrl, close_gripper_control)

    def initialize_episode(self, physics):
        """Sets the state of the environment at the start of each episode."""
        super().initialize_episode(physics)

    @staticmethod
    def get_qpos(physics):
        qpos_raw = physics.data.qpos.copy()
        left_qpos_raw = qpos_raw[:8]
        right_qpos_raw = qpos_raw[8:16]
        left_arm_qpos = left_qpos_raw[:6]
        right_arm_qpos = right_qpos_raw[:6]
        left_gripper_qpos = [PUPPET_GRIPPER_POSITION_NORMALIZE_FN(left_qpos_raw[6])]
        right_gripper_qpos = [PUPPET_GRIPPER_POSITION_NORMALIZE_FN(right_qpos_raw[6])]
        return np.concatenate([left_arm_qpos, left_gripper_qpos, right_arm_qpos, right_gripper_qpos])

    @staticmethod
    def get_qvel(physics):
        qvel_raw = physics.data.qvel.copy()
        left_qvel_raw = qvel_raw[:8]
        right_qvel_raw = qvel_raw[8:16]
        left_arm_qvel = left_qvel_raw[:6]
        right_arm_qvel = right_qvel_raw[:6]
        left_gripper_qvel = [PUPPET_GRIPPER_VELOCITY_NORMALIZE_FN(left_qvel_raw[6])]
        right_gripper_qvel = [PUPPET_GRIPPER_VELOCITY_NORMALIZE_FN(right_qvel_raw[6])]
        return np.concatenate([left_arm_qvel, left_gripper_qvel, right_arm_qvel, right_gripper_qvel])

    @staticmethod
    def get_env_state(physics):
        raise NotImplementedError

    def get_observation(self, physics):
        # note: it is important to do .copy()
        obs = collections.OrderedDict()
        obs['qpos'] = self.get_qpos(physics)
        obs['qvel'] = self.get_qvel(physics)
        obs['env_state'] = self.get_env_state(physics)
        obs['images'] = dict()
        obs['images']['top'] = physics.render(height=480, width=640, camera_id='top')
        obs['images']['angle'] = physics.render(height=480, width=640, camera_id='angle')
        obs['images']['vis'] = physics.render(height=480, width=640, camera_id='front_close')
        # used in scripted policy to obtain starting pose
        obs['mocap_pose_left'] = np.concatenate([physics.data.mocap_pos[0], physics.data.mocap_quat[0]]).copy()
        obs['mocap_pose_right'] = np.concatenate([physics.data.mocap_pos[1], physics.data.mocap_quat[1]]).copy()

        # used when replaying joint trajectory
        obs['gripper_ctrl'] = physics.data.ctrl.copy()
        return obs

    def get_reward(self, physics):
        raise NotImplementedError


class GraspCubeEETask(BimanualViperXEETask):
    def __init__(self, grasp_box='red_box', random=None):
        super().__init__(random=random)
        self.max_reward = 4
        self.grasp_box = grasp_box


    def initialize_episode(self, physics):
        """Sets the state of the environment at the start of each episode."""
        self.initialize_robots(physics)

        red_cube_pose, blue_cube_pose = sample_2_box_poses()

        red_box_start_idx = physics.model.name2id('red_box_joint', 'joint')
        np.copyto(physics.data.qpos[red_box_start_idx : red_box_start_idx + 7], red_cube_pose)

        blue_box_start_idx = physics.model.name2id('blue_box_joint', 'joint') + 6
        np.copyto(physics.data.qpos[blue_box_start_idx : blue_box_start_idx + 7], blue_cube_pose)

        super().initialize_episode(physics)

    @staticmethod
    def get_env_state(physics):
        env_state = physics.data.qpos.copy()[16:]
        return env_state

    def get_reward(self, physics):
        all_contact_pairs = []
        for i_contact in range(physics.data.ncon):
            id_geom_1 = physics.data.contact[i_contact].geom1
            id_geom_2 = physics.data.contact[i_contact].geom2
            name_geom_1 = physics.model.id2name(id_geom_1, 'geom')
            name_geom_2 = physics.model.id2name(id_geom_2, 'geom')
            contact_pair = (name_geom_1, name_geom_2)
            all_contact_pairs.append(contact_pair)

        touch_right_gripper = (self.grasp_box, "vx300s_right/10_right_gripper_finger") in all_contact_pairs or \
                             ("vx300s_right/10_right_gripper_finger", self.grasp_box) in all_contact_pairs
        grasp_box_on_table = (self.grasp_box, "table") in all_contact_pairs or \
                             ("table", self.grasp_box) in all_contact_pairs

        reward = 0
        if touch_right_gripper:
            reward = 2
        if touch_right_gripper and not grasp_box_on_table:
            reward = 4
        
        return reward


class StackCubeEETask(BimanualViperXEETask):
    def __init__(self, top_box='red_box', random=None):
        super().__init__(random=random)
        self.max_reward = 4
        self.top_box = top_box
        if top_box == 'blue_box':
            self.bottom_box = 'red_box'
        else:
            self.bottom_box = 'blue_box'


    def initialize_episode(self, physics):
        """Sets the state of the environment at the start of each episode."""
        self.initialize_robots(physics)

        red_cube_pose, blue_cube_pose = sample_2_box_poses()

        red_box_start_idx = physics.model.name2id('red_box_joint', 'joint')
        np.copyto(physics.data.qpos[red_box_start_idx : red_box_start_idx + 7], red_cube_pose)

        blue_box_start_idx = physics.model.name2id('blue_box_joint', 'joint') + 6
        np.copyto(physics.data.qpos[blue_box_start_idx : blue_box_start_idx + 7], blue_cube_pose)

        super().initialize_episode(physics)

    @staticmethod
    def get_env_state(physics):
        env_state = physics.data.qpos.copy()[16:]
        return env_state

    def get_reward(self, physics):
        all_contact_pairs = []
        for i_contact in range(physics.data.ncon):
            id_geom_1 = physics.data.contact[i_contact].geom1
            id_geom_2 = physics.data.contact[i_contact].geom2
            name_geom_1 = physics.model.id2name(id_geom_1, 'geom')
            name_geom_2 = physics.model.id2name(id_geom_2, 'geom')
            contact_pair = (name_geom_1, name_geom_2)
            all_contact_pairs.append(contact_pair)

        touch_right_gripper = (self.top_box, "vx300s_right/10_right_gripper_finger") in all_contact_pairs or \
                             ("vx300s_right/10_right_gripper_finger", self.top_box) in all_contact_pairs
        top_box_on_table = (self.top_box, "table") in all_contact_pairs or \
                             ("table", self.top_box) in all_contact_pairs
        bottom_box_on_table = (self.bottom_box, "table") in all_contact_pairs or \
                             ("table", self.bottom_box) in all_contact_pairs
        boxes_touching = (self.top_box, self.bottom_box) in all_contact_pairs or \
                             (self.bottom_box, self.top_box) in all_contact_pairs

        reward = 0
        if touch_right_gripper:
            reward = 1
        if touch_right_gripper and not top_box_on_table:
            reward = 2
        if touch_right_gripper and not top_box_on_table and boxes_touching:
            reward = 3
        if not touch_right_gripper and not top_box_on_table and boxes_touching and bottom_box_on_table:
            reward = 4
        
        return reward


class TransferCubeEETask(BimanualViperXEETask):
    def __init__(self, transfer_box='red_box', random=None):
        super().__init__(random=random)
        self.transfer_box = transfer_box
        self.max_reward = 4

    def initialize_episode(self, physics):
        """Sets the state of the environment at the start of each episode."""
        self.initialize_robots(physics)

        red_cube_pose, blue_cube_pose = sample_2_box_poses()

        red_box_start_idx = physics.model.name2id('red_box_joint', 'joint')
        np.copyto(physics.data.qpos[red_box_start_idx : red_box_start_idx + 7], red_cube_pose)

        blue_box_start_idx = physics.model.name2id('blue_box_joint', 'joint') + 6
        np.copyto(physics.data.qpos[blue_box_start_idx : blue_box_start_idx + 7], blue_cube_pose)

        super().initialize_episode(physics)

    @staticmethod
    def get_env_state(physics):
        env_state = physics.data.qpos.copy()[16:]
        return env_state

    def get_reward(self, physics):
        # return whether left gripper is holding the box
        all_contact_pairs = []
        for i_contact in range(physics.data.ncon):
            id_geom_1 = physics.data.contact[i_contact].geom1
            id_geom_2 = physics.data.contact[i_contact].geom2
            name_geom_1 = physics.model.id2name(id_geom_1, 'geom')
            name_geom_2 = physics.model.id2name(id_geom_2, 'geom')
            contact_pair = (name_geom_1, name_geom_2)
            all_contact_pairs.append(contact_pair)

        touch_left_gripper = (self.transfer_box, "vx300s_left/10_left_gripper_finger") in all_contact_pairs \
                            or ("vx300s_left/10_left_gripper_finger", self.transfer_box) in all_contact_pairs
        touch_right_gripper = (self.transfer_box, "vx300s_right/10_right_gripper_finger") in all_contact_pairs \
                            or ("vx300s_right/10_right_gripper_finger", self.transfer_box) in all_contact_pairs
        touch_table = (self.transfer_box, "table") in all_contact_pairs \
                            or ("table", self.transfer_box) in all_contact_pairs

        reward = 0
        if touch_right_gripper:
            reward = 1
        if touch_right_gripper and not touch_table: # lifted
            reward = 2
        if touch_left_gripper: # attempted transfer
            reward = 3
        if touch_left_gripper and not touch_table: # successful transfer
            reward = 4
        
        return reward
