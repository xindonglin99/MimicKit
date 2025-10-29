import enum
import gymnasium.spaces as spaces
import numpy as np
import os
import torch

import anim.kin_char_model as kin_char_model
import anim.motion_lib as motion_lib
import envs.sim_env as sim_env
import envs.base_env as base_env
from util.logger import Logger
import util.torch_util as torch_util

import engines.engine as engine

class CameraMode(enum.Enum):
    still = 0
    track = 1

class CharEnv(sim_env.SimEnv):
    def __init__(self, config, num_envs, device, visualize):
        env_config = config["env"]
        self._global_obs = env_config["global_obs"]
        self._root_height_obs = env_config.get("root_height_obs", True)
        self._zero_center_action = env_config.get("zero_center_action", False)

        self._camera_mode = CameraMode[env_config["camera_mode"]]

        super().__init__(config=config, num_envs=num_envs, device=device,
                         visualize=visualize)
        return

    def _parse_init_pose(self, init_pose, device):
        if (init_pose is not None):
            init_pose = torch.tensor(init_pose, device=device)
        else:
            dof_size = self._kin_char_model.get_dof_size()
            init_pose = torch.zeros(6 + dof_size, dtype=torch.float32, device=device)
            
        init_root_pos, init_root_rot, init_dof_pos = motion_lib.extract_pose_data(init_pose)
        self._init_root_pos = init_root_pos
        self._init_root_rot = torch_util.exp_map_to_quat(init_root_rot)
        self._init_dof_pos = init_dof_pos
        return

    def _build_envs(self, config, num_envs):
        env_config = config["env"]
        char_file = env_config["char_file"]
        self._build_kin_char_model(char_file)
        
        init_pose = env_config.get("init_pose", None)
        self._parse_init_pose(init_pose, self._device)
        
        self._char_ids = []
        
        for e in range(num_envs):
            Logger.print("Building {:d}/{:d} envs".format(e + 1, num_envs), end='\r')
            env_id = self._engine.create_env()
            assert(env_id == e)
            self._build_env(env_id, config)

        Logger.print("\n")

        self._validate_envs()

        return
    
    def _build_env(self, env_id, config):
        char_col = self._get_char_color()
        char_id = self._build_character(env_id, config, color=char_col)

        if (env_id == 0):
            self._print_actor_prop(env_id, char_id)

        if (env_id == 0):
            self._char_ids.append(char_id)
        else:
            char_id0 = self._char_ids[0]
            assert(char_id0 == char_id)
        
        return 
    
    def _build_character(self, env_id, config, color=None):
        char_file = config["env"]["char_file"]
        char_id = self._engine.create_actor(env_id=env_id, 
                                             asset_file=char_file, 
                                             name="character",
                                             color=color)
        return char_id
    
    def _build_kin_char_model(self, char_file):
        _, file_ext = os.path.splitext(char_file)
        if (file_ext == ".xml"):
            char_model = kin_char_model.KinCharModel(self._device)
        else:
            print("Unsupported character file format: {:s}".format(file_ext))
            assert(False)

        self._kin_char_model = char_model
        self._kin_char_model.load_char_file(char_file)
        return
    
    def _build_sim_tensors(self, config):
        super()._build_sim_tensors(config)
        
        self._action_bound_low = torch.tensor(self._action_space.low, device=self._device)
        self._action_bound_high = torch.tensor(self._action_space.high, device=self._device)
        
        env_config = config["env"]
        key_bodies = env_config.get("key_bodies", [])
        self._key_body_ids = self._build_body_ids_tensor(key_bodies)

        return
    
    def _build_action_space(self):
        control_mode = self._engine.get_control_mode()

        if (control_mode == engine.ControlMode.none):
            low, high = self._build_action_bounds_none()

        elif (control_mode == engine.ControlMode.vel):
            low, high = self._build_action_bounds_vel()

        elif (control_mode == engine.ControlMode.torque):
            char_id = self._get_char_id()
            torque_lim = self._engine.get_actor_torque_lim(0, char_id)
            low, high = self._build_action_bounds_torque(torque_lim)

        elif (control_mode == engine.ControlMode.pos
              or control_mode == engine.ControlMode.pd_1d):
            char_id = self._get_char_id()
            dof_low, dof_high = self._engine.get_actor_dof_limits(0, char_id)
            low, high = self._build_action_bounds_pos(dof_low, dof_high)

        else:
            assert(False), "Unsupported control mode: {}".format(control_mode)
        
        # check to make sure that pd_1d is only used for 1D joints
        if (control_mode == engine.ControlMode.pd_1d):
            num_joints = self._kin_char_model.get_num_joints()
            for j in range(1, num_joints):
                j_dim = self._kin_char_model.get_joint_dof_dim(j)
                assert(j_dim <= 1), "pd_1d only supports 1D joints"

        action_space = spaces.Box(low=low, high=high)
        return action_space
    
    def _build_action_bounds_none(self):
        char_id = self._get_char_id()
        dof_pos = self._engine.get_dof_pos(char_id)
        action_size = int(dof_pos.shape[-1])
        low = -np.ones([action_size], dtype=np.float32)
        high = np.ones([action_size], dtype=np.float32)
        return low, high

    def _build_action_bounds_pos(self, dof_low, dof_high):
        low = np.zeros(dof_high.shape, dtype=np.float32)
        high = np.zeros(dof_high.shape, dtype=np.float32)

        num_joints = self._kin_char_model.get_num_joints()
        for j in range(1, num_joints):
            curr_joint = self._kin_char_model.get_joint(j)
            j_dof_dim = curr_joint.get_dof_dim()

            if (j_dof_dim > 0):
                if (j_dof_dim == 3): # 3D spherical j
                    # spherical joints are modeled as exponential maps
                    # so the bounds are computed a bit differently from revolute joints
                    j_low = curr_joint.get_joint_dof(dof_low)
                    j_high = curr_joint.get_joint_dof(dof_high)
                    j_low = np.max(np.abs(j_low))
                    j_high = np.max(np.abs(j_high))
                    curr_scale = max([j_low, j_high])
                    curr_scale = 1.2 * curr_scale

                    curr_low = -curr_scale
                    curr_high = curr_scale
                else:
                    j_low = curr_joint.get_joint_dof(dof_low)
                    j_high = curr_joint.get_joint_dof(dof_high)

                    if (self._zero_center_action):
                        curr_mid = np.zeros_like(j_high)
                    else:
                        curr_mid = 0.5 * (j_high + j_low)

                    diff_high = np.abs(j_high - curr_mid)
                    diff_low = np.abs(j_low - curr_mid)
                    curr_scale = np.maximum(diff_high, diff_low)
                    curr_scale *= 1.4

                    curr_low = curr_mid - curr_scale
                    curr_high = curr_mid + curr_scale

                curr_joint.set_joint_dof(curr_low, low)
                curr_joint.set_joint_dof(curr_high, high)

        return low, high

    def _build_action_bounds_vel(self):
        char_id = self._get_char_id()
        dof_pos = self._engine.get_dof_pos(char_id)
        action_size = int(dof_pos.shape[-1])
        low = -2.0 * np.pi * np.ones([action_size], dtype=np.float32)
        high = 2.0 * np.pi * np.ones([action_size], dtype=np.float32)
        return low, high

    def _build_action_bounds_torque(self, torque_lim):
        char_id = self._get_char_id()
        dof_pos = self._engine.get_dof_pos(char_id)
        assert(dof_pos.shape[-1] == len(torque_lim))
        low = -np.array(torque_lim, dtype=np.float32)
        high = np.array(torque_lim, dtype=np.float32)
        return low, high
    
    def _print_actor_prop(self, env_id, char_id):
        num_dofs = self._engine.get_actor_dof_count(env_id, char_id)
        total_mass = self._engine.calc_actor_mass(env_id, char_id)
        char_info = "Char properties\n\tDoFs: {:d}\n\tMass: {:.3f} kg\n".format(num_dofs, total_mass)
        Logger.print(char_info)
        return
    
    def _validate_envs(self):
        # checks to make sure the kinematic model is consistent with the simulation model
        char_id = self._get_char_id()
        sim_body_names = self._engine.get_actor_body_names(0, char_id)
        kin_body_names = self._kin_char_model.get_body_names()

        for sim_name, kin_name in zip(sim_body_names, kin_body_names):
            assert(sim_name == kin_name)
        return
    
    def _get_char_id(self):
        return self._char_ids[0]
    
    def _update_reward(self):
        char_id = self._get_char_id()
        char_root_pos = self._engine.get_root_pos(char_id)
        self._reward_buf[:] = compute_reward(char_root_pos)
        return

    def _update_done(self):
        self._done_buf[:] = compute_done(self._done_buf, self._time_buf, 
                                         self._episode_length)
        return
    
    def _compute_obs(self, env_ids=None):
        char_id = self._get_char_id()
        root_pos = self._engine.get_root_pos(char_id)
        root_rot = self._engine.get_root_rot(char_id)
        root_vel = self._engine.get_root_vel(char_id)
        root_ang_vel = self._engine.get_root_ang_vel(char_id)
        dof_pos = self._engine.get_dof_pos(char_id)
        dof_vel = self._engine.get_dof_vel(char_id)

        if (env_ids is not None):
            root_pos = root_pos[env_ids]
            root_rot = root_rot[env_ids]
            root_vel = root_vel[env_ids]
            root_ang_vel = root_ang_vel[env_ids]
            dof_pos = dof_pos[env_ids]
            dof_vel = dof_vel[env_ids]

        joint_rot = self._kin_char_model.dof_to_rot(dof_pos)

        if (self._has_key_bodies()):
            body_pos, _ = self._kin_char_model.forward_kinematics(root_pos, root_rot, joint_rot)
            key_pos = body_pos[..., self._key_body_ids, :]
        else:
            key_pos = torch.zeros([0], device=self._device)

        obs = compute_char_obs(root_pos=root_pos,
                               root_rot=root_rot, 
                               root_vel=root_vel,
                               root_ang_vel=root_ang_vel,
                               joint_rot=joint_rot,
                               dof_vel=dof_vel,
                               key_pos=key_pos,
                               global_obs=self._global_obs,
                               root_height_obs=self._root_height_obs)
        return obs
    
    def _reset_envs(self, env_ids):
        super()._reset_envs(env_ids)

        if (len(env_ids) > 0):
            self._reset_char(env_ids)
        return

    def _reset_char(self, env_ids):
        char_id = self._get_char_id()

        self._engine.set_root_pos(env_ids, char_id, self._init_root_pos)
        self._engine.set_root_rot(env_ids, char_id, self._init_root_rot)
        self._engine.set_root_vel(env_ids, char_id, 0.0)
        self._engine.set_root_ang_vel(env_ids, char_id, 0.0)
        
        self._engine.set_dof_pos(env_ids, char_id, self._init_dof_pos)
        self._engine.set_dof_vel(env_ids, char_id, 0.0)
        
        self._engine.set_body_vel(env_ids, char_id, 0.0)
        self._engine.set_body_ang_vel(env_ids, char_id, 0.0)
        return

    def _apply_action(self, actions):
        char_id = self._get_char_id()
        clip_action = torch.minimum(torch.maximum(actions, self._action_bound_low), self._action_bound_high)
        self._engine.set_cmd(char_id, clip_action)
        return
    
    def _build_body_ids_tensor(self, body_names):
        char_id = self._get_char_id()
        body_ids = []

        for body_name in body_names:
            body_id = self._engine.find_actor_body_id(0, char_id, body_name)
            assert(body_id != -1)
            body_ids.append(body_id)

        body_ids = torch.tensor(body_ids, device=self._device, dtype=torch.long)
        return body_ids
    
    def _has_key_bodies(self):
        return len(self._key_body_ids) > 0

    def _init_camera(self):
        char_id = self._get_char_id()
        char_root_pos = self._engine.get_root_pos(char_id)
        char_pos = char_root_pos[0].cpu().numpy()
            
        cam_pos = [char_pos[0], char_pos[1] - 5.0, 3.0]
        cam_target = [char_pos[0], char_pos[1], 0.0]

        self._engine.update_camera(cam_pos, cam_target)
        self._cam_prev_char_pos = char_pos
        return

    def _update_camera(self):
        if (self._camera_mode is CameraMode.still):
            pass
        elif (self._camera_mode is CameraMode.track):
            char_id = self._get_char_id()
            char_root_pos = self._engine.get_root_pos(char_id)
            char_pos = char_root_pos[0].cpu().numpy()
            
            cam_pos = self._engine.get_camera_pos()
            cam_delta = cam_pos - self._cam_prev_char_pos

            new_cam_target = np.array([char_pos[0], char_pos[1], 1.0])
            new_cam_pos = np.array([char_pos[0] + cam_delta[0], 
                                    char_pos[1] + cam_delta[1], 
                                    cam_pos[2]])

            self._engine.update_camera(new_cam_pos, new_cam_target)

            self._cam_prev_char_pos[:] = char_pos
        else:
            assert(False), "Unsupported camera mode {}".format(self._camera_mode)

        return

    def _get_char_color(self):
        return np.array([0.5, 0.65, 0.95])



#####################################################################
###=========================jit functions=========================###
#####################################################################

@torch.jit.script
def convert_to_local_body_pos(root_rot, body_pos):
    # type: (Tensor, Tensor) -> Tensor
    
    heading_inv_rot = torch_util.calc_heading_quat_inv(root_rot)
    heading_rot_expand = heading_inv_rot.unsqueeze(-2)
    heading_rot_expand = heading_rot_expand.repeat((1, body_pos.shape[1], 1))
    flat_heading_rot_expand = heading_rot_expand.reshape(heading_rot_expand.shape[0] * heading_rot_expand.shape[1], 
                                                            heading_rot_expand.shape[2])
    flat_body_pos = body_pos.reshape(body_pos.shape[0] * body_pos.shape[1], body_pos.shape[2])
    flat_local_body_pos = torch_util.quat_rotate(flat_heading_rot_expand, flat_body_pos)
    local_body_pos = flat_local_body_pos.reshape(body_pos.shape[0], body_pos.shape[1], body_pos.shape[2])

    return local_body_pos

@torch.jit.script
def convert_to_local_root_body_pos(root_rot, body_pos):
    # type: (Tensor, Tensor) -> Tensor
    
    root_inv_rot = torch_util.quat_conjugate(root_rot)
    root_rot_expand = root_inv_rot.unsqueeze(-2)
    root_rot_expand = root_rot_expand.repeat((1, body_pos.shape[1], 1))
    flat_root_rot_expand = root_rot_expand.reshape(root_rot_expand.shape[0] * root_rot_expand.shape[1], 
                                                   root_rot_expand.shape[2])
    flat_body_pos = body_pos.reshape(body_pos.shape[0] * body_pos.shape[1], body_pos.shape[2])
    flat_local_body_pos = torch_util.quat_rotate(flat_root_rot_expand, flat_body_pos)
    local_body_pos = flat_local_body_pos.reshape(body_pos.shape[0], body_pos.shape[1], body_pos.shape[2])

    return local_body_pos

@torch.jit.script
def compute_char_obs(root_pos, root_rot, root_vel, root_ang_vel, joint_rot, dof_vel, key_pos, global_obs, root_height_obs):
    # type: (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, bool, bool) -> Tensor
    heading_rot = torch_util.calc_heading_quat_inv(root_rot)
    
    if (global_obs):
        root_rot_obs = torch_util.quat_to_tan_norm(root_rot)
        root_vel_obs = root_vel
        root_ang_vel_obs = root_ang_vel
    else:
        local_root_rot = torch_util.quat_mul(heading_rot, root_rot)
        root_rot_obs = torch_util.quat_to_tan_norm(local_root_rot)
        root_vel_obs = torch_util.quat_rotate(heading_rot, root_vel)
        root_ang_vel_obs = torch_util.quat_rotate(heading_rot, root_ang_vel)


    joint_rot_flat = torch.reshape(joint_rot, [joint_rot.shape[0] * joint_rot.shape[1], joint_rot.shape[2]])
    joint_rot_obs_flat = torch_util.quat_to_tan_norm(joint_rot_flat)
    joint_rot_obs = torch.reshape(joint_rot_obs_flat, [joint_rot.shape[0], joint_rot.shape[1] * joint_rot_obs_flat.shape[-1]])

    obs = [root_rot_obs, root_vel_obs, root_ang_vel_obs, joint_rot_obs, dof_vel]

    if (len(key_pos) > 0):
        root_pos_expand = root_pos.unsqueeze(-2)
        key_pos = key_pos - root_pos_expand
        if (not global_obs):
            key_pos = convert_to_local_body_pos(root_rot, key_pos)

        key_pos_flat = torch.reshape(key_pos, [key_pos.shape[0], key_pos.shape[1] * key_pos.shape[2]])
        obs = obs + [key_pos_flat]

    if (root_height_obs):
        root_h = root_pos[:, 2:3]
        obs = [root_h] + obs
    
    obs = torch.cat(obs, dim=-1)
    return obs

@torch.jit.script
def compute_reward(root_pos):
    # type: (Tensor) -> Tensor
    r = torch.ones_like(root_pos[..., 0])
    return r

@torch.jit.script
def compute_done(done_buf, time, ep_len):
    # type: (Tensor, Tensor, float) -> Tensor
    timeout = time >= ep_len
    done = torch.full_like(done_buf, base_env.DoneFlags.NULL.value)
    done[timeout] = base_env.DoneFlags.TIME.value
    return done