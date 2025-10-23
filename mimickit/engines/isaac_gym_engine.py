import isaacgym.gymapi as gymapi
import isaacgym.gymtorch as gymtorch
import isaacgym.gymutil as gymutil

import enum
import numpy as np
import os
import re
import sys
import torch
import time

import engines.engine as engine

class ControlMode(enum.Enum):
    none = 0
    pos = 1
    vel = 2
    torque = 3
    pd_1d = 4

class IsaacGymEngine(engine.Engine):
    def __init__(self, config, num_envs, device, visualize, control_mode=None):
        super().__init__()
        physics_engine = gymapi.SimType.SIM_PHYSX

        self._device = device
        self._num_envs = num_envs
        self._enable_viewer_sync = True
        
        self._gym = gymapi.acquire_gym()
        
        sim_freq = config.get("sim_freq", 60)
        control_freq = config.get("control_freq", 10)
        assert(sim_freq >= control_freq and sim_freq % control_freq == 0), \
            "Simulation frequency must be a multiple of the control frequency"

        self._timestep = 1.0 / control_freq
        self._sim_steps = int(sim_freq / control_freq)
        sim_timestep = 1.0 / sim_freq
        self._sim = self._create_simulator(physics_engine, config, sim_timestep, visualize)

        self._env_spacing = config["env_spacing"]
        self._envs = []
        
        if (control_mode is not None):
            self._control_mode = control_mode
        elif ("control_mode" in config):
            self._control_mode = ControlMode[config["control_mode"]]
        else:
            self._control_mode = ControlMode.none

        self._actor_kp = [[] for i in range(num_envs)]
        self._actor_kd = [[] for i in range(num_envs)]
        self._actor_torque_lim = [[] for i in range(num_envs)]

        self._apply_forces_callback = None

        if (visualize):
            self._build_viewer()
            self._prev_frame_time = 0.0

        return
    
    def create_env(self):
        env_spacing = self._get_env_spacing()
        num_envs = self.get_num_envs()
        num_env_per_row = int(np.sqrt(num_envs))
        lower = gymapi.Vec3(-env_spacing, -env_spacing, 0.0)
        upper = gymapi.Vec3(env_spacing, env_spacing, env_spacing)
        env_ptr = self._gym.create_env(self._sim, lower, upper, num_env_per_row)

        env_id = len(self._envs)
        self._envs.append(env_ptr)

        return env_id
    
    def finalize_sim(self):
        self._gym.prepare_sim(self._sim)
        self._build_sim_tensors()
        return
    
    def step(self):
        self._apply_cmd()

        for i in range(self._sim_steps):
            self._pre_sim_step()
            self._sim_step()
            
        self._refresh_sim_tensors()
        return

    def update_sim_state(self):
        actor_ids = self._need_reset_buf.nonzero(as_tuple=False)
        actor_ids = actor_ids.type(torch.int32).flatten()

        if (len(actor_ids) > 0):
            self._gym.set_actor_root_state_tensor_indexed(self._sim,
                                                         gymtorch.unwrap_tensor(self._root_state_raw),
                                                         gymtorch.unwrap_tensor(actor_ids), len(actor_ids))
            if (self._has_dof()):
                has_dof = self._actor_dof_dims[actor_ids.type(torch.long)] > 0
                dof_actor_ids = actor_ids[has_dof]
                self._gym.set_dof_state_tensor_indexed(self._sim,
                                                      gymtorch.unwrap_tensor(self._dof_state_raw),
                                                      gymtorch.unwrap_tensor(dof_actor_ids), len(dof_actor_ids))
                
                dof_pos = self._dof_state_raw[..., :, 0]
                dof_pos = dof_pos.contiguous()
                self._gym.set_dof_position_target_tensor_indexed(self._sim,
                                                      gymtorch.unwrap_tensor(dof_pos),
                                                      gymtorch.unwrap_tensor(dof_actor_ids), len(dof_actor_ids))

            self._actors_need_reset[:] = False
            self._refresh_sim_tensors()

        return
    
    def load_asset(self, file, fix_base=False):
        asset_options = gymapi.AssetOptions()
        asset_options.angular_damping = 0.01
        asset_options.max_angular_velocity = 100.0
        asset_options.fix_base_link = fix_base

        file_dir = os.path.dirname(file)
        file_name = os.path.basename(file)
        asset = self._gym.load_asset(self._sim, file_dir, file_name, asset_options)
        return asset
    
    def create_actor(self, env_id, asset, name, col_group, col_filter, segmentation_id, start_pos=None, start_rot=None, color=None, disable_motors=False):
        env_ptr = self.get_env(env_id)
        start_pose = gymapi.Transform()
        
        if (start_pos is not None):
            start_pose.p = gymapi.Vec3(start_pos[0], start_pos[1], start_pos[2])
        else:
            start_pose.p = gymapi.Vec3(0.0, 0.0, 0.0)

        if (start_rot is not None):
            start_pose.r = gymapi.Quat(start_rot[0], start_rot[1], start_rot[2], start_rot[3])
        else:
            start_pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)

        actor_id = self._gym.create_actor(env_ptr, asset, start_pose, name, col_group, col_filter, segmentation_id)
        
        if (self._enable_dof_force_sensors()):
            self._gym.enable_actor_dof_force_sensors(env_ptr, actor_id)

        if (disable_motors):
            control_mode = ControlMode.none
        else:
            control_mode = self.get_control_mode()

        dof_prop = self._gym.get_actor_dof_properties(env_ptr, actor_id)
        kp = dof_prop["stiffness"]
        kd = dof_prop["damping"]
        
        actuator_props = self._gym.get_actor_actuator_properties(env_ptr, actor_id)
        motor_efforts = [prop.motor_effort for prop in actuator_props]
        torque_lim = np.array(motor_efforts)
        
        if (control_mode == ControlMode.none):
            kp = kp * 0
            kd = kd * 0

        self._actor_kp[env_id].append(torch.tensor(kp, device=self._device, dtype=torch.float32))
        self._actor_kd[env_id].append(torch.tensor(kd, device=self._device, dtype=torch.float32))
        self._actor_torque_lim[env_id].append(torch.tensor(torque_lim, device=self._device, dtype=torch.float32))
        assert(actor_id == len(self._actor_kp[env_id]) - 1)
        assert(actor_id == len(self._actor_kd[env_id]) - 1)
        assert(actor_id == len(self._actor_torque_lim[env_id]) - 1)

        drive_mode = self._control_mode_to_drive_mode(control_mode)
        dof_prop["driveMode"] = drive_mode
        self._modify_control_mode_dof_prop(control_mode, dof_prop)

        self._gym.set_actor_dof_properties(env_ptr, actor_id, dof_prop)

        if (color is not None):
            num_bodies = self._gym.get_actor_rigid_body_count(env_ptr, actor_id)
            for b in range(num_bodies):
                self.set_rigid_body_color(env_id, actor_id, b, color)

        return actor_id
        
    def set_cmd(self, actor_id, cmd):
        if (self._has_dof()):
            actor_cmd = self._actor_dof_cmd[actor_id]
            actor_cmd[:] = cmd
        return
    
    def update_camera(self, pos, look_at):
        self._gym.viewer_camera_look_at(self._viewer, None, 
                                      gymapi.Vec3(pos[0], pos[1], pos[2]),
                                      gymapi.Vec3(look_at[0], look_at[1], look_at[2]))
        return
    
    def get_camera_pos(self):
        cam_trans = self._gym.get_viewer_camera_transform(self._viewer, None)
        cam_pos = np.array([cam_trans.p.x, cam_trans.p.y, cam_trans.p.z])
        return cam_pos
    
    def render(self):
        # check for window closed
        if (self._gym.query_viewer_has_closed(self._viewer)):
            sys.exit()

        # check for keyboard events
        for evt in self._gym.query_viewer_action_events(self._viewer):
            if evt.action == "QUIT" and evt.value > 0:
                sys.exit()
            elif evt.action == "toggle_viewer_sync" and evt.value > 0:
                self._enable_viewer_sync = not self._enable_viewer_sync

        # fetch results
        if (self._device != "cpu"):
            self._gym.fetch_results(self._sim, True)
                
        # step graphics
        if (self._enable_viewer_sync):
            self._gym.step_graphics(self._sim)
            self._gym.draw_viewer(self._viewer, self._sim, True)

            # Wait for dt to elapse in real time.
            # This synchronizes the physics simulation with the rendering rate.
            self._gym.sync_frame_time(self._sim)

            # it seems that in some cases sync_frame_time still results in higher-than-realtime framerate
            # this code will slow down the rendering to real time
            now = time.time()
            delta = now - self._prev_frame_time

            if (delta < self._timestep):
                time.sleep(self._timestep - delta)

            self._prev_frame_time = time.time()

        else:
            self._gym.poll_viewer_events(self._viewer)
            
        self._gym.clear_lines(self._viewer)
        return
    
    def get_timestep(self):
        return self._timestep
    
    def get_sim_timestep(self):
        sim_params = self._gym.get_sim_params(self._sim)
        return sim_params.dt
    
    def get_num_envs(self):
        return self._num_envs
    
    def get_actors_per_env(self):
        return self._actors_per_env
    
    def get_root_pos(self, actor_id):
        return self._root_pos[..., actor_id, :]
    
    def get_root_rot(self, actor_id):
        return self._root_rot[..., actor_id, :]
    
    def get_root_vel(self, actor_id):
        return self._root_vel[..., actor_id, :]
    
    def get_root_ang_vel(self, actor_id):
        return self._root_ang_vel[..., actor_id, :]
    
    def get_dof_pos(self, actor_id):
        return self._actor_dof_pos[actor_id]
    
    def get_dof_vel(self, actor_id):
        return self._actor_dof_vel[actor_id]
    
    def get_dof_forces(self, actor_id):
        return self._actor_dof_forces[actor_id]
    
    def get_body_pos(self, actor_id):
        return self._actor_body_pos[actor_id]
    
    def get_body_rot(self, actor_id):
        return self._actor_body_rot[actor_id]
    
    def get_body_vel(self, actor_id):
        return self._actor_body_vel[actor_id]
    
    def get_body_ang_vel(self, actor_id):
        return self._actor_body_ang_vel[actor_id]
    
    def get_contact_forces(self, actor_id):
        return self._actor_contact_forces[actor_id]
    
    def set_root_pos(self, env_id, actor_id, root_pos):
        if (env_id is None):
            self._root_pos[:, actor_id, :] = root_pos
        else:
            self._root_pos[env_id, actor_id, :] = root_pos

        self._flag_actor_needs_reset(env_id, actor_id)
        return
    
    def set_root_rot(self, env_id, actor_id, root_rot):
        if (env_id is None):
            self._root_rot[:, actor_id, :] = root_rot
        else:
            self._root_rot[env_id, actor_id, :] = root_rot
        
        self._flag_actor_needs_reset(env_id, actor_id)
        return
    
    def set_root_vel(self, env_id, actor_id, root_vel):
        if (env_id is None):
            self._root_vel[:, actor_id, :] = root_vel
        else:
            self._root_vel[env_id, actor_id, :] = root_vel
        
        self._flag_actor_needs_reset(env_id, actor_id)
        return
    
    def set_root_ang_vel(self, env_id, actor_id, root_ang_vel):
        if (env_id is None):
            self._root_ang_vel[:, actor_id, :] = root_ang_vel
        else:
            self._root_ang_vel[env_id, actor_id, :] = root_ang_vel
        
        self._flag_actor_needs_reset(env_id, actor_id)
        return
    
    def set_dof_pos(self, env_id, actor_id, dof_pos):
        if (env_id is None):
            self._actor_dof_pos[actor_id][:] = dof_pos
        else:
            self._actor_dof_pos[actor_id][env_id, :] = dof_pos
        
        self._flag_actor_needs_reset(env_id, actor_id)
        return
    
    def set_dof_vel(self, env_id, actor_id, dof_vel):
        if (env_id is None):
            self._actor_dof_vel[actor_id][:] = dof_vel
        else:
            self._actor_dof_vel[actor_id][env_id, :] = dof_vel
        
        self._flag_actor_needs_reset(env_id, actor_id)
        return
    
    def set_body_vel(self, env_id, actor_id, body_vel):
        if (env_id is None):
            self._actor_body_vel[actor_id][:] = body_vel
        else:
            self._actor_body_vel[actor_id][env_id, :] = body_vel
        
        self._flag_actor_needs_reset(env_id, actor_id)
        return
    
    def set_body_ang_vel(self, env_id, actor_id, body_ang_vel):
        if (env_id is None):
            self._actor_body_ang_vel[actor_id][:] = body_ang_vel
        else:
            self._actor_body_ang_vel[actor_id][env_id, :] = body_ang_vel
        
        self._flag_actor_needs_reset(env_id, actor_id)
        return
    
    def set_body_forces(self, env_id, actor_id, body_id, forces):
        if (env_id is None):
            self._actor_body_forces[actor_id][:, body_id, :] = forces
        else:
            self._actor_body_forces[actor_id][env_id, body_id, :] = forces

        if (env_id is None or len(env_id) > 0):
            self._has_body_forces = True
        return
    
    def get_actor_kp(self, actor_id):
        return self._actor_kp[actor_id]
    
    def get_actor_kd(self, actor_id):
        return self._actor_kd[actor_id]
    
    def get_actor_torque_lim(self, actor_id):
        return self._actor_torque_lim[actor_id]
    
    def get_actor_dof_properties(self, env_id, actor_id):
        env_ptr = self.get_env(env_id)
        return self._gym.get_actor_dof_properties(env_ptr, actor_id)
    
    def find_actor_body_id(self, env_id, actor_id, body_name):
        env_ptr = self.get_env(env_id)
        return self._gym.find_actor_rigid_body_handle(env_ptr, actor_id, body_name)
    
    def get_env(self, env_id):
        return self._envs[env_id]
    
    def set_rigid_body_color(self, env_id, actor_id, body_id, color):
        env_ptr = self.get_env(env_id)
        self._gym.set_rigid_body_color(env_ptr, actor_id, body_id, gymapi.MESH_VISUAL,
                                       gymapi.Vec3(color[0], color[1], color[2]))
        return
    
    def build_ground_plane(self, config):
        plane_config = config["plane"]

        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
        plane_params.static_friction = plane_config["static_friction"]
        plane_params.dynamic_friction = plane_config["dynamic_friction"]
        plane_params.restitution = plane_config["restitution"]

        self._gym.add_ground(self._sim, plane_params)
        return
    
    def get_actor_dof_count(self, env_id, actor_id):
        env_ptr = self.get_env(env_id)
        num_dofs = self._gym.get_actor_dof_count(env_ptr, actor_id)
        return num_dofs
    
    def calc_actor_mass(self, env_id, actor_id):
        env_ptr = self.get_env(env_id)
        rb_props = self._gym.get_actor_rigid_body_properties(env_ptr, actor_id)
        total_mass = sum(rb.mass for rb in rb_props)
        return total_mass
    
    def get_actor_body_names(self, env_id, actor_id):
        env_ptr = self.get_env(env_id)
        body_names = self._gym.get_actor_rigid_body_names(env_ptr, actor_id)
        return body_names
    
    def get_control_mode(self):
        return self._control_mode
    
    def draw_lines(self, env_id, verts, cols):
        env_ptr = self.get_env(env_id)
        num_lines = verts.shape[0]
        self._gym.add_lines(self._viewer, env_ptr, num_lines, verts, cols)
        return

    def set_apply_forces_callback(self, callback):
        self._apply_forces_callback = callback
        return

    def _create_simulator(self, physics_engine, config, sim_timestep, visualize):
        sim_params = self._parse_sim_params(config, sim_timestep)
        compute_device_id = self._get_device_idx()
        if (visualize):
            graphics_device_id = compute_device_id
        else:
            graphics_device_id = -1

        sim = self._gym.create_sim(compute_device_id, graphics_device_id, 
                                   physics_engine, sim_params)
        assert(sim is not None), "Failed to create simulator."
        return sim

    def _apply_cmd(self):
        if (self._has_dof()):
            if (self._control_mode == ControlMode.none):
                pass
            elif (self._control_mode == ControlMode.pos):
                cmd_buf = self._get_dof_cmd_buf()
                cmd_tensor = gymtorch.unwrap_tensor(cmd_buf)
                self._gym.set_dof_position_target_tensor(self._sim, cmd_tensor)
            elif (self._control_mode == ControlMode.vel):
                cmd_buf = self._get_dof_cmd_buf()
                cmd_tensor = gymtorch.unwrap_tensor(cmd_buf)
                self._gym.set_dof_velocity_target_tensor(self._sim, cmd_tensor)
            elif (self._control_mode == ControlMode.torque):
                pass
            elif (self._control_mode == ControlMode.pd_1d):
                pass
            else:
                assert(False), "Unsupported control mode: {}".format(self._control_mode)

        return

    def _pre_sim_step(self):
        if (self._apply_forces_callback is not None
            or self._control_mode == ControlMode.pd_1d):
            self._gym.refresh_dof_state_tensor(self._sim)

        if (self._apply_forces_callback is not None):
            self._apply_forces_callback()

        if (self._control_mode == ControlMode.torque):
            cmd_buf = self._get_dof_cmd_buf()
            self._set_actuation_torque(cmd_buf)

        elif (self._control_mode == ControlMode.pd_1d):
            torque = self._calc_pd_1d_torque()
            self._set_actuation_torque(torque)

        if (self._has_body_forces):
            self._gym.apply_rigid_body_force_tensors(self._sim, gymtorch.unwrap_tensor(self._body_forces_raw), 
                                                     None, gymapi.CoordinateSpace.GLOBAL_SPACE)
        return

    def _sim_step(self):
        self._gym.simulate(self._sim)
        if (self._device == "cpu"):
            self._gym.fetch_results(self._sim, True)
        return

    def _set_actuation_torque(self, torque):
        torque_clip = torch.clip(torque, -self._torque_lim_raw, self._torque_lim_raw)
        torque_tensor = gymtorch.unwrap_tensor(torque_clip)
        self._gym.set_dof_actuation_force_tensor(self._sim, torque_tensor)
        return
    
    def _flag_actor_needs_reset(self, env_id, actor_id):
        if (env_id is None):
            self._actors_need_reset[:, actor_id] = True
        else:
            self._actors_need_reset[env_id, actor_id] = True
        return
    
    def _refresh_sim_tensors(self):
        self._gym.refresh_dof_state_tensor(self._sim)
        self._gym.refresh_actor_root_state_tensor(self._sim)
        self._gym.refresh_rigid_body_state_tensor(self._sim)

        self._gym.refresh_force_sensor_tensor(self._sim)
        self._gym.refresh_net_contact_force_tensor(self._sim)
        
        if self._enable_dof_force_sensors():
            self._gym.refresh_dof_force_tensor(self._sim)
            
        self._has_body_forces = False
        self._body_forces_raw[:] = 0.0
        return
    
    def _control_mode_to_drive_mode(self, mode):
        if (mode == ControlMode.none):
            drive_mode = gymapi.DOF_MODE_NONE
        elif (mode == ControlMode.pos):
            drive_mode = gymapi.DOF_MODE_POS
        elif (mode == ControlMode.vel):
            drive_mode = gymapi.DOF_MODE_VEL
        elif (mode == ControlMode.torque):
            drive_mode = gymapi.DOF_MODE_EFFORT   
        elif (mode == ControlMode.pd_1d):
            drive_mode = gymapi.DOF_MODE_EFFORT 
        else:
            assert(False), "Unsupported control mode: {}".format(mode)
        return drive_mode
    
    def _parse_sim_params(self, config, sim_timestep):
        sim_params = gymapi.SimParams()
        sim_params.dt = sim_timestep
        sim_params.num_client_threads = 0
        
        sim_params.up_axis = gymapi.UP_AXIS_Z
        sim_params.gravity.x = 0
        sim_params.gravity.y = 0
        sim_params.gravity.z = -9.81

        sim_params.physx.solver_type = 1
        sim_params.physx.num_position_iterations = 4
        sim_params.physx.num_velocity_iterations = 0
        sim_params.physx.num_threads = 4
        sim_params.physx.num_subscenes = 0
        sim_params.physx.max_gpu_contact_pairs = 8 * 1024 * 1024

        if ("gpu" in self._device or "cuda" in self._device):
            sim_params.physx.use_gpu = True
            sim_params.use_gpu_pipeline = True
        elif ("cpu" in self._device):
            sim_params.physx.use_gpu = False
            sim_params.use_gpu_pipeline = False
        else:
            assert(False), "Unsupported simulation device: {}".format(self._device)

        # if sim options are provided in cfg, parse them and update/override above:
        if ("sim" in config):
            gymutil.parse_sim_config(config["sim"], sim_params)
        
        return sim_params
    
    def _get_device_idx(self):
        re_idx = re.search(r"\d", self._device)
        if (re_idx is None):
            device_idx = 0
        else:
            num_idx = re_idx.start()
            device_idx = int(self._device[num_idx:])
        return device_idx
    
    def _modify_control_mode_dof_prop(self, control_mode, dof_prop):
        if (control_mode == ControlMode.none):
            dof_prop["stiffness"] = 0.0
            dof_prop["damping"] = 0.0
        elif (control_mode == ControlMode.pos):
            pass
        elif (control_mode == ControlMode.vel):
            dof_prop["stiffness"] = 0.0
        elif (control_mode == ControlMode.torque):
            dof_prop["stiffness"] = 0.0
            dof_prop["damping"] = 0.0
        elif (control_mode == ControlMode.pd_1d):
            dof_prop["stiffness"] = 0.0
            dof_prop["damping"] = 0.0
        else:
            assert(False), "Unsupported control mode: {}".format(control_mode)

        return

    def _build_sim_tensors(self):
        root_state_tensor = self._gym.acquire_actor_root_state_tensor(self._sim)
        dof_state_tensor = self._gym.acquire_dof_state_tensor(self._sim)
        body_state_tensor = self._gym.acquire_rigid_body_state_tensor(self._sim)
        contact_force_tensor = self._gym.acquire_net_contact_force_tensor(self._sim)
        
        self._root_state_raw = gymtorch.wrap_tensor(root_state_tensor)
        self._dof_state_raw = gymtorch.wrap_tensor(dof_state_tensor)
        self._body_state_raw = gymtorch.wrap_tensor(body_state_tensor)
        self._contact_force_raw = gymtorch.wrap_tensor(contact_force_tensor)
        
        num_envs = self.get_num_envs()
        num_actors = self._root_state_raw.shape[0]
        self._actors_per_env = num_actors // num_envs

        root_state = self._root_state_raw.view([num_envs, self._actors_per_env, 
                                                self._root_state_raw.shape[-1]])
        
        self._root_state = root_state
        self._root_pos = root_state[..., 0:3]
        self._root_rot = root_state[..., 3:7]
        self._root_vel = root_state[..., 7:10]
        self._root_ang_vel = root_state[..., 10:13]

        if (self._dof_state_raw is not None):
            dofs_per_env = self._dof_state_raw.shape[0] // num_envs
            dof_state = self._dof_state_raw.view([num_envs, dofs_per_env, 2])
            self._dof_state = dof_state
            self._actor_dof_pos = []
            self._actor_dof_vel = []
            
            if self._enable_dof_force_sensors():
                dof_force_tensor = self._gym.acquire_dof_force_tensor(self._sim)
                self._dof_forces_raw = gymtorch.wrap_tensor(dof_force_tensor)
                dof_forces = self._dof_forces_raw.view([num_envs, dofs_per_env])
                self._actor_dof_forces = []

            dof_idx0 = 0
            for a in range(self._actors_per_env):
                actor_dof_dim = self.get_actor_dof_count(0, a)
                dof_idx1 = dof_idx0 + actor_dof_dim
                actor_dof_pos = dof_state[..., dof_idx0:dof_idx1, 0]
                actor_dof_vel = dof_state[..., dof_idx0:dof_idx1, 1]

                self._actor_dof_pos.append(actor_dof_pos)
                self._actor_dof_vel.append(actor_dof_vel)
                
                if self._enable_dof_force_sensors():
                    actor_dof_force = dof_forces[..., dof_idx0:dof_idx1]
                    self._actor_dof_forces.append(actor_dof_force)

                dof_idx0 = dof_idx1

            self._build_control_tensors()
        
        else:
            self._dof_state_raw = None

        bodies_per_env = self._body_state_raw.shape[0] // num_envs
        body_state = self._body_state_raw.view([num_envs, bodies_per_env, 
                                                                  self._body_state_raw.shape[-1]])

        contact_forces = self._contact_force_raw.view([num_envs, bodies_per_env, 3])

        self._body_forces_raw = torch.zeros_like(self._body_state_raw[..., :3])
        body_forces = self._body_forces_raw.view([num_envs, bodies_per_env, 3])

        # Note: NEVER use these tensors in observations calculations
        # they are not updated immediately during episode resets, and are not valid
        # until the second timestep after a reset
        self._actor_body_pos = []
        self._actor_body_rot = []
        self._actor_body_vel = []
        self._actor_body_ang_vel = []
        self._actor_contact_forces = []
        self._actor_body_forces = []
        self._has_body_forces = False

        env_ptr = self.get_env(0)
        body_idx0 = 0
        for a in range(self._actors_per_env):
            num_bodies = self._gym.get_actor_rigid_body_count(env_ptr, a)
            body_idx1 = body_idx0 + num_bodies

            actor_body_pos = body_state[..., body_idx0:body_idx1, 0:3]
            actor_body_rot = body_state[..., body_idx0:body_idx1, 3:7]
            actor_body_vel = body_state[..., body_idx0:body_idx1, 7:10]
            actor_body_ang_vel = body_state[..., body_idx0:body_idx1, 10:13]
            actor_body_forces = body_forces[..., body_idx0:body_idx1, :]
            actor_contact_forces = contact_forces[..., body_idx0:body_idx1, :]

            self._actor_body_pos.append(actor_body_pos)
            self._actor_body_rot.append(actor_body_rot)
            self._actor_body_vel.append(actor_body_vel)
            self._actor_body_ang_vel.append(actor_body_ang_vel)
            self._actor_body_forces.append(actor_body_forces)
            self._actor_contact_forces.append(actor_contact_forces)

            body_idx0 = body_idx1
        
        self._need_reset_buf = torch.zeros(num_actors, device=self._device, dtype=torch.bool)
        self._actors_need_reset = self._need_reset_buf.view((num_envs, self._actors_per_env))
        self._actor_dof_dims = self._build_actor_dof_dims()

        return
    
    def _build_control_tensors(self):
        self._kp_raw = [torch.cat(kp) for kp in self._actor_kp]
        self._kp_raw = torch.stack(self._kp_raw, dim=0)
        
        self._kd_raw = [torch.cat(kd) for kd in self._actor_kd]
        self._kd_raw = torch.stack(self._kd_raw, dim=0)
        
        self._torque_lim_raw = [torch.cat(lim) for lim in self._actor_torque_lim]
        self._torque_lim_raw = torch.stack(self._torque_lim_raw, dim=0)

        self._dof_cmd_raw = torch.zeros_like(self._kp_raw)

        self._actor_dof_cmd = []
        self._actor_kp = []
        self._actor_kd = []
        self._actor_torque_lim = []

        dof_idx0 = 0
        actors_per_env = self.get_actors_per_env()
        for a in range(actors_per_env):
            actor_dof_dim = self.get_actor_dof_count(0, a)
            dof_idx1 = dof_idx0 + actor_dof_dim
            actor_dof_cmd = self._dof_cmd_raw[..., dof_idx0:dof_idx1]
            actor_kp = self._kp_raw[..., dof_idx0:dof_idx1]
            actor_kd = self._kd_raw[..., dof_idx0:dof_idx1]
            actor_torque_lim = self._torque_lim_raw[..., dof_idx0:dof_idx1]

            self._actor_dof_cmd.append(actor_dof_cmd)
            self._actor_kp.append(actor_kp)
            self._actor_kd.append(actor_kd)
            self._actor_torque_lim.append(actor_torque_lim)

            dof_idx0 = dof_idx1

        return
    
    def _build_actor_dof_dims(self):
        num_envs = self.get_num_envs()
        actors_per_env = self.get_actors_per_env()

        actor_dof_dims = torch.zeros(num_envs * actors_per_env, device=self._device, dtype=torch.int)
        env_actor_dof_dims = actor_dof_dims.view([num_envs, actors_per_env])

        for e in range(num_envs):
            for a in range(actors_per_env):
                num_dofs = self.get_actor_dof_count(e, a)
                env_actor_dof_dims[e, a] = num_dofs

        return actor_dof_dims

    def _build_viewer(self):
        # subscribe to keyboard shortcuts
        self._viewer = self._gym.create_viewer(
            self._sim, gymapi.CameraProperties())
        self._gym.subscribe_viewer_keyboard_event(
            self._viewer, gymapi.KEY_ESCAPE, "QUIT")
        self._gym.subscribe_viewer_keyboard_event(
            self._viewer, gymapi.KEY_V, "toggle_viewer_sync")

        # set the camera position based on up axis
        sim_params = self._gym.get_sim_params(self._sim)
        if sim_params.up_axis == gymapi.UP_AXIS_Z:
            cam_pos = gymapi.Vec3(20.0, 25.0, 3.0)
            cam_target = gymapi.Vec3(10.0, 15.0, 0.0)
        else:
            cam_pos = gymapi.Vec3(20.0, 3.0, 25.0)
            cam_target = gymapi.Vec3(10.0, 0.0, 15.0)

        self._gym.viewer_camera_look_at(
            self._viewer, None, cam_pos, cam_target)

        return
    
    def _enable_dof_force_sensors(self):
        enable_dof_force_sensors = False
        return (enable_dof_force_sensors and self._has_dof())
    
    def _has_dof(self):
        return self._dof_state_raw is not None

    def _get_env_spacing(self):
        return self._env_spacing

    def _get_dof_cmd_buf(self):
        return self._dof_cmd_raw
    
    def _calc_pd_1d_torque(self):
        dof_pos = self._dof_state[..., :, 0]
        dof_vel = self._dof_state[..., :, 1]
        tar_dof = self._get_dof_cmd_buf()

        torque = self._kp_raw * (tar_dof - dof_pos) - self._kd_raw * dof_vel
        return torque
