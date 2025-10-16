import envs.base_env as base_env
import envs.char_env as char_env
import anim.motion as motion
import anim.motion_lib as motion_lib

import numpy as np
import torch


class ViewMotionEnv(char_env.CharEnv):
    def __init__(self, config, num_envs, device, visualize):
        self._time_scale = 1.0
        self._motion_rotation_type = config["env"].get("motion_rotation_type", motion_lib.RotationType.EXP_MAP.value)

        super().__init__(config=config, num_envs=num_envs, device=device,
                         visualize=visualize)
        
        return

    def _build_envs(self, config, num_envs):
        super()._build_envs(config, num_envs)

        motion_file = config["env"]["motion_file"]
        self._load_motions(motion_file)
        return

    def _load_motions(self, motion_file):
        self._motion_lib = motion_lib.MotionLib(motion_file=motion_file, 
                                                kin_char_model=self._kin_char_model,
                                                device=self._device,
                                                motion_rotation_type=self._motion_rotation_type)
        return

    def _update_misc(self):
        super()._update_misc()
        self._sync_motion()
        return

    def _apply_action(self, actions):
        return

    def _sync_motion(self):
        motion_ids = self._get_env_motion_ids()
        motion_times = self._time_buf * self._time_scale
        root_pos, root_rot, root_vel, root_ang_vel, joint_rot, dof_vel = self._motion_lib.calc_motion_frame(motion_ids, motion_times)
        joint_dof = self._motion_lib.joint_rot_to_dof(joint_rot)
        
        char_id = self._get_char_id()
        
        self._engine.set_root_pos(None, char_id, root_pos)
        self._engine.set_root_rot(None, char_id, root_rot)
        self._engine.set_root_vel(None, char_id, 0.0)
        self._engine.set_root_ang_vel(None, char_id, 0.0)
        
        self._engine.set_dof_pos(None, char_id, joint_dof)
        self._engine.set_dof_vel(None, char_id, 0.0)
        
        if (self._has_key_bodies()):
            body_pos, body_rot = self._kin_char_model.forward_kinematics(root_pos=root_pos,
                                                                         root_rot=root_rot,
                                                                         joint_rot=joint_rot)
            self._ref_body_pos[:] = body_pos

        return

    def _render(self):
        super()._render()
        self._render_key_points()
        return
    
    def _build_sim_tensors(self, config):
        super()._build_sim_tensors(config)
        
        if (self._has_key_bodies()):
            char_id = self._get_char_id()
            body_pos = self._engine.get_body_pos(char_id)
            self._ref_body_pos = torch.zeros_like(body_pos)
        return

    def _get_env_motion_ids(self):
        num_envs = self.get_num_envs()
        num_motions = self._motion_lib.get_num_motions()

        motion_ids = torch.arange(num_envs, device=self._device)
        motion_ids = torch.remainder(motion_ids, num_motions)

        return motion_ids

    def _update_done(self):
        motion_ids = self._get_env_motion_ids()
        motion_len = self._motion_lib.get_motion_length(motion_ids)
        motion_loop_mode = self._motion_lib.get_motion_loop_mode(motion_ids)
        self._done_buf[:] = compute_done(self._done_buf, self._time_buf, 
                                         motion_len, motion_loop_mode)
        return

    def _render_key_points(self):
        if (self._has_key_bodies()):
            num_key_bodies = self._key_body_ids.shape[0]
            cols = np.array(3 * num_key_bodies * [[1.0, 0.0, 0.0]], dtype=np.float32)
            
            num_envs = self.get_num_envs()
            for i in range(num_envs):
                key_body_pos = self._ref_body_pos[i][self._key_body_ids]
                key_body_pos = key_body_pos.cpu().numpy()

                verts = 0.2 * np.array([[-1.0, 0.0, 0.0], [1.0, 0.0, 0.0],
                                        [0.0, -1.0, 0.0], [0.0, 1.0, 0.0],
                                        [0.0, 0.0, -1.0], [0.0, 0.0, 1.0]],
                                       dtype=np.float32)

                key_body_pos = np.expand_dims(key_body_pos, -2)
                verts = np.expand_dims(verts, 0)
                verts = key_body_pos + verts
                verts = np.reshape(verts, (-1, 6))

                self._engine.draw_lines(i, verts, cols)

        return
    
    def _get_char_color(self):
        return np.array([0.5, 0.9, 0.1])

    def _ig_build_character(self, env_id, config, color=None):
        col_group = env_id
        col_filter = 1
        segmentation_id = 0
        char_id = self._engine.create_actor(env_id=env_id, 
                                             asset=self._char_asset, 
                                             name="character", 
                                             col_group=col_group, 
                                             col_filter=col_filter, 
                                             segmentation_id=segmentation_id,
                                             color=color)
        return char_id



@torch.jit.script
def compute_done(done_buf, time, motion_len, motion_loop_mode):
    # type: (Tensor, Tensor, Tensor, Tensor) -> Tensor
    num_loops = 5

    timeout = torch.zeros_like(done_buf)
    end_time = motion_len.clone()
    loop_ids = motion_loop_mode == motion.LoopMode.WRAP.value
    end_time[loop_ids] *= num_loops

    timeout = time >= end_time
    done = torch.full_like(done_buf, base_env.DoneFlags.NULL.value)
    done[timeout] = base_env.DoneFlags.TIME.value

    return done