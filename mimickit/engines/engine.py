import abc
import enum

class ControlMode(enum.Enum):
    none = 0
    pos = 1
    vel = 2
    torque = 3
    pd_1d = 4

class Engine:
    def __init__(self):
        return
    
    @abc.abstractmethod
    def create_env(self, env_id):
        return
    
    @abc.abstractmethod
    def create_actor(self, env_id, asset_file, name, is_visual=False, enable_self_collisions=True, 
                     fix_base=False, start_pos=None, start_rot=None, color=None, disable_motors=False):
        return

    @abc.abstractmethod
    def finalize_sim(self):
        return
    
    @abc.abstractmethod
    def set_cmd(self, actor_id, cmd):
        return

    @abc.abstractmethod
    def step(self):
        return

    @abc.abstractmethod
    def update_sim_state(self):
        return
    
    @abc.abstractmethod
    def update_camera(self, pos, look_at):
        return
    
    @abc.abstractmethod
    def get_camera_pos(self):
        return
    
    @abc.abstractmethod
    def render(self):
        return
    
    @abc.abstractmethod
    def get_timestep(self):
        return 0.0
    
    @abc.abstractmethod
    def get_num_envs(self):
        return 0
    
    @abc.abstractmethod
    def get_root_pos(self, actor_id):
        return
    
    @abc.abstractmethod
    def get_root_rot(self, actor_id):
        return
    
    @abc.abstractmethod
    def get_root_vel(self, actor_id):
        return
    
    @abc.abstractmethod
    def get_root_ang_vel(self, actor_id):
        return
    
    @abc.abstractmethod
    def get_dof_pos(self, actor_id):
        return
    
    @abc.abstractmethod
    def get_dof_vel(self, actor_id):
        return
    
    @abc.abstractmethod
    def get_dof_forces(self, actor_id):
        return
    
    @abc.abstractmethod
    def get_body_pos(self, actor_id):
        return
    
    @abc.abstractmethod
    def get_body_rot(self, actor_id):
        return
    
    @abc.abstractmethod
    def get_body_vel(self, actor_id):
        return
    
    @abc.abstractmethod
    def get_body_ang_vel(self, actor_id):
        return
    
    @abc.abstractmethod
    def get_contact_forces(self, actor_id):
        return

    
    @abc.abstractmethod
    def set_root_pos(self, env_id, actor_id, root_pos):
        return
    
    @abc.abstractmethod
    def set_root_rot(self, env_id, actor_id, root_rot):
        return
    
    @abc.abstractmethod
    def set_root_vel(self, env_id, actor_id, root_vel):
        return
    
    @abc.abstractmethod
    def set_root_ang_vel(self, env_id, actor_id, root_ang_vel):
        return
    
    @abc.abstractmethod
    def set_dof_pos(self, env_id, actor_id, dof_pos):
        return
    
    @abc.abstractmethod
    def set_dof_vel(self, env_id, actor_id, dof_vel):
        return
    
    @abc.abstractmethod
    def set_body_vel(self, env_id, actor_id, body_vel):
        return
    
    @abc.abstractmethod
    def set_body_ang_vel(self, env_id, actor_id, body_ang_vel):
        return
    
    @abc.abstractmethod
    def set_body_forces(self, env_id, actor_id, body_id, forces):
        return
    
    
    @abc.abstractmethod
    def get_actor_dof_count(self, env_id, actor_id):
        return
    
    @abc.abstractmethod
    def get_actor_body_names(self, env_id, actor_id):
        return

    @abc.abstractmethod
    def find_actor_body_id(self, env_id, actor_id, body_name):
        return 
    
    @abc.abstractmethod
    def get_actor_torque_lim(self, actor_id):
        return
    
    @abc.abstractmethod
    def get_actor_dof_limits(self, env_id, actor_id):
        return
    
    @abc.abstractmethod
    def calc_actor_mass(self, env_id, actor_id):
        return
    
    @abc.abstractmethod
    def get_control_mode(self):
        return
    
    def draw_lines(self, env_id, verts, cols):
        return