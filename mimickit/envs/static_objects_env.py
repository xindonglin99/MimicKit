import envs.add_env as add_env
import numpy as np

class StaticObjectsEnv(add_env.ADDEnv):
    def __init__(self, config, num_envs, device, visualize):
        super().__init__(config=config, num_envs=num_envs, device=device, visualize=visualize)
        return
    
    def _build_env(self, env_id, config):
        super()._build_env(env_id, config)
        self._build_static_object(env_id, config)
        return
    
    def _build_static_object(self, env_id, config):
        objs_config = config["env"]["objects"]

        for i, obj_config in enumerate(objs_config):
            asset_file = obj_config["file"]
            pos = obj_config["pos"]
            rot = obj_config.get("rot", [0.0, 0.0, 0.0, 1.0])

            pos = np.array(pos)
            rot = np.array(rot)

            obj_name = "static_object{:d}".format(i)
            self._engine.create_actor(env_id=env_id,
                                      asset_file=asset_file,
                                      name=obj_name,
                                      start_pos=pos,
                                      start_rot=rot,
                                      fix_base=True)
        return