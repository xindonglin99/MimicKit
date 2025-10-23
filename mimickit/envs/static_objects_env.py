import envs.add_env as add_env
import numpy as np

class StaticObjectsEnv(add_env.ADDEnv):
    def __init__(self, config, num_envs, device, visualize):
        super().__init__(config=config, num_envs=num_envs, device=device, visualize=visualize)
        return
    

    ######################
    # Isaac Gym Builders
    ######################

    def _ig_build_envs(self, config, num_envs):
        self._ig_load_obj_assets(config)
        super()._ig_build_envs(config, num_envs)
        return
    
    def _ig_load_obj_assets(self, config):
        objs_config = config["env"]["objects"]
        self._obj_assets = []

        for obj_config in objs_config:
            asset_file = obj_config["file"]
            pos = obj_config["pos"]
            rot = obj_config.get("rot", [0.0, 0.0, 0.0, 1.0])

            obj_asset = self._engine.load_asset(asset_file, fix_base=True)
            pos = np.array(pos)
            rot = np.array(rot)

            obj_dict = {
                "asset": obj_asset,
                "pos": pos,
                "rot": rot
            }

            self._obj_assets.append(obj_dict)

        return
    
    def _ig_build_env(self, env_id, config):
        super()._ig_build_env(env_id, config)
        self._ig_build_static_object(env_id)
        return
    
    def _ig_build_static_object(self, env_id):
        for i, obj in enumerate(self._obj_assets):
            asset = obj["asset"]
            pos = obj["pos"]
            rot = obj["rot"]

            obj_name = "static_object{:d}".format(i)
            self._engine.create_actor(env_id=env_id,
                                      asset=asset,
                                      name=obj_name,
                                      col_group=env_id,
                                      col_filter=0,
                                      segmentation_id=0,
                                      start_pos=pos,
                                      start_rot=rot)
        return