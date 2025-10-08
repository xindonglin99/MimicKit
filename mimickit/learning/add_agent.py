import torch

import learning.amp_agent as amp_agent
import learning.ppo_agent as ppo_agent
import learning.add_model as add_model
import util.torch_util as torch_util
import learning.diff_normalizer as diff_normalizer

class ADDAgent(amp_agent.AMPAgent):
    NAME = "ADD"

    def __init__(self, config, env, device):
        super().__init__(config, env, device)

        self._pos_diff = self._build_pos_diff()
        return
    
    def _build_model(self, config):
        model_config = config["model"]
        self._model = add_model.ADDModel(model_config, self._env)
        return
    
    def _build_pos_diff(self):
        disc_obs_space = self._env.get_disc_obs_space()
        disc_obs_dtype = torch_util.numpy_dtype_to_torch(disc_obs_space.dtype)
        pos_diff = torch.zeros(disc_obs_space.shape, device=self._device, dtype=disc_obs_dtype)
        return pos_diff
    
    def _build_normalizers(self):
        ppo_agent.PPOAgent._build_normalizers(self)

        disc_obs_space = self._env.get_disc_obs_space()
        disc_obs_dtype = torch_util.numpy_dtype_to_torch(disc_obs_space.dtype)
        self._disc_obs_norm = diff_normalizer.DiffNormalizer(disc_obs_space.shape, device=self._device, dtype=disc_obs_dtype)
        return
    
    def _record_data_post_step(self, next_obs, r, done, next_info):
        ppo_agent.PPOAgent._record_data_post_step(self, next_obs, r, done, next_info)

        disc_obs = next_info["disc_obs"]
        disc_obs_demo = next_info["disc_obs_demo"]
        self._exp_buffer.record("disc_obs_demo", disc_obs_demo)
        self._exp_buffer.record("disc_obs", disc_obs)
        
        if (self._need_normalizer_update()):
            obs_diff = disc_obs_demo - disc_obs
            self._disc_obs_norm.record(obs_diff)
        return
    
    def _build_train_data(self):
        task_r = self._exp_buffer.get_data_flat("reward")
        
        disc_obs = self._exp_buffer.get_data_flat("disc_obs")
        disc_obs_demo = self._exp_buffer.get_data_flat("disc_obs_demo")
        disc_r = self._calc_disc_rewards(disc_obs=disc_obs, disc_obs_demo=disc_obs_demo)
        
        r = self._task_reward_weight * task_r + self._disc_reward_weight * disc_r
        self._exp_buffer.set_data_flat("reward", r)

        info = super(amp_agent.AMPAgent, self)._build_train_data()

        disc_reward_std, disc_reward_mean = torch.std_mean(disc_r)
        info["disc_reward_mean"] = disc_reward_mean
        info["disc_reward_std"] = disc_reward_std

        return info
    
    def _calc_disc_rewards(self, disc_obs, disc_obs_demo):
        obs_diff = disc_obs_demo - disc_obs
        norm_obs_diff = self._disc_obs_norm.normalize(obs_diff)
        reward = super()._calc_disc_rewards(norm_obs_diff)
        return reward
    
    def _compute_disc_loss(self, batch):
        disc_obs = batch["disc_obs"]
        tar_disc_obs = batch["disc_obs_demo"]

        pos_diff = self._pos_diff
        pos_diff = pos_diff.unsqueeze(dim=0)
        
        disc_pos_logit = self._model.eval_disc(pos_diff)
        disc_pos_logit = disc_pos_logit.squeeze(-1)
        
        diff_obs = tar_disc_obs - disc_obs
        norm_diff_obs = self._disc_obs_norm.normalize(diff_obs)
        norm_diff_obs.requires_grad_(True)
        disc_neg_logit = self._model.eval_disc(norm_diff_obs)
        disc_neg_logit = disc_neg_logit.squeeze(-1)
        
        disc_loss_pos = self._disc_loss_pos(disc_pos_logit)
        disc_loss_neg = self._disc_loss_neg(disc_neg_logit)
        disc_loss = 0.5 * (disc_loss_pos + disc_loss_neg)

        # logit reg
        logit_weights = self._model.get_disc_logit_weights()
        disc_logit_loss = torch.sum(torch.square(logit_weights))
        disc_loss += self._disc_logit_reg * disc_logit_loss

        # grad penalty
        disc_neg_grad = torch.autograd.grad(disc_neg_logit, norm_diff_obs, grad_outputs=torch.ones_like(disc_neg_logit),
                                             create_graph=True, retain_graph=True, only_inputs=True)
        disc_neg_grad = disc_neg_grad[0]
        disc_neg_grad_squared = torch.sum(torch.square(disc_neg_grad), dim=-1)
        disc_grad_penalty = torch.mean(disc_neg_grad_squared)
        disc_loss += self._disc_grad_penalty * disc_grad_penalty

        # weight decay
        if (self._disc_weight_decay != 0):
            disc_weights = self._model.get_disc_weights()
            disc_weights = torch.cat(disc_weights, dim=-1)
            disc_weight_decay = torch.sum(torch.square(disc_weights))
            disc_loss += self._disc_weight_decay * disc_weight_decay

        disc_neg_acc, disc_pos_acc = self._compute_disc_acc(disc_neg_logit, disc_pos_logit)

        disc_pos_logit_mean = torch.mean(disc_pos_logit)
        disc_neg_logit_mean = torch.mean(disc_neg_logit)

        disc_info = {
            "disc_loss": disc_loss,
            "disc_grad_penalty": disc_grad_penalty.detach(),
            "disc_logit_loss": disc_logit_loss.detach(),
            "disc_pos_acc": disc_pos_acc.detach(),
            "disc_neg_acc": disc_neg_acc.detach(),
            "disc_pos_logit": disc_pos_logit_mean.detach(),
            "disc_neg_logit": disc_neg_logit_mean.detach()
        }
        return disc_info
