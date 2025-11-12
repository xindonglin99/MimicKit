import os
import yaml
from datetime import date

# === Base paths ===
base_env_path = "data/envs/deepmimic_pi_plus_env.yaml"
output_env_dir = "data/envs/exps"
output_arg_dir = "args/exps"
os.makedirs(output_env_dir, exist_ok=True)
os.makedirs(output_arg_dir, exist_ok=True)

# === List of motion files to experiment with ===
motion_files = [
    "data/motions/hightorque/pi_plus/mimickit/walk_50fps_highdist.pkl",
    "data/motions/hightorque/pi_plus/mimickit/walk_50fps_lowdist.pkl",
]

# === Load base env config ===
with open(base_env_path, "r") as f:
    base_env = yaml.safe_load(f)

# === Generate experiment-specific env + arg files ===
for i, motion in enumerate(motion_files):
    exp_name = f"deepmimic_exp{i+1}"
    env_path = os.path.join(output_env_dir, f"{exp_name}.yaml")
    arg_path = os.path.join(output_arg_dir, f"{exp_name}_args.txt")

    # Update motion file path
    base_env["env"]["motion_file"] = motion

    # Save new env YAML
    with open(env_path, "w") as f:
        yaml.dump(base_env, f)

    # === Output directory structure ===
    motion_name = os.path.basename(motion).split(".")[0]
    today = date.today()
    date_str = f"{today.strftime('%b')}{today.day}"
    output_dir = f"output/pi_plus/deepmimic/{date_str}/{motion_name}"
    os.makedirs(output_dir, exist_ok=True)

    # === Generate argument file ===
    arg_text = f"""--num_envs 4096
--env_config {env_path}
--agent_config data/agents/deepmimic_pi_plus_ppo_agent.yaml
--log_file {output_dir}/log.txt
--out_model_file {output_dir}/model.pt
--max_samples 300000000
"""

    with open(arg_path, "w") as f:
        f.write(arg_text)

    print(f"✅ Generated:\n  {env_path}\n  {arg_path}")

print("\n🎉 All experiment configs generated successfully.")
