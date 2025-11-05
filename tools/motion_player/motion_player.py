
import sys
import torch
import os
import argparse
import numpy as np

sys.path.append(".")  # Ensure the repository root is on sys.path so we can use some utilities.
sys.path.append("mimickit")

from mimickit.anim.motion_lib import MotionLib, extract_pose_data
from mimickit.anim.motion import Motion
from mimickit.anim.kin_char_model import KinCharModel
from mimickit.util import torch_util
from pathlib import Path

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def build_kin_char_model(char_file):
    _, file_ext = os.path.splitext(char_file)
    if (file_ext == ".xml"):
        char_model = KinCharModel(DEVICE)
    else:
        print("Unsupported character file format: {:s}".format(file_ext))
        assert(False)

    kin_char_model = char_model
    kin_char_model.load_char_file(char_file)
    return kin_char_model   


def load_motion_lib(motion_file, kin_char_model):
    motion_lib = MotionLib(motion_file=motion_file,
                           kin_char_model=kin_char_model,
                           device=DEVICE)
    return motion_lib


def analyze_ground_clearance(motion_lib: MotionLib, kin_char_model: KinCharModel):
    """
    Analyze the minimum ground clearance for all bodies across all motions and time frames.
    """
    print("\n" + "="*80)
    print("GROUND CLEARANCE ANALYSIS")
    print("="*80)
    
    num_motions = motion_lib.get_num_motions()
    body_names = kin_char_model.get_body_names()
    
    # Find indices for left and right ankle roll bodies
    left_ankle_idx = None
    right_ankle_idx = None
    for i, name in enumerate(body_names):
        if name == "l_ankle_roll_link":
            left_ankle_idx = i
        elif name == "r_ankle_roll_link":
            right_ankle_idx = i
    
    if left_ankle_idx is None or right_ankle_idx is None:
        print("⚠️  Warning: Could not find ankle roll links in body names")
        print(f"   Available bodies: {body_names}")
    
    # Global minimums
    global_min_feet = float('inf')
    global_min_all = float('inf')
    
    feet_info = {"motion_id": None, "frame": None, "body": None, "height": None}
    all_info = {"motion_id": None, "frame": None, "body": None, "height": None}
    
    # Analyze all motions at once (fully vectorized)
    print(f"\nAnalyzing {num_motions} motion(s) with total {motion_lib._frame_root_pos.shape[0]} frames...")
    
    # Use the pre-concatenated frame data from motion_lib (all motions, all frames)
    # These tensors have all frames from all motions concatenated along dim 0
    all_root_pos = motion_lib._frame_root_pos  # shape: (total_frames, 3)
    all_root_rot = motion_lib._frame_root_rot  # shape: (total_frames, 4)
    all_joint_rot = motion_lib._frame_joint_rot  # shape: (total_frames, num_joints, 4)
    
    # Perform forward kinematics on ALL frames from ALL motions at once
    body_pos_all, body_rot_all = kin_char_model.forward_kinematics(
        all_root_pos, 
        all_root_rot, 
        all_joint_rot
    )
    # body_pos_all shape: (total_frames, num_bodies, 3)
    
    # Extract z-coordinates (height) for all frames and bodies
    all_body_heights = body_pos_all[:, :, 2]  # shape: (total_frames, num_bodies)
    
    # Now process per-motion statistics using the start indices
    frame_start_idx = 0
    for motion_id in range(num_motions):
        motion = motion_lib.get_motion(motion_id)
        num_frames = motion.frames.shape[0]
        frame_end_idx = frame_start_idx + num_frames
        
        print(f"\nMotion {motion_id}: {num_frames} frames, length={motion.get_length():.3f}s")
        
        # Slice the heights for this motion
        body_heights = all_body_heights[frame_start_idx:frame_end_idx]  # shape: (num_frames, num_bodies)
        
        # Check feet (ankle roll links) across all frames
        if left_ankle_idx is not None and right_ankle_idx is not None:
            left_ankle_heights = body_heights[:, left_ankle_idx]  # shape: (num_frames,)
            right_ankle_heights = body_heights[:, right_ankle_idx]  # shape: (num_frames,)
            
            # Get minimum for each frame, then find global minimum
            feet_heights = torch.minimum(left_ankle_heights, right_ankle_heights)
            min_frame_idx = torch.argmin(feet_heights).item()
            min_feet_height = feet_heights[min_frame_idx].item()
            
            if min_feet_height < global_min_feet:
                global_min_feet = min_feet_height
                feet_info = {
                    "motion_id": motion_id,
                    "frame": min_frame_idx,
                    "body": "l_ankle_roll_link" if left_ankle_heights[min_frame_idx] < right_ankle_heights[min_frame_idx] else "r_ankle_roll_link",
                    "height": min_feet_height
                }
        
        # Check all bodies across all frames
        min_heights_per_frame = torch.min(body_heights, dim=1)  # min across bodies for each frame
        min_frame_idx = torch.argmin(min_heights_per_frame.values).item()
        min_body_idx = min_heights_per_frame.indices[min_frame_idx].item()
        min_height = min_heights_per_frame.values[min_frame_idx].item()
        
        if min_height < global_min_all:
            global_min_all = min_height
            all_info = {
                "motion_id": motion_id,
                "frame": min_frame_idx,
                "body": body_names[min_body_idx],
                "height": min_height
            }
        
        frame_start_idx = frame_end_idx
    
    # Print results
    print("\n" + "="*80)
    print("RESULTS")
    print("="*80)
    
    if left_ankle_idx is not None and right_ankle_idx is not None:
        print("\n📏 FEET CLEARANCE (l_ankle_roll_link & r_ankle_roll_link):")
        print(f"   Minimum height: {global_min_feet:.6f} m")
        if feet_info["motion_id"] is not None:
            print(f"   Occurred at: Motion {feet_info['motion_id']}, Frame {feet_info['frame']}")
            print(f"   Body: {feet_info['body']}")
        
        if global_min_feet < 0:
            print(f"\n⚠️  WARNING: Feet penetrate ground by {abs(global_min_feet):.6f} m")
    
    print("\n📏 ALL BODIES CLEARANCE:")
    print(f"   Minimum height: {global_min_all:.6f} m")
    if all_info["motion_id"] is not None:
        print(f"   Occurred at: Motion {all_info['motion_id']}, Frame {all_info['frame']}")
        print(f"   Body: {all_info['body']}")
    
    if global_min_all < 0:
        print(f"\n⚠️  WARNING: Body '{all_info['body']}' penetrates ground by {abs(global_min_all):.6f} m")
    
    print("\n" + "="*80 + "\n")
    
    return global_min_feet, global_min_all


def main():
    parser = argparse.ArgumentParser(description='Motion player for MimicKit')
    parser.add_argument('motion_path', 
                        help='Path to motion file relative to repository root')
    parser.add_argument('--char_file', 
                        default='data/assets/hightorque/pi_plus/pi_22dof.xml',
                        help='Path to character file relative to repository root (default: data/assets/hightorque/pi_plus/pi_22dof.xml)')
    
    args = parser.parse_args()
    
    # Assume repository root is two parents above this file (MimicKit/)
    repo_root = Path(__file__).resolve().parents[2]
    
    # Construct full path to character file
    char_file = str(repo_root / args.char_file)
    
    if not Path(char_file).exists():
        print(f"Could not find character file: {char_file}")
        sys.exit(1)
    
    print(f"Using character file: {char_file}")

    kin_char_model = build_kin_char_model(char_file)

    # Construct full path to motion file
    motion_file = str(repo_root / args.motion_path)
    
    if not Path(motion_file).exists():
        print(f"Could not find motion file: {motion_file}")
        sys.exit(1)
    
    motion_lib = load_motion_lib(motion_file, kin_char_model)
    print(f"Loaded motion library: {motion_file}")
    
    # Analyze ground clearance
    min_feet_clearance, min_all_clearance = analyze_ground_clearance(motion_lib, kin_char_model)



if __name__ == "__main__":
    main()