"""
This module provides functionality to convert motion data from GMR format to MimicKit format. 

Usage:
    Command line:
        python tools/data_format/gmr_to_mimickit.py --input_file <path> --output_file <path> [options]
    Required arguments:
        --input_file PATH       Path to the input GMR pickle file
        --output_file PATH      Path to save the output MimicKit pickle file
    Optional arguments:
        --loop {wrap,clamp}     Loop mode for the motion (default: wrap)
        --start_frame INT       Start frame for motion clipping (default: 0)
        --end_frame INT         End frame for motion clipping (default: -1, uses all frames)
    
GMR Format:
    The input GMR format should be a pickle file containing a dictionary with keys:
    - 'fps': Frame rate (int)
    - 'root_pos': Root position array, shape (num_frames, 3)
    - 'root_rot': Root rotation quaternions, shape (num_frames, 4) (x, y, z, w)
    - 'dof_pos': Degrees of freedom positions, shape (num_frames, num_dofs)
    - 'local_body_pos': Currently unused (can be None)
    - 'link_body_list': Currently unused (can be None)

Output:
    Creates a MimicKit Motion object saved as a pickle file, with motion data stored as
    concatenated arrays of [root_pos, root_rot_expmap, dof_pos] per frame.
"""

import argparse
import pickle
import numpy as np
import sys

import torch

sys.path.append("mimickit")  # Ensure the repository root is on sys.path so we can import mimickit when executed directly.

import anim.motion as motion
from util.torch_util import quat_to_exp_map

def convert_gmr_to_mimickit(gmr_file_path, output_file_path, loop_mode, start_frame, end_frame):
    """
    Convert a GMR compatible motion dataset to MimicKit compatible dataset.
    
    Args:
        gmr_file_path (str): Path to the GMR format pickle file
        output_file_path (str): Path to save the MimicKit format pickle file
        loop_mode (bool): Whether the motion should loop (Set to wrap as default)
    """
    if loop_mode == "wrap":
        loop_mode = motion.LoopMode.WRAP
    elif loop_mode == "clamp":
        loop_mode = motion.LoopMode.CLAMP
    else:
        raise ValueError(f"Invalid loop_mode: {loop_mode}. Choose 'wrap' or 'clamp'.")
    
    # Load GMR format data
    with open(gmr_file_path, 'rb') as f:
        gmr_data = pickle.load(f)
    
    # Extract data from GMR format
    fps = gmr_data['fps']
    root_pos = gmr_data['root_pos']  # Shape: (num_frames, 3)
    root_rot_quat = gmr_data['root_rot']  # Shape: (num_frames, 4), quaternion format
    dof_pos = gmr_data['dof_pos']    # Shape: (num_frames, num_dofs)
    
    # Verify shapes
    if root_pos.ndim != 2 or root_pos.shape[1] != 3:
        raise ValueError(f"Expected root_pos shape (num_frames, 3), got {root_pos.shape}")
        
    if root_rot_quat.ndim != 2 or root_rot_quat.shape[1] != 4:
        raise ValueError(f"Expected root_rot_quat shape (num_frames, 4), got {root_rot_quat.shape}")
        
    if dof_pos.ndim != 2:
        raise ValueError(f"Expected dof_pos to be 2D array, got {dof_pos.ndim}D")

    ## NOTE: Here we might get some q = -q case but it should be fine.
    root_rot = quat_to_exp_map(torch.tensor(root_rot_quat)).numpy()  # Convert quaternion to exponential map
    
    # Stack all motion data along the last dimension
    # frames shape: (num_frames, 3 + 3 + num_dofs) = (num_frames, 6 + num_dofs)
    frames = np.concatenate([root_pos, root_rot, dof_pos], axis=1)

    # Chop frames
    if end_frame == -1:
        end_frame = frames.shape[0]
    assert 0 <= start_frame < end_frame <= frames.shape[0], "Invalid start_frame or end_frame."
    frames = frames[start_frame:end_frame, :]
    
    # Create MimicKit Motion object
    motion_class = motion.Motion(fps=fps, loop_mode=loop_mode, frames=frames)
    
    # Save as MimicKit format
    motion_class.save(output_file_path)
    
    print(f"Converted motion from {gmr_file_path} to {output_file_path}")
    print(f"Motion info: {frames.shape[0]} frames, {fps} fps, loop={loop_mode}")
    
    return motion_class

def main():
    parser = argparse.ArgumentParser(description="Convert GMR motion data to MimicKit format.")
    parser.add_argument("--input_file", required=True, help="Path to the input GMR pickle file")
    parser.add_argument("--output_file", required=True, help="Path to the output MimicKit pickle file")
    parser.add_argument("--loop", default="wrap", choices=["wrap", "clamp"], help="Enable loop mode on the converted motion")
    parser.add_argument("--start_frame", type=int, default=0, help="Start frame for chopping the motion")
    parser.add_argument("--end_frame", type=int, default=-1, help="End frame for chopping the motion")
    args = parser.parse_args()

    convert_gmr_to_mimickit(args.input_file, args.output_file, loop_mode=args.loop, start_frame=args.start_frame, end_frame=args.end_frame)


if __name__ == "__main__":
    main()