import argparse
import pickle
import numpy as np
import anim.motion as motion

def convert_gmr_to_mimickit(gmr_file_path, output_file_path, loop_mode=motion.LoopMode.WRAP):
    """
    Convert a GMR compatible motion dataset to MimicKit compatible dataset.
    
    Args:
        gmr_file_path (str): Path to the GMR format pickle file
        output_file_path (str): Path to save the MimicKit format pickle file
        loop_mode (bool): Whether the motion should loop (default: False)
    """
    # Load GMR format data
    with open(gmr_file_path, 'rb') as f:
        gmr_data = pickle.load(f)
    
    # Extract data from GMR format
    fps = gmr_data['fps']
    root_pos = gmr_data['root_pos']  # Shape: (num_frames, 3)
    root_rot = gmr_data['root_rot']  # Shape: (num_frames, 4)
    dof_pos = gmr_data['dof_pos']    # Shape: (num_frames, 22)
    
    # Stack all motion data along the last dimension
    # frames shape: (num_frames, 3 + 4 + 22) = (num_frames, 29)
    frames = np.concatenate([root_pos, root_rot, dof_pos], axis=1)

    # Chop frames
    frames = frames[60:300, :]
    
    # Create MimicKit Motion object
    motion_class = motion.Motion(fps=fps, loop_mode=loop_mode, frames=frames)
    
    # Save as MimicKit format
    with open(output_file_path, 'wb') as f:
        pickle.dump(motion_class, f)
    
    print(f"Converted motion from {gmr_file_path} to {output_file_path}")
    print(f"Motion info: {frames.shape[0]} frames, {fps} fps, loop={loop_mode}")
    
    return motion_class

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert GMR motion data to MimicKit format.")
    parser.add_argument("--input_file", help="Path to the input GMR pickle file")
    parser.add_argument("--output_file", help="Path to the output MimicKit pickle file")
    # parser.add_argument("--loop", action="store_true", help="Enable loop mode on the converted motion")
    args = parser.parse_args()

    convert_gmr_to_mimickit(args.input_file, args.output_file)