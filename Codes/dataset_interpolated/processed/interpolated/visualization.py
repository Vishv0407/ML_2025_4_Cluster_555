import os
import pandas as pd
from PIL import Image, ImageDraw
import cv2
import numpy as np
import matplotlib.pyplot as plt
from math import ceil

# Get the absolute base directory relative to the script location
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.join(SCRIPT_DIR)  # Current directory contains subdirs 10, 11, 12

# Step 1: Load appropriate base image based on subdirectory
def load_base_image(subdir):
    image_mapping = {
        '10': os.path.join(SCRIPT_DIR, 'sample10.jpg'),
        '11': os.path.join(SCRIPT_DIR, 'sample11.jpg'),
        '12': os.path.join(SCRIPT_DIR, 'sample12.jpg')
    }
    image_path = image_mapping.get(subdir, os.path.join(SCRIPT_DIR, 'sample10.jpg'))  # Default to sample10.jpg
    try:
        base_image = Image.open(image_path).convert('RGB')
        return base_image
    except FileNotFoundError:
        print(f"Error: {image_path} not found. Please upload the corresponding image for subdir {subdir} to {SCRIPT_DIR}.")
        return None

# Step 2: Process CSV files to extract trajectory points
def process_trajectories():
    trajectories = []
    for root, _, files in os.walk(BASE_DIR):
        subdir = os.path.basename(os.path.dirname(root)) if os.path.basename(root) in ['10', '11', '12'] else '10'
        for file in files:
            if file.endswith('.csv'):
                folder_name = os.path.basename(os.path.dirname(root))
                label = 'abnormal' if 'abnormal' in folder_name else 'normal'
                color = 'red' if label == 'abnormal' else 'green'
                df = pd.read_csv(os.path.join(root, file))
                if all(col in df.columns for col in ['frameNo', 'left', 'top', 'w', 'h']):
                    points = []
                    for _, row in df.iterrows():
                        frame = int(row['frameNo'])
                        x_center = row['left'] + row['w'] / 2
                        y_center = row['top'] + row['h'] / 2
                        points.append((frame, x_center, y_center))
                    trajectories.append({'points': points, 'color': color, 'label': label, 'file': file, 'path': os.path.join(root, file), 'subdir': subdir})
    return trajectories

# Step 3: Draw trajectories on the image for a specific frame
def draw_trajectories(base_image, trajectories, frame, max_frame):
    img = base_image.copy()
    draw = ImageDraw.Draw(img)
    
    for traj in trajectories:
        color = traj['color']
        points = [(p[1], p[2]) for p in traj['points'] if p[0] <= frame]
        last_frame = max(p[0] for p in traj['points']) if traj['points'] else 0
        
        if last_frame >= frame and points:
            for i in range(len(points) - 1):
                draw.line([points[i], points[i + 1]], fill=color, width=4)
            
            # Draw larger vehicle dot at current position
            for p in traj['points']:
                if p[0] == frame:
                    x, y = p[1], p[2]
                    draw.ellipse([(x-10, y-10), (x+10, y+10)], fill=color, outline='black')
    
    img_np = np.array(img)
    return img_np

# Step 4: Create individual video for each trajectory
def create_individual_video(base_image, trajectory, output_path, fps=30, duration=2):
    max_frame = max(p[0] for p in trajectory['points']) if trajectory['points'] else 0
    total_frames = int(duration * fps)
    frame_step = max(1, max_frame // total_frames) if max_frame > 0 else 1
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(output_path, fourcc, fps, (base_image.width, base_image.height))
    
    for frame in range(0, min(max_frame + 1, total_frames * frame_step), frame_step):
        img_np = draw_trajectories(base_image, [trajectory], frame, max_frame)
        video_writer.write(cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR))
    
    video_writer.release()
    print(f"Individual video saved as {output_path}")

# Step 5: Create combined videos with up to 30 trajectories each
def create_combined_videos(base_image, trajectories, fps=30, duration=2):
    total_trajectories = len(trajectories)
    num_videos = ceil(total_trajectories / 30)
    total_frames = int(duration * fps)
    
    for video_idx in range(num_videos):
        start_idx = video_idx * 30
        end_idx = min((video_idx + 1) * 30, total_trajectories)
        current_trajectories = trajectories[start_idx:end_idx]
        
        # Use the base image corresponding to the first trajectory's subdirectory
        subdir = current_trajectories[0]['subdir'] if current_trajectories else '10'
        base_image = load_base_image(subdir)
        if base_image is None:
            return
        
        max_frame = max(max(p[0] for p in t['points']) for t in current_trajectories) if current_trajectories else 0
        frame_step = max(1, max_frame // total_frames) if max_frame > 0 else 1
        
        output_path = os.path.join(SCRIPT_DIR, f'combined_trajectories_video_{video_idx + 1}.mp4')
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(output_path, fourcc, fps, (base_image.width, base_image.height))
        
        for frame in range(0, min(max_frame + 1, total_frames * frame_step), frame_step):
            img_np = draw_trajectories(base_image, current_trajectories, frame, max_frame)
            video_writer.write(cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR))
        
        video_writer.release()
        print(f"Combined video {video_idx + 1}/{num_videos} saved as {output_path}")

# Main execution
def main():
    # Set working directory to script location
    os.chdir(SCRIPT_DIR)
    
    # Process trajectories first to determine subdirectories
    trajectories = process_trajectories()
    if not trajectories:
        print("No valid trajectories found.")
        return
    
    # Create individual videos for each trajectory
    for traj in trajectories:
        base_image = load_base_image(traj['subdir'])
        if base_image is None:
            continue
        output_path = os.path.join(SCRIPT_DIR, f'trajectory_{traj["file"].split(".")[0]}.mp4')
        create_individual_video(base_image, traj, output_path)
    
    # Create combined videos (up to 30 trajectories per video)
    create_combined_videos(load_base_image('10'), trajectories)  # Default to sample10.jpg for combined if needed

if __name__ == "__main__":
    main()