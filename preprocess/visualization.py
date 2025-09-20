"""
Visualization functions for LATTE preprocessing.
Functions for creating plots of stations, events, and velocity models.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import imageio
import shutil
from pathlib import Path
from omegaconf import DictConfig
from typing import Tuple

from utils.io_utils import ensure_directory_exists


def create_station_event_visualization(config: DictConfig) -> None:
    """
    Create 3D visualization of stations and events with synchronized viewing angles.
    """
    print("Generating station and event visualization...")
    
    # Load data
    stations_df = pd.read_csv(config.station_file)
    events_df = pd.read_csv(config.event_file)
    
    # Ensure non-negative depths
    stations_df['depth'] = stations_df['depth'].apply(lambda x: max(x, 0.0))
    events_df['depth'] = events_df['depth'].apply(lambda x: max(x, 0.0))
    
    # Create figure
    fig = plt.figure(figsize=(8.5, 8.5))
    fig.suptitle("Visualization of Stations and Receivers", fontsize=14)

    # View configuration
    view_elevation = config.view_elevation
    view_from_direction = config.view_from_direction
    
    # Calculate center longitude for geographic plot rotation
    center_lon = (stations_df['lon'].min() + stations_df['lon'].max()) / 2

    # Translate the chosen direction into specific azimuth angles
    direction_map = {
        'south': (180, center_lon),
        'east': (90, center_lon - 90),
        'west': (-90, center_lon + 90),
        'north': (0, center_lon + 180),
        'southeast': (135, center_lon - 45)
    }
    
    azim1, azim2 = direction_map.get(view_from_direction.lower(), (135, center_lon - 45))

    # Local Cartesian 3D Plot
    ax1 = fig.add_subplot(1, 1, 1, projection='3d')
    ax1.scatter(events_df['i'], events_df['j'], events_df['k'], 
                c='red', marker='o', s=15, label='Events', alpha=0.7)
    ax1.scatter(stations_df['i'], stations_df['j'], stations_df['k'], 
                c='blue', marker='^', s=40, label='Stations', depthshade=False)
    ax1.set_title("Local Cartesian View", fontsize=12)
    ax1.set_xlabel("X Grid (km - East)")
    ax1.set_ylabel("Y Grid (km - South)")
    ax1.set_zlabel("Z Grid / Depth (km)")
    ax1.invert_zaxis()
    ax1.legend()
    ax1.grid(True)
    ax1.set_box_aspect([1, 1, 1])
    ax1.view_init(elev=view_elevation, azim=azim1)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    # Ensure output directory exists
    ensure_directory_exists(Path(config.output_image).parent)
    
    try:
        print(f"Saving plot to: {config.output_image}")
        plt.savefig(config.output_image, dpi=300)
        print("Plot saved successfully.")
        plt.close()
    except Exception as e:
        print(f"Error saving plot: {e}")
        plt.close()


def auto_detect_dimensions_from_csv(csv_file: str) -> Tuple[int, int, int]:
    """Auto-detect grid dimensions from CSV file to match writing script."""
    df = pd.read_csv(csv_file)
    nx = int(df['i'].max())
    ny = int(df['j'].max()) 
    nz = int(df['k'].max())
    print(f"Auto-detected dimensions from CSV: NX={nx}, NY={ny}, NZ={nz}")
    return nx, ny, nz


def load_velocity_model_binary(bin_file: str, shape: Tuple[int, int, int]) -> np.ndarray:
    """
    Load a Fortran stream access binary file (raw binary with no record markers).
    This matches the format written by the writing script using tofile().
    """
    expected_n_floats = np.prod(shape)
    
    # Read raw binary data directly - no Fortran record markers
    data = np.fromfile(bin_file, dtype=np.float32)
    
    print(f"Binary file size: {data.size} floats, expected: {expected_n_floats}")
    
    if data.size != expected_n_floats:
        raise ValueError(f"Data size read ({data.size}) does not match expected grid size ({expected_n_floats})")
    
    # Reshape in Fortran order (column-major) to match writing script
    model = data.reshape(shape, order="F")
    
    # Convert from m/s to km/s to match CSV units
    model = model / 1000.0
    print(f"Velocity range after conversion: {model[model>0].min():.2f} - {model[model>0].max():.2f} km/s")
    
    return model


def plot_velocity_from_csv(config: DictConfig, output_png: str) -> None:
    """Create velocity visualization from CSV file."""
    df = pd.read_csv(config.csv_file)

    if config.coord_system == "flattened" or config.coord_system == "enu":
        x = df["X_grid_km"].values
        y = df["Y_grid_km"].values
        z = df["Z_grid_km"].values
        xlabel, ylabel, zlabel = "X (km)", "Y (km)", "Z (km)"
    else:
        raise ValueError("Unsupported coordinate system for this example.")

    v = df["vp"].values

    mask = v > 0
    x, y, z, v = x[mask], y[mask], z[mask], v[mask]

    fig = plt.figure(figsize=(8.5, 7))
    ax = fig.add_subplot(111, projection="3d")
    img = ax.scatter(x, y, z, c=v, cmap="viridis", marker=".", s=5)

    ax.invert_zaxis()
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_zlabel(zlabel)
    ax.set_title(f"Input Velocity Distribution", fontsize=12)
    cbar = fig.colorbar(img, ax=ax, label="Velocity (km/s)")
    cbar.set_label("Velocity (km/s)", fontsize=12)
    plt.tight_layout()
    
    ensure_directory_exists(Path(output_png).parent)
    plt.savefig(output_png, dpi=300)
    plt.close()
    print(f"Saved {output_png}")


def plot_velocity_from_bin(config: DictConfig, output_png: str, nx: int, ny: int, nz: int) -> None:
    """Create velocity visualization from binary file."""
    model = load_velocity_model_binary(config.bin_file, (nx, ny, nz))

    x, y, z = np.meshgrid(np.arange(nx), np.arange(ny), np.arange(nz), indexing="ij")

    x = x.flatten()
    y = y.flatten()
    z = z.flatten()
    v = model.flatten()

    mask = v > 0
    x, y, z, v = x[mask], y[mask], z[mask], v[mask]

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")
    img = ax.scatter(x, y, z, c=v, cmap="viridis", marker=".", s=5)

    ax.invert_zaxis()
    ax.set_xlabel("X Index")
    ax.set_ylabel("Y Index")
    ax.set_zlabel("Depth Index")
    ax.set_title("Velocity Distribution (BIN)")
    fig.colorbar(img, ax=ax, label="Velocity (km/s)")
    plt.tight_layout()
    
    ensure_directory_exists(Path(output_png).parent)
    plt.savefig(output_png, dpi=300)
    plt.close()
    print(f"Saved {output_png}")


def create_velocity_comparison_plots(config: DictConfig) -> None:
    """Create comparison plots of velocity models from CSV and binary files."""
    print("Creating velocity comparison plots...")
    
    # Auto-detect dimensions from CSV to match writing script
    nx, ny, nz = auto_detect_dimensions_from_csv(config.csv_file)
    
    # Create output directory
    output_dir = Path(config.csv_file).parent / "visualization_output"
    ensure_directory_exists(output_dir)
    
    # Plot from CSV
    csv_output = output_dir / "vp_from_csv.pdf"
    plot_velocity_from_csv(config, str(csv_output))
    
    # Plot from binary (if file exists)
    if os.path.exists(config.bin_file):
        bin_output = output_dir / "vp_from_bin.pdf"
        plot_velocity_from_bin(config, str(bin_output), nx, ny, nz)
    else:
        print(f"Binary file {config.bin_file} not found, skipping binary visualization")


def read_binary_model_for_animation(filepath: str, dimensions: Tuple[int, int, int]) -> np.ndarray:
    """
    Read a 3D Fortran-ordered binary file and reshape it into a NumPy array for animation.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Binary file not found at: {filepath}. Please update the file path.")

    data = np.fromfile(filepath, dtype=np.float32)
    expected_size = dimensions[0] * dimensions[1] * dimensions[2]
    if data.size != expected_size:
        raise ValueError(
            f"The number of data points in the file ({data.size}) does not match "
            f"the expected grid size of {expected_size} ({dimensions[0]}x{dimensions[1]}x{dimensions[2]}). "
            "Please check the grid dimensions."
        )
    return data.reshape(dimensions, order='F')


def generate_rotation_frames(model: np.ndarray, title: str, output_dir: str, num_frames: int) -> None:
    """
    Create a series of 3D scatter plots from different viewing angles for animation.
    """
    print(f"Preparing to generate {num_frames} animation frames...")

    i, j, k = np.nonzero(model)
    velocities = model[i, j, k]

    if velocities.size == 0:
        print("Warning: The model contains all zeros. No frames will be generated.")
        return

    # Create the output directory if it doesn't exist
    ensure_directory_exists(output_dir)

    # Generate a plot for each angle
    for frame, angle in enumerate(np.linspace(0, 360, num_frames, endpoint=False)):
        print(f"  - Generating frame {frame + 1}/{num_frames} (Angle: {angle:.1f}°)")
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')

        scatter = ax.scatter(i, j, k, c=velocities, cmap='viridis', marker='s', s=10)
        
        ax.set_xlabel('X index (Longitude direction)')
        ax.set_ylabel('Y index (Latitude direction)')
        ax.set_zlabel('Z index (Depth, positive down)')
        ax.set_title(f"{title}\nRotation Angle: {angle:.1f}°")
        ax.invert_zaxis()

        # Set the camera's viewing angle for the rotation
        ax.view_init(elev=30, azim=angle)

        cbar = fig.colorbar(scatter, shrink=0.5, aspect=10)
        cbar.set_label('Velocity (km/s)')

        frame_path = os.path.join(output_dir, f'frame_{frame:04d}.png')
        plt.savefig(frame_path, dpi=150, bbox_inches='tight')
        plt.close(fig)  # Close the figure to free up memory


def create_gif_from_frames(frames_dir: str, output_filepath: str, duration: float) -> None:
    """
    Compile a directory of image frames into a GIF.
    """
    print(f"\nCreating GIF from frames in '{frames_dir}'...")
    
    # Get a sorted list of frame files
    frame_files = sorted([os.path.join(frames_dir, f) for f in os.listdir(frames_dir) if f.endswith('.png')])
    
    if not frame_files:
        print("No frames found to create a GIF.")
        return

    # Read images and create the GIF
    with imageio.get_writer(output_filepath, mode='I', duration=duration, loop=0) as writer:
        for filename in frame_files:
            image = imageio.imread(filename)
            writer.append_data(image)
    
    print(f"GIF saved successfully to {output_filepath}")


def create_velocity_animation(config: DictConfig) -> None:
    """
    Create animated GIF visualization of velocity model.
    """
    print("--- Creating Velocity Model Animation ---")
    
    # Auto-detect dimensions from associated CSV file if available
    csv_file = config.get('csv_file', None)
    if csv_file and os.path.exists(csv_file):
        nx, ny, nz = auto_detect_dimensions_from_csv(csv_file)
    else:
        # Default dimensions if CSV not available
        nx, ny, nz = 41, 37, 41
        print(f"Using default dimensions: NX={nx}, NY={ny}, NZ={nz}")
    
    # Temporary directory for frames
    temp_frames_dir = 'temp_animation_frames'
    
    # Clean up previous runs if necessary
    if os.path.exists(temp_frames_dir):
        shutil.rmtree(temp_frames_dir)

    try:
        dimensions = (nx, ny, nz)
        print(f"--- Reading 3D model from: {config.bin_file} ---")
        velocity_model = read_binary_model_for_animation(config.bin_file, dimensions)
        
        plot_title = f'3D Visualization for {os.path.basename(config.bin_file)}'
        
        # Step 1: Generate all the individual frame images
        generate_rotation_frames(velocity_model, plot_title, temp_frames_dir, config.rotation_frames)
        
        # Step 2: Compile the frames into a GIF
        ensure_directory_exists(Path(config.output_gif).parent)
        create_gif_from_frames(temp_frames_dir, config.output_gif, config.frame_duration)

    except (FileNotFoundError, ValueError) as e:
        print(f"\nERROR: {e}")
        raise
    finally:
        # Step 3: Clean up the temporary frames directory
        if os.path.exists(temp_frames_dir):
            print(f"Cleaning up temporary directory: {temp_frames_dir}")
            shutil.rmtree(temp_frames_dir)