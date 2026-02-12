#!/usr/bin/env python3
"""
Example: Using the heightmap module to process LiDAR data.

This example demonstrates how to use the satellit_sam.image_processing module
to load LiDAR data, create height maps with different settings, and save them.
"""

from pathlib import Path

from satellit_sam.core import (
    HeightMap,
    LiDARData,
    create_heightmap_from_las,
)


def example_basic_usage():
    """Example 1: Basic usage with convenience function."""
    print("Example 1: Basic usage")
    print("-" * 50)

    las_file = Path("../data/Traunstein/2018/inventory_plot_normalized.las")

    # Simple one-line creation and saving with resolution
    heightmap = create_heightmap_from_las(las_file, resolution=1.0, method="max")

    print(f"Height map shape: {heightmap.shape}")
    print(f"Height range: {heightmap.z_range}")
    print(f"Resolution: {heightmap.resolution}m per pixel")

    # Save as image
    heightmap.save("../output/example_basic.png")
    print("Saved to: ../output/example_basic.png")

    # Create with fixed dimensions
    heightmap_fixed = create_heightmap_from_las(las_file, width=1024, height=512)
    print(f"\nFixed dimensions shape: {heightmap_fixed.shape}")
    heightmap_fixed.save("../output/example_basic_1024x512.png")
    print("Saved to: ../output/example_basic_1024x512.png\n")


def example_detailed_workflow():
    """Example 2: Detailed workflow with explicit steps."""
    print("Example 2: Detailed workflow")
    print("-" * 50)

    las_file = Path("../data/Traunstein/2018/inventory_plot_normalized.las")

    # Step 1: Load LiDAR data
    print("Loading LiDAR data...")
    lidar = LiDARData.from_las(las_file)
    print(f"  Total points: {lidar.num_points:,}")
    print(f"  X range: {lidar.x_range}")
    print(f"  Y range: {lidar.y_range}")
    print(f"  Z range: {lidar.z_range}")

    # Step 2: Create height map with custom settings
    print("\nCreating height map...")
    heightmap = HeightMap.from_lidar(lidar, resolution=0.5, method="max")
    print(f"  Grid dimensions: {heightmap.shape}")
    print(f"  Height range: {heightmap.z_range}")

    # Step 3: Save with custom normalization
    z_min, z_max = heightmap.z_range
    heightmap.save("../output/example_detailed.png", z_min=0, z_max=50)
    print("Saved to: ../output/example_detailed.png\n")


def example_multiple_methods():
    """Example 3: Compare different aggregation methods."""
    print("Example 3: Comparing aggregation methods")
    print("-" * 50)

    las_file = Path("../data/Traunstein/2018/inventory_plot_normalized.las")

    # Load data once
    lidar = LiDARData.from_las(las_file)

    methods = ["max", "mean", "median"]
    for method in methods:
        print(f"\nCreating height map with method: {method}")
        heightmap = HeightMap.from_lidar(lidar, resolution=1.0, method=method)

        output_path = f"../output/example_{method}.png"
        heightmap.save(output_path)
        print(f"  Saved to: {output_path}")
        print(f"  Height range: {heightmap.z_range}")


def example_convert_to_image():
    """Example 4: Convert to Image object for further processing."""
    print("Example 4: Converting to Image object")
    print("-" * 50)

    las_file = Path("../data/Traunstein/2018/inventory_plot_normalized.las")

    # Create height map
    heightmap = create_heightmap_from_las(las_file, resolution=1.0)

    # Convert to Image object (from the image_processing module)
    image = heightmap.to_image()

    print(f"Image size: {image.size}")
    print(f"Channels: {image.channels}")
    print(f"Data type: {image.data.dtype}")

    # Now you can use all Image methods
    image.save("../output/example_as_image.png")
    print("Saved using Image.save(): ../output/example_as_image.png\n")


def example_different_resolutions():
    """Example 5: Create height maps at different resolutions and dimensions."""
    print("Example 5: Different resolutions and dimensions")
    print("-" * 50)

    las_file = Path("../data/Traunstein/2018/inventory_plot_normalized.las")
    lidar = LiDARData.from_las(las_file)

    # Different resolutions
    print("\nUsing resolution parameter:")
    resolutions = [0.5, 1.0, 2.0]
    for res in resolutions:
        heightmap = HeightMap.from_lidar(lidar, resolution=res)
        output_path = f"../output/example_res_{res:.2f}m.png"
        heightmap.save(output_path)
        print(f"  Resolution {res}m: {heightmap.shape} -> {output_path}")

    # Fixed dimensions
    print("\nUsing fixed dimensions:")
    dimensions = [(512, 256), (1024, 512), (2048, 1024)]
    for width, height in dimensions:
        heightmap = HeightMap.from_lidar(lidar, width=width, height=height)
        output_path = f"../output/example_{width}x{height}.png"
        heightmap.save(output_path)
        print(f"  {width}x{height}: {heightmap.shape} -> {output_path}")

    # Width only (height auto-calculated)
    print("\nUsing width only (height auto-calculated):")
    heightmap = HeightMap.from_lidar(lidar, width=1024)
    output_path = "../output/example_width_1024.png"
    heightmap.save(output_path)
    print(f"  Width 1024: {heightmap.shape} -> {output_path}")


def main():
    """Run all examples."""
    # Create output directory
    Path("../output").mkdir(exist_ok=True)

    print("=" * 70)
    print("LiDAR Height Map Examples")
    print("=" * 70)
    print()

    try:
        # Run examples
        example_basic_usage()
        example_detailed_workflow()
        example_multiple_methods()
        example_convert_to_image()
        example_different_resolutions()

        print("=" * 70)
        print("All examples completed successfully!")
        print("=" * 70)

    except FileNotFoundError as e:
        print(f"\nError: {e}")
        print(
            "\nMake sure the LiDAR file exists at: data/Traunstein/2018/inventory_plot_normalized.las"
        )
    except Exception as e:
        print(f"\nUnexpected error: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
