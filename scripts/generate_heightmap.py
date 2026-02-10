#!/usr/bin/env python3
"""
Generate a grayscale height map from LiDAR point cloud data (.las file).

This script uses the satellit_sam.image_processing module to create height maps
from LiDAR data and save them as grayscale images.
"""

import argparse
import sys
from pathlib import Path

# Add the src directory to the path so we can import satellit_sam
sys.path.insert(0, str(Path(__file__).parent.parent / "with_sam" / "src"))

from satellit_sam.image_processing import (
    HeightMap,
    LiDARData,
    create_heightmap_from_las,
)


def main():
    parser = argparse.ArgumentParser(
        description="Generate grayscale height map from LiDAR point cloud data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate height map with default settings (0.5m resolution, max method)
  %(prog)s data/input.las

  # Generate with 1 meter resolution
  %(prog)s data/input.las -r 1.0

  # Generate with fixed dimensions (1024x768 pixels)
  %(prog)s data/input.las -W 1024 -H 768

  # Use mean height instead of max
  %(prog)s data/input.las -m mean

  # Specify output path
  %(prog)s data/input.las -o output/heightmap.png

  # Combine options
  %(prog)s data/input.las -W 2048 -H 1024 -m median -o output/heightmap.png
        """,
    )
    parser.add_argument("input", type=str, help="Input .las file path")
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default=None,
        help="Output image path (default: <input>_heightmap.png)",
    )
    parser.add_argument(
        "-r",
        "--resolution",
        type=float,
        default=None,
        help="Grid resolution in meters (default: 0.5). Cannot be used with -W/-H.",
    )
    parser.add_argument(
        "-W",
        "--width",
        type=int,
        default=None,
        help="Output image width in pixels. Cannot be used with -r.",
    )
    parser.add_argument(
        "-H",
        "--height",
        type=int,
        default=None,
        help="Output image height in pixels. Cannot be used with -r.",
    )
    parser.add_argument(
        "-m",
        "--method",
        type=str,
        choices=["max", "mean", "median"],
        default="max",
        help="Method to aggregate heights in each cell (default: max)",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Print detailed information",
    )

    args = parser.parse_args()

    # Validate input file
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: Input file not found: {args.input}", file=sys.stderr)
        sys.exit(1)

    # Set default output path
    if args.output is None:
        output_path = input_path.parent / f"{input_path.stem}_heightmap.png"
    else:
        output_path = Path(args.output)

    # Create output directory if needed
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Validate parameter combinations
    if args.resolution is not None and (
        args.width is not None or args.height is not None
    ):
        print(
            "Error: Cannot specify both --resolution and --width/--height",
            file=sys.stderr,
        )
        sys.exit(1)

    try:
        # Load LiDAR data
        if args.verbose:
            print(f"Reading LiDAR data from: {input_path}")

        lidar = LiDARData.from_las(input_path)

        if args.verbose:
            print(f"Total points: {lidar.num_points:,}")
            x_min, x_max = lidar.x_range
            y_min, y_max = lidar.y_range
            z_min, z_max = lidar.z_range
            print(f"X range: [{x_min:.2f}, {x_max:.2f}]")
            print(f"Y range: [{y_min:.2f}, {y_max:.2f}]")
            print(f"Z range: [{z_min:.2f}, {z_max:.2f}]")

            if args.resolution is not None:
                print(
                    f"\nCreating height map with resolution: {args.resolution}m, method: {args.method}"
                )
            elif args.width is not None or args.height is not None:
                dims = []
                if args.width:
                    dims.append(f"width={args.width}")
                if args.height:
                    dims.append(f"height={args.height}")
                print(
                    f"\nCreating height map with {', '.join(dims)}, method: {args.method}"
                )
            else:
                print(
                    f"\nCreating height map with resolution: 0.5m (default), method: {args.method}"
                )

        # Create height map
        heightmap = HeightMap.from_lidar(
            lidar,
            resolution=args.resolution,
            method=args.method,
            width=args.width,
            height=args.height,
        )

        if args.verbose:
            height, width = heightmap.shape
            z_min, z_max = heightmap.z_range
            print(f"Grid size: {width} x {height}")
            print(f"Height map created. Min: {z_min:.2f}, Max: {z_max:.2f}")
            print(f"\nSaving height map to: {output_path}")

        # Save as image
        heightmap.save(output_path)

        if args.verbose:
            print(f"Done! Height map dimensions: {heightmap.shape}")
        else:
            print(f"Height map saved to: {output_path}")

    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Unexpected error: {e}", file=sys.stderr)
        if args.verbose:
            import traceback

            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
