from argparse import ArgumentParser, ArgumentTypeError
from pathlib import Path

# Validates during input process
def verify_dir(dir_string):
    path = Path(dir_string)
    if path.exists() and path.is_dir():
        return path
    
    raise ArgumentTypeError(f"No existing directory found at {path}.")

# Make double sure that they actually want to ovewrite the data directory. Defaults to false
def confirm_overwrite(overwrite, output_dir):
    if not overwrite or not Path(output_dir).exists():
        return False
    print(f"WARNING: The directory {output_dir} currently exists.")
    confirmation = input("Are you sure you want to overwrite this directory? [y/n]")

    # If you put in a stupid input, it's gonna reject by default
    return (confirmation).lower() == "y" 

# This is for the recreation of stress testing from the report, otherwise if 
# you want to test a particular case call each test individually
def parse_terminal_inputs():
    parser = ArgumentParser(description="Parameters for the stress testing of 3D Gaussian Splatting")

    parser.add_argument(
        "--data_dir", "-d", 
        type=verify_dir, 
        default="./datasets/EndoNerf/pulling_soft_tissues",
        help="Absolute path to the input data directory"
    )

    parser.add_argument(
        "--output_dir", "-o",
        type=str,
        default = "results",
        help="Absolute path to the directory results are saved in"
    ) 

    parser.add_argument(
        "--overwrite", 
        action="store_true", 
        help="Overwrite the folder's original data"
    )

    return parser.parse_args()

# Run the actual experiment :)
if __name__ == "__main__":
    args = parse_terminal_inputs()
    print(f"Reading from data directory {args.data_dir}")
    print(f"Overwrite flag is {'enabled' if args.overwrite else 'disabled'}.")

    if args.overwrite:
        if not confirm_overwrite(args.overwrite, args.output_dir):
            print("Overwrite cancelled.")
            args.overwrite = False
        else:
            print(f"Overwrite confirmed for data directory {args.output_dir}.")

    from VisualStressTests.apply import run_stress_testing as run_vis
    from SegmStressTests.apply import run_stress_testing as run_segm
    from DepthStressTests.apply import run_stress_testing as run_depth

    run_vis(data_dir = args.data_dir, randomisation_lvl="normal")
    run_segm(data_dir = args.data_dir)
    run_depth(data_dir = args.data_dir)



