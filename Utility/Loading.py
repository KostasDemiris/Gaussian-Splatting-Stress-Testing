import os
import yaml

# This allows each stress test file to treat this as though there's a global 'config' for load_transform
_internal_config = {}

# The settings for these are controlled via a configuration file stored at filepath, 
# RELATIVE TO THE "StressTesting" DIRECTORY!!!!
def load_stress_test_config(config_directory=None, config_filename=None):
    global _internal_config
    if config_directory is None:
        config_directory = os.path.dirname(os.path.abspath(__file__))
    
    config_directory = os.path.dirname(os.path.abspath(__file__)) if config_directory is None else config_directory
    config_filename  = "config.yaml" if config_filename is None else config_filename
    config_file_path = os.path.join(config_directory, config_filename)

    try:
        with open(config_file_path, 'r') as file:
            _internal_config = yaml.safe_load(file)
            return _internal_config
        
    except FileNotFoundError:
        print(f"Error: {config_file_path} not found.")
        return {}
    
    except yaml.YAMLError as exc:
        print(f"Error parsing YAML: {exc}")
        return {}

# Since the morphological operations all use very similar sets of parameters, it makes more sense to
# just fix them in the loading function, with the exception of the larger random operation function.
def load_morph_operation(transform_name, manual_params):
    if manual_params:
        return manual_params.get("kernel_size", 5), manual_params.get("iters_count", 1)

    transform_config = _internal_config.get(transform_name, {})
    kernel_size = params.get("kernel_size", 5)
    iterations  = params.get("iters_count", 1)
        
    return kernel_size, iterations

def format_morph_params(kernel_size=None, iterations=None):
    if kernel_size is None or iterations is None:
        return None

    params = {
        "kernel_size": kernel_size,
        "iters_count": iterations,
    }

    return params

def load_transform(transform_name, likelihood, params):
    if likelihood is None:
        likelihood = _internal_config.get(transform_name, {}).get("likelihood", 0.0)
    if params is None:
        params = _internal_config.get(transform_name, {}).get("params", {})

    return likelihood, params

