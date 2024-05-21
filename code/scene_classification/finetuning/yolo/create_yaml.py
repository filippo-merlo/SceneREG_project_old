import os
import yaml
import sys

def generate_yaml_for_dataset(dataset_directory):
    # Find the first folder in the dataset directory
    first_folder = dataset_directory

    if first_folder is None:
        raise Exception("No subdirectories found in the dataset directory")

    # Define the YAML configuration based on the dataset structure
    yaml_content = {
        'dataset_name': os.path.basename(dataset_directory),
        'description': 'Automatically generated dataset information.',
        'version': '1.0',
        'created_by': 'Your Name',
        'data_format': 'unknown',  # Default value, can be updated based on actual files
        'files': [],
        'subdirectories': []
    }

    # Analyze the first folder contents
    for root, dirs, files in os.walk(first_folder):
        for file in files:
            file_path = os.path.relpath(os.path.join(root, file), dataset_directory)
            yaml_content['files'].append(file_path)
        for dir in dirs:
            dir_path = os.path.relpath(os.path.join(root, dir), dataset_directory)
            yaml_content['subdirectories'].append(dir_path)

        # Only process the top-level directory
        break

    # Update data_format if possible
    if yaml_content['files']:
        first_file = yaml_content['files'][0]
        file_extension = os.path.splitext(first_file)[1][1:]  # Get the file extension without the dot
        yaml_content['data_format'] = file_extension

    # Define the path to the YAML file
    yaml_file_path = os.path.join(first_folder, 'dataset_info.yaml')

    # Save the YAML configuration to a file
    with open(yaml_file_path, 'w') as yaml_file:
        yaml.dump(yaml_content, yaml_file, default_flow_style=False)

    print(f"YAML file created at: {yaml_file_path}")

if __name__ == "__main__":
    
    generate_yaml_for_dataset('/mnt/cimec-storage6/users/filippo.merlo/ade20k_adapted')