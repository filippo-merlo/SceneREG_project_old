import os 

def get_files(directory):
    """
    Get all files in a directory with specified extensions.

    Args:
    - directory (str): The directory path.
    - extensions (list): A list of extensions to filter files by.

    Returns:
    - files (list): A list of file paths.
    """
    files = []
    for file in os.listdir(directory):
        if file.endswith(tuple([".json",".jpg"])):
            files.append(os.path.join(directory, file))
    return files

def visualize_dict_structure(dictionary, indent=0):
    for key, value in dictionary.items():
        if isinstance(value, dict):
            print("  " * indent + str(key) + ": {")
            visualize_dict_structure(value, indent + 1)
            print("  " * indent + "}")
        else:
            print("  " * indent + str(key) + ": " + str(value))