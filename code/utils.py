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