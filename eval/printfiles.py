
import os

def list_files(directory):
    """Prints all files in the given directory"""
    try:
        for filename in os.listdir(directory):
            file_path = os.path.join(directory, filename)
            if os.path.isfile(file_path):
                print(filename)
    except FileNotFoundError:
        print("Error: The directory does not exist.")
    except PermissionError:
        print("Error: Permission denied.")

# Example usage
path = "D:\semester_3\AML\project\AnomalySegmentation_CourseProjectBaseCode\\train"  # Replace with the actual directory path
list_files(path)