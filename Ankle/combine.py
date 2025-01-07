import os

# Path to the folder containing the files
folder_path = 'B:\Dataset\Ankle\Lateral Fall'  # Replace with your actual path

# Create a mapping of participant names to p1, p2, etc.
participants = ['anton', 'arek', 'dan', 'harish', 'kannan', 'pondy', 'shrip', 'sudharshan', 'suhaas', 'swami']
participant_map = {name.lower(): f"p{i+1}" for i, name in enumerate(participants)}

# List all files in the folder
files = os.listdir(folder_path)

# Rename the files
for file_name in files:
    for original_name, new_name in participant_map.items():
        if original_name in file_name.lower():
            # Generate the new file name
            new_file_name = file_name.lower().replace(original_name, new_name)
            
            # Get full paths for renaming
            old_file_path = os.path.join(folder_path, file_name)
            new_file_path = os.path.join(folder_path, new_file_name)
            
            # Rename the file
            os.rename(old_file_path, new_file_path)
            print(f"Renamed: {file_name} -> {new_file_name}")
