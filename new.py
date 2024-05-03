import os
import json

def update_structure_file(directory):
    """
    Update 'structure.json' files within directories ending with 'v2' by appending 'v2' to specific keys
    that match the directory name excluding the 'v2' suffix.
    """
    for root, dirs, files in os.walk(directory):
        # Check if the current directory ends with 'v2'
        if root.endswith("v2"):
            structure_path = os.path.join(root, "structure.json")
            if os.path.exists(structure_path):
                with open(structure_path, 'r', encoding='utf-8') as file:
                    data = json.load(file)
                
                folder_name = os.path.basename(root)[:-2]  # Get the directory name without the 'v2'
                updated_data = {}
                updated = False

                # Update the key in the JSON structure if it matches the folder name
                for key, value in data.items():
                    if key == folder_name:
                        new_key = f"{key}v2"
                        updated_data[new_key] = value
                        updated = True
                    else:
                        updated_data[key] = value

                if updated:
                    # Write the updated dictionary back to the structure.json file
                    with open(structure_path, 'w', encoding='utf-8') as file:
                        json.dump(updated_data, file, indent=4)
                    print(f"Updated {structure_path}")

def main():
    base_directory = "/home/rk010/DM/NewDataset/rumoureval-2019-training-data/twitter-english/sydneysiege"
    update_structure_file(base_directory)

if __name__ == "__main__":
    main()
