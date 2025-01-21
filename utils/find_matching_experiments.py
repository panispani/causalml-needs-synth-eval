import os
import shutil

# Define file paths
input_file = "output_causal_nf_robustness_jobs.log"
source_folder = "output_causal_nf/robustness"
destination_folder = "output_causal_nf/seg_lin_fun"

try:
    # Read lines from the input file
    with open(input_file, "r") as f:
        lines = f.readlines()

    # Extract relevant strings from lines that start with "Experiment folder:"
    extracted_strings = [
        line.split("/")[-1].strip()
        for line in lines
        if line.startswith("Experiment folder:")
    ]

    # Get the list of folders in the output directory
    if os.path.exists(source_folder):
        existing_folders = set(os.listdir(source_folder))
    else:
        existing_folders = set()
        print(f"Warning: The output folder '{source_folder}' does not exist.")

    # Identify folders in the output folder that do not have a match
    unmatched_folders = existing_folders - set(extracted_strings)

    # Print results
    print("Extracted Strings:", extracted_strings)
    print("Unmatched Folders:", list(unmatched_folders))

    for folder in extracted_strings:
        source_path = os.path.join(source_folder, folder)
        destination_path = os.path.join(destination_folder, folder)
        if os.path.exists(source_path):
            shutil.move(source_path, destination_path)
            print(f"Moved '{source_path}' to '{destination_path}'")
        else:
            print(f"Warning: Folder '{source_path}' does not exist.")


except Exception as e:
    print(f"An error occurred: {e}")
