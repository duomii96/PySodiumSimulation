import os
from pathlib import Path
from collections import defaultdict


def check_method_files_specific(root_folder):
    # Convert string path to a Path object
    root_path = Path(root_folder)

    if not root_path.exists():
        print(f"Error: The directory '{root_folder}' does not exist.")
        return

    print(f"Scanning subfolders in '{root_folder}' for 'method' files...\n")

    # Track our stats
    match_count = 0
    files_checked = 0

    # rglob searches recursively through all subfolders for files named exactly "method"
    for filepath in root_path.rglob("method"):
        if filepath.is_file():
            try:
                # Open with utf-8 and ignore decoding errors
                with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
                    # We successfully opened a method file, so increment our check counter
                    files_checked += 1

                    # Read through the file line by line
                    for line in f:
                        if line.startswith("##$Method="):
                            # Extract the value inside the angle brackets <>
                            start = line.find("<") + 1
                            end = line.find(">")

                            if start > 0 and end > 0 and start < end:
                                method_value = line[start:end]

                                # Check if it matches your target value
                                if method_value == "User:sr_SE":
                                    # Print the parent folder path
                                    print(filepath.parent)
                                    match_count += 1

                            # Break the inner loop since we found the method line for this file.
                            # No need to read the remaining lines (like ##$PVM_RepetitionTime)
                            break

            # Catch file read errors (e.g., PermissionError)
            except Exception as e:
                print(f"Skipping {filepath} due to error: {e}")

    # Print the final summary
    print(f"\n--- Scan Complete ---")
    print(f"Total 'method' files checked: {files_checked}")
    print(f"Total matching files found:   {match_count}")


def format_ranges(folder_names):
    """
    Helper function to convert a list of folder names into a grouped string.
    E.g., ['1', '2', '3', '5', '8', '9'] -> '1-3, 5, 8-9'
    Handles non-integer folder names gracefully by just appending them.
    """
    ints = []
    strs = []

    # Separate purely numeric folders from any string-based folders
    for name in folder_names:
        if name.isdigit():
            ints.append(int(name))
        else:
            strs.append(name)

    ints.sort()
    strs.sort()

    ranges = []
    if ints:
        start = ints[0]
        prev = ints[0]

        for n in ints[1:]:
            if n == prev + 1:
                prev = n
            else:
                # Close the current range and start a new one
                ranges.append(f"{start}-{prev}" if start != prev else str(start))
                start = n
                prev = n
        # Append the final range
        ranges.append(f"{start}-{prev}" if start != prev else str(start))

    # Combine the formatted number ranges with any text-based folder names
    ranges.extend(strs)
    return ", ".join(ranges)


def check_method_files(root_folder):
    # Convert string path to a Path object
    root_path = Path(root_folder)

    if not root_path.exists():
        print(f"Error: The directory '{root_folder}' does not exist.")
        return

    print(f"Scanning subfolders in '{root_folder}' for 'method' files...\n")

    files_checked = 0
    # Dictionary to store { "MethodName": ["1", "2", "3", ...] }
    method_dict = defaultdict(list)

    # rglob searches recursively through all subfolders for files named exactly "method"
    for filepath in root_path.rglob("method"):
        if filepath.is_file():
            try:
                # Open with utf-8 and ignore decoding errors
                with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
                    files_checked += 1

                    for line in f:
                        if line.startswith("##$Method="):
                            start = line.find("<") + 1
                            end = line.find(">")

                            if start > 0 and end > 0 and start < end:
                                method_value = line[start:end]
                                folder_number = filepath.parent.name

                                # Print to terminal as before
                                print(f"Folder: {folder_number:<5} | Method: {method_value}")

                                # Add the folder to our dictionary for the text file later
                                method_dict[method_value].append(folder_number)

                            break

            except Exception as e:
                print(f"Skipping {filepath} due to error: {e}")

    # Generate the content.txt file
    output_file = root_path / "content.txt"
    try:
        with open(output_file, "w", encoding="utf-8") as out:
            # Print the root folder name at the top
            out.write(f"Folder Name: {root_path.name}\n")
            out.write("=" * 60 + "\n")

            # Print the table header
            out.write(f"{'Method Name':<25} | {'Folders'}\n")
            out.write("-" * 25 + "-+-" + "-" * 32 + "\n")

            # Print each method and its folder ranges
            for method, folders in method_dict.items():
                folder_str = format_ranges(folders)
                out.write(f"{method:<25} | {folder_str}\n")

        print(f"\n--- Scan Complete ---")
        print(f"Total 'method' files checked: {files_checked}")
        print(f"Success: Created summary file at '{output_file}'")

    except Exception as e:
        print(f"\n--- Scan Complete (with errors) ---")
        print(f"Error writing to '{output_file}': {e}")


# Example usage:
check_method_files(r"J:\AG_XNMR\14_Reichert\Messdaten\RawData\NEO\ZI\SR_SR_SEvTQTPPI_BrukerLin_1_9_20230623_165629")

#check_method_files_specific(r"J:\AG_XNMR\14_Reichert\Messdaten\RawData\NEO\ZI\SR_SR_SEvTQTPPI_BrukerLin_1_9_20230623_165629")