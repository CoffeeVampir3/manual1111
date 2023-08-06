from huggingface_hub import snapshot_download, hf_hub_download
import os
import platform
import subprocess
import argparse

def create_symlink(source, destination):
    try:
        os.symlink(source, destination)
        print(f"Symbolic link created at {destination}")
    except Exception as e:
        print(f"Error creating symlink: {e}")

def search_files(extension, path="/"):
    try:
        result = subprocess.check_output(f"find {path} -name '*.{extension}'", shell=True).decode()
        files = result.split("\n")[:-1]
        return files
    except Exception as e:
        print(f"Error searching for files: {e}")
        return []

parser = argparse.ArgumentParser(description='Search for a file and create a symbolic link if it exists.')
parser.add_argument('-s', '--symdir', type=str, required=False, help='Directory to search for the file')
args = parser.parse_args()

try:
    if platform.system() == "Linux" and args.symdir:
        print("Linux detected, checking if it's possible to create a symlink instead of downloading a new file.")
        files = search_files("safetensors", path=args.symdir)

        if files:
            print("Found!")
            for file in files:
                filename = os.path.basename(file)
                destination = f"models/{filename}"
                create_symlink(file, destination)
            exit()
        else:
            print("File not found. Proceed with download.")
except Exception as e:
    print(e)
    print("Encountered an error. Proceeding to download.")

repo_id = "stabilityai/stable-diffusion-xl-base-1.0"
    
hf_hub_download(
    repo_id=repo_id, 
    filename="sd_xl_base_1.0_0.9vae.safetensors",
    local_dir="models",
    local_dir_use_symlinks=False)