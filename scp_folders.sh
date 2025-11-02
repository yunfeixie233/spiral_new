#!/bin/bash

# Configuration
HOST="38.80.122.117"
USER="ubuntu"
KEY_FILE="./game_ssh_key"
SSH_OPTS="-i $KEY_FILE -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null"

# List of folders to copy
FOLDERS=(
    "/ephemeral/games-workspace/spiral_new/oat-output/run_4b_4env_noresume_randenv_1101T1544/saved_models"
    "/ephemeral/games-workspace/spiral_new/oat-output/run_pig_noeval_1025T1515/saved_models"
    "/ephemeral/games-workspace/spiral_new/oat-output/run_tic_noeval_1024T0722/saved_models"

    # Add more folders here, one per line
    # "/ephemeral/games-workspace/spiral_new/oat-output/another_folder"
    # "/ephemeral/games-workspace/spiral_new/script"
)

# Set correct permissions for the key file
chmod 600 "$KEY_FILE"

echo "Copying to $USER@$HOST with preserved folder structure"
echo "----------------------------------------"

# Copy each folder/file in the list
for item in "${FOLDERS[@]}"; do
    if [ -e "$item" ]; then
        # Get absolute path
        abs_path=$(cd "$(dirname "$item")" && pwd)/$(basename "$item")
        
        # Get parent directory
        parent_dir=$(dirname "$abs_path")
        
        echo "Copying $abs_path..."
        echo "Creating remote directory: $parent_dir"
        
        # Create parent directory on remote server
        ssh $SSH_OPTS "$USER@$HOST" "mkdir -p $parent_dir"
        
        # Copy the folder/file to the same path on remote
        scp -r $SSH_OPTS "$abs_path" "$USER@$HOST:$abs_path"
        
        if [ $? -eq 0 ]; then
            echo "Successfully copied to $abs_path on remote server"
        else
            echo "Failed to copy $item"
        fi
        echo "----------------------------------------"
    else
        echo "Warning: $item does not exist"
        echo "----------------------------------------"
    fi
done

echo "Transfer complete!"

