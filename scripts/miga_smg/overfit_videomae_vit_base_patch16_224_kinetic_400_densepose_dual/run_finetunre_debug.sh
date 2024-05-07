#!/bin/bash

# Function to run finetune_debug.sh script and pause on error
run_finetune_debug_script() {
    # Define the path to your finetune_debug.sh script
    ls
    script_path="scripts/miga_smg/videomae_vit_base_patch16_224_kinetic_400_densepose_dual/finetune_debug.sh"

    # Add divider before running the script
    echo "----------------------------------------"
    echo "Running finetune_debug.sh script..."
    echo "----------------------------------------"

    # Run the finetune_debug.sh script
    bash "$script_path"

    # Check the exit status of the script
    if [ $? -ne 0 ]; then
        echo "----------------------------------------"
        echo "Error occurred. Press any key to exit."
        read -n 1 -s -r -p "Press any key to continue..."
    fi
}

# Call the function to run the script
run_finetune_debug_script
