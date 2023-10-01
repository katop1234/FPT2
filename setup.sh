#!/bin/bash
# Use this script to setup the project
# In your bash terminal, run:
# chmod +x setup.sh
# ./setup.sh

# Step 1: Create a new conda environment named FPT with Python 3.9
echo "Creating a new conda environment named FPT with Python 3.9..."
conda create -y -n FPT python=3.9

# Step 2: Activate the newly created environment
echo "Activating the FPT environment..."
source activate FPT

# If the above 'source activate' doesn't work for you, you might want to use the below line instead:
# conda activate FPT

# Step 3: Install dependencies from requirements.txt using pip
echo "Installing dependencies from requirements.txt..."
pip install -r requirements.txt

# Step 4: Run the get_data.py script using Python
echo "Running the get_data.py script..."
python3 get_data.py

# End the script
echo "Setup script has completed its tasks."
