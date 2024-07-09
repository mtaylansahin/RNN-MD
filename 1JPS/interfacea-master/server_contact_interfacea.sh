#!/bin/bash

# Define the path of your complex.pdb file
read -p "Enter the path of your complex: " COMPLEX_NAME

# Define the directory where the output PDB files will be saved
output_dir="PDBs"
mkdir -p "$output_dir"
mkdir -p ../replica1/rep1-interfacea

# Initialize variables
current_model=""
model_data=""

# Read the complex.pdb file
while IFS= read -r line; do
    if [[ $line == MODEL* ]]; then
        # Save the current model
        if [[ -n $current_model ]]; then
            echo "$model_data" > "${output_dir}/model_${current_model}.pdb"
            model_data=""
            python contacts_type.py "${output_dir}/model_${current_model}.pdb" > ../replica/rep1-interfacea/"$i.interfacea"
        fi
        # Update the current model number
        current_model=$(echo $line | awk '{print $2}')
    fi
    # Collect data for the current model
    model_data+="$line"$'\n'
done < "$COMPLEX_NAME"

# Save the last model
if [[ -n $current_model ]]; then
    echo "$model_data" > "${output_dir}/model_${current_model}.pdb"
fi

echo "Models were successfully separated and saved."

echo "Processing completed."
