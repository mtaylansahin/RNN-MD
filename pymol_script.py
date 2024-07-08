from pymol import cmd
import sys

# Get the input PDB file name from the user
input_file = sys.argv[1]

# Set the output PDB file name
output_file = input_file.replace('.pdb', '_modified.pdb')

# Load the PDB file
cmd.load(input_file)

# Add 1000 to the residue numbers in chain B
cmd.alter('chain B', 'resi=str(int(resi)+1000)')

# Create a new object by merging chains A and B
cmd.create('AB', 'chain A or chain B')

# Show only the AB object, hide the others
cmd.hide('everything', 'not AB')
cmd.show('cartoon', 'AB')

# Save the changes
cmd.save(output_file, 'AB')

# Quit Pymol
cmd.quit()
