## Converting an Ensemble PDB File from Trajectory and Topology Data and Generating Interface Outputs

We used molecular dynamics simulation data from the crystal structure of tissue factor (TF) conjugated to humanized Fab D3h4 (Fab), identified with PDB ID 1JPS.This assembly's dynamic 100 ns trajectory is derived from Jandova et al. [https://doi.org/10.1021/acs.jctc.1c00336](https://doi.org/10.1021/acs.jctc.1c00336). Using the CHARMM36m force field and the GROMACS simulation program version 2019, the authors ran simulations on 1JPS. We took samples of this trajectory at 500 ps intervals, which gave us 200 time-dependent coordinates for 1JPS.

Three chains were present in the 1JPS structure: tissue factor (TF), two antibody chains (heavy and light). We combined the heavy and light chains of the MD ensemble (100 ns, 200 frames) into a single chain in order to examine Fab as a monomeric entity. We added 1000 to the original numbers of the heavy chain residues of Fab in order to prevent overlapping residue numbers. Then, We used the [interfacea](https://github.com/JoaoRodrigues/interfacea/tree/master) Python module to calculate the non-covalent interaction characteristics for each snapshot. This library computes the hydrogen bonds, salt bridges, and inter-chain hydrophobic interactions given a complicated PDB coordinate as input. For Quick generating the interaction data, you must run `interaction_analysis.ipynb` notebook.

Once you have obtained the trajectory and topology files, transfer them to the HPC and execute the following commands to generate a PDB file. If you are working locally, skip *module load* section and ensure that GROMACS and CUDA 9.2 are installed:

```
module load gromacs/2020.4
module load cuda92 openblas hwloc openmpi

gmx trjconv -f traj.trr -s topo.tpr -o complex_md-whole.xtc -pbc whole # Select protein

gmx trjconv -f complex_md-whole.xtc -s topo.tpr -o complex_md-nojump.xtc -pbc nojump # Select protein

gmx trjconv -f complex_md-nojump.xtc -s topo.tpr -pbc mol -center -ur compact -o complex_md-nojump-cent.xtc # Select protein

gmx trjconv -f complex_md-nojump-cent.xtc -s topo.tpr -o complex_md-nojump-cent-cluster.xtc -pbc cluster # Select protein

gmx trjconv -f complex_md-nojump-cent-cluster.xtc -s topo.tpr -fit rot+trans -o complex_md-fit.xtc # Select backbone and then protein

gmx trjconv -f complex_md-fit.xtc -s topo.tpr -dt 100 -o complex_md-fit.pdb # Select protein
```
**Now, add "1000" to heavy chain (chain B) and merge it with light chain.** 

Then, atomic interactions (e.g. ionic, hydrogen bonds, and hydrophobics) are identified with `interfacea`. For this, follow the installation instructions of interfacea and then run the `server_contact_interfacea.sh` script under **interfacea folder**.

```
chmod +x server_contact_interfacea.sh
./server_contact_interfacea.sh
```

As a final step you can run the cells inside the `interaction_analysis.ipynb` notebook to create the merged atomic interactions files and plots.
