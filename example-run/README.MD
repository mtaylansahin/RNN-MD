## How can output be generated from given input?

This directory includes example [input](/test-run/) folder to run the RNN-MD script and its [output](/test-run_results_l0m50/). Make sure that under the input folder you provide, there is a folder named `replica<replica_number>` and under it, a folder named `rep<replica_number>-interfacea`, with naming done in this manner. In MD simulation, the interaction details of every PDB generated should be kept in a separate file under `rep<replica_number>-interfacea`. You can find necessary information about the files in the output folder on the [README](../README.md) page located in the main directory.   

## Example run

To run RNN-MD with the **test-set** input using default parameters, follow the code below. **Don't forget to activate the RNN-MD environment before submit a run.**

```
python RNN-MD.py --data_dir test-run --replica 1 --chain1 A --chain2 C --train_ratio 0.8 --valid_ratio 0.1
```