## Content of the input folder

Make sure to include a get_history_graph.py script in this folder. Ensure that within the input folder you provide, there is a subfolder named `replica<replica_number>`, and within that, another subfolder named `rep<replica_number>-interfacea`. The naming should follow this format. In the MD simulation, the interaction details of each generated PDB should be stored in a separate file within `rep<replica_number>-interfacea`. 

## Example run

To run RNN-MD with the **test-set** input using default parameters, follow the code below. **Don't forget to activate the RNN-MD environment before submit a run.**

```
python RNN-MD.py --data_dir test-run --replica 1 --chain1 A --chain2 C --train_ratio 0.8 --valid_ratio 0.1
```