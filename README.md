# AttFP-PyG

This is a minimal working version AttentiveFP

the code is under dev

One major difference to the original code is the inital atom / bond UpProject (encoder)

I suspect that the error is in the atom softmax method that need to be per atom and not per molecule:
something like this may work ?
avu = EdgePooling.compute_edge_score_softmax(evu, edge_index[0], x_i.size()) # code 11

it's based on this code:
https://www.programcreek.com/python/?code=GraphNAS%2FGraphNAS%2FGraphNAS-master%2Fgraphnas_variants%2Fmacro_graphnas%2Fpyg%2Fpyg_gnn_layer.py


I added 3 versions:
- the original (just pytorch fast but not sparse + mask) see https://github.com/OpenDrugAI/AttentiveFP
- the DGL version which seems not converging as well as the original code see https://github.com/awslabs/dgl-lifesci
- my PyG versions logic / logic2 which does not converge yet (probabibly due to the softmax operation) inspired by https://github.com/xiongzhp/AttentiveFP_geometric

I followed this algorithm pseudo-code from the original paper https://pubs.acs.org/doi/10.1021/acs.jmedchem.9b00959:

![image](https://user-images.githubusercontent.com/1186658/111895472-03575980-8a13-11eb-947f-bf7a6ddaad72.png)


you may need to run all those python packages:

RDKit 
tqdm
pandas
sklean
torch
torch_scatter
torch_geometric
