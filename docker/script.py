import sys
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
from torchdrug import data, layers
from torchdrug.layers import geometry

from models import LMGearNetModel
from tasks import NodePropertyPrediction

# Get the input and output filenames from the command line arguments
input_filename = sys.argv[1]
output_filename = sys.argv[2]

graph_construction_model = layers.GraphConstruction(
    node_layers=[geometry.AlphaCarbonNode()],
    edge_layers=[
        geometry.SpatialEdge(radius=10.0, min_distance=5),
        geometry.KNNEdge(k=10, min_distance=5),
        geometry.SequentialEdge(max_distance=2),
    ],
    edge_feature="gearnet"
)

task = NodePropertyPrediction(
    model=LMGearNetModel(),
    graph_construction_model=graph_construction_model,
    num_mlp_layer=2
)
task.simple_preprocess()

task.load_state_dict(torch.load("model.pt", map_location=torch.device('cpu')), strict=False)
task.eval()

dataset = data.ProteinDataset()
dataset.load_pdbs([input_filename])
dataset.targets = defaultdict(list)
dataset.targets['binding'] = [[0] * dataset[0]['graph'].num_residue]

print(dataset[0])

dataloader = data.DataLoader(dataset, batch_size=1)

with torch.no_grad():
    for batch in dataloader:
        pred = task.predict(batch)
        break

df = pd.DataFrame({
    'residue_index': np.arange(1, len(pred) + 1),
    'prediction': np.where(pred > -2, 1, 0).flatten()
})
df.to_csv(output_filename, index=False)
print(pred)
