from models import CVAE_nf, CVAE_nnf
from general_arguments import genargs
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


model_nf = CVAE_nf(args=genargs).to(genargs.device)
model_nf.load_state_dict(torch.load("trained_models/CVAE_optimal_NF.pt", map_location=genargs.device))

model_nnf = CVAE_nnf(args=genargs).to(genargs.device)
model_nnf.load_state_dict(torch.load("trained_models/CVAE_optimal_NNF.pt", map_location=genargs.device))

model_mse = CVAE_nf(args=genargs).to(genargs.device)
model_mse.load_state_dict(torch.load("trained_models/CVAE_MSE.pt", map_location=genargs.device))


test_attr = pd.read_csv("/content/list_attr_celeba_test.csv")
test_attr[test_attr == -1] = 0
counts = (test_attr.iloc[:, 1:].sum(axis=0)/len(test_attr)).sort_values(ascending=False)
drop_cols = list(counts.index[30:])
test_attr = test_attr.drop(drop_cols, axis=1)
columns = test_attr.columns[1:]

index = np.random.randint(0, len(test_attr))
row = test_attr.iloc[index]
features = [f for f in row.index[1:] if row[f] == 1]
attribute = row.values[1:].astype(int)
attribute = torch.tensor(attribute).type(torch.int32).unsqueeze(0)
print(features)

#function to shift values between (0, 1) from (-1, 1)
def transform_to_plot(tensor):
    return (tensor+1)/2

attr = attribute.to(genargs.device)

z_nf = model_nf.get_sample_z(attr, 1)
z_mse = model_mse.get_sample_z(attr, 1)
z_nnf = torch.randn(model_nnf.z_dim).unsqueeze(0)

#reconstructions
rec_nf = model_nf.decode(z_nf, attr).cpu().detach().numpy().transpose(0,2,3,1)[0]
rec_mse = model_mse.decode(z_mse, attr).cpu().detach().numpy().transpose(0,2,3,1)[0]
rec_nnf = model_nnf.decode(z_nnf, attr).cpu().detach().numpy().transpose(0,2,3,1)[0]

rec_nf = transform_to_plot(rec_nf)
rec_mse = transform_to_plot(rec_mse)
rec_nnf = transform_to_plot(rec_nnf)

fig, axes = plt.subplots(1, 3, figsize=(12, 4))
axes[0].imshow(rec_mse)
axes[0].axis('off')  # Hide axes
axes[0].set_title('MSE VAE')

axes[1].imshow(rec_nnf)
axes[1].axis('off')  # Hide axes
axes[1].set_title('NNF VAE')

axes[2].imshow(rec_nf)
axes[2].axis('off')  # Hide axes
axes[2].set_title('NF VAE')
plt.tight_layout()
plt.show()