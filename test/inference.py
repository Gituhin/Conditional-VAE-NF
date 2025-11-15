from modules.models import CVAE_nf, CVAE_nnf
from modules.general_arguments import genargs
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns



def load_models():
    model_nf = CVAE_nf(args=genargs).to(genargs.device)
    model_nf.load_state_dict(torch.load("trained_models/CVAE_optimal_NF.pt", map_location=genargs.device))

    model_nnf = CVAE_nnf(args=genargs).to(genargs.device)
    model_nnf.load_state_dict(torch.load("trained_models/CVAE_optimal_NNF.pt", map_location=genargs.device))

    model_mse = CVAE_nf(args=genargs).to(genargs.device)
    model_mse.load_state_dict(torch.load("trained_models/CVAE_MSE.pt", map_location=genargs.device))
    return [model_mse, model_nnf, model_nf]


def load_attributes_data(file_path):
    test_attr = pd.read_csv(file_path)
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
    return attribute

#function to shift values between (0, 1) from (-1, 1)
def transform_to_plot(tensor):
    return (tensor+1)/2


def generate_samples(models, attribute_data):
    attr = attribute_data.to(genargs.device)
    n = len(models)
    reconstructed_images = []
    for idx, model in enumerate(models):
        if idx != 1:
            z = model.get_sample_z(attr, 1)
        else:
            z = torch.randn(model.z_dim).unsqueeze(0)
        recon = model.decode(z, attr).cpu().detach().numpy().transpose(0,2,3,1)[0]
        reconstructed_images.append(transform_to_plot(recon))
    return reconstructed_images



if __name__ == '__main__':
    models = load_models()
    attributes = load_attributes_data(file_path="list_attr_celeba_test.csv")
    recons = generate_samples(models, attributes)

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(recons[0])
    axes[0].axis('off')  # Hide axes
    axes[0].set_title('MSE VAE')

    axes[1].imshow(recons[1])
    axes[1].axis('off')  # Hide axes
    axes[1].set_title('NNF VAE')

    axes[2].imshow(recons[2])
    axes[2].axis('off')  # Hide axes
    axes[2].set_title('NF VAE')
    plt.tight_layout()
    plt.show()