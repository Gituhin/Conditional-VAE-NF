#Contains code for training the models.
from models import CVAE_nf, CVAE_nnf
import torch
from general_arguments import genargs
from torch import nn
import torch.nn.functional as F
from torchvision import transforms, datasets, utils, models

import numpy as np
from tqdm.notebook import tqdm
from data import train_loader, test_loader
import matplotlib.pyplot as plt



#Early stopper and saver of the best model
class saveBestModel():
    def __init__(self, tolerance):
        self.best_valid_loss = float('inf')
        self.best_state_dict = None
        self.best_epoch = None
        self.tolerance_to = tolerance
        self.tolerance = 0


    def early_stop(self, current_valid_loss, epoch, model):
        if current_valid_loss < self.best_valid_loss:
            self.best_valid_loss = current_valid_loss
            self.best_state_dict = model.state_dict()
            self.best_epoch = epoch
            self.tolerance = 0
        else:
            self.tolerance += 1
            print('Current test loss is greater than min loss\n')
            if self.tolerance >= self.tolerance_to:
                print(f'Early stopping triggered at epoch {epoch}')
                return True
        return False
    

def train_CVAE_nf(epoch):
    print(f'Doing Epoch: {epoch}')
    model.train()
    train_loss = 0
    test_loss = 0
    for batch_idx, (data, attr) in enumerate(tqdm(train_loader)):
        inp = data.to(genargs.device)
        #attr = to_one_hot(attr)
        #print(attr.shape, data.shape)
        optimizer.zero_grad()

        # Run VAE
        recon, fz, log_det_jac, dist_params = model(inp, attr.to(genargs.device))
        # Compute loss
        rec, kl = model.total_loss_function(recon, inp, dist_params['mu_p'],
                                            dist_params['logsigma_p'], dist_params['logsigma_q'], fz, log_det_jac)

        total_loss = rec + kl
        total_loss.backward()
        train_loss += total_loss.item()
        optimizer.step()

        if batch_idx % genargs.log_interval == 0:
            print(f'For Train Epoch: {epoch}, {batch_idx * len(data)}/{len(train_loader.dataset)}, {100. * batch_idx / len(train_loader)}% Over')
            print(f"NLL: {rec.item() / len(data)}, KL: {kl.item() / len(data)}, log_sigma: {model.log_sigma}")
            # Plot reconstructions
            n = min(inp.size(0), 8)
            #print(recon.shape)
            model.eval()
            inp_scaled = (inp+1)/2
            recon_scaled = (recon+1)/2
            comparison = torch.cat([inp_scaled[:n], recon_scaled.view(genargs.batch_size, -1, genargs.img_dim, genargs.img_dim)[:n]])
            comparison = utils.make_grid(comparison)
            print("Trainset Reconstructions: ")
            plt.imshow(comparison.detach().cpu().numpy().transpose(1,2,0))
            plt.show()

            train_loss /=  len(train_loader.dataset)
            print('Average train loss: {:.4f}'.format(train_loss))

            with torch.no_grad():
                model.eval()
                test_recon_loss = 0
                test_kl = 0
                for data_test, attr_test in tqdm(test_loader):
                    inp_test = data_test.to(genargs.device)
                    attr_test = attr_test.to(genargs.device)
                    recon, fz, log_det_jac, dist_params = model(inp_test, attr_test)
                    rec, kl = model.total_loss_function(recon, inp_test, dist_params['mu_p'],
                                            dist_params['logsigma_p'], dist_params['logsigma_q'], fz, log_det_jac)

                    test_recon_loss += rec.item()
                    test_kl += kl.item()
                    total_test_loss = test_recon_loss + test_kl
                    test_loss += total_test_loss
                test_loss /= len(test_loader.dataset)
                test_recon_loss /= len(test_loader.dataset)
                test_kl /= len(test_loader.dataset)
                print('Test Total loss: {:.4f}, Recon : {:.4f}, KL: {:.4f}\n'.format(test_loss, test_recon_loss, test_kl))
                if best_model_saver.early_stop(test_recon_loss, epoch, model):
                    break


def train_CVAE_nnf(epoch):
    print(f'Doing Epoch: {epoch}')
    model.train()
    train_loss = 0
    test_loss = 0
    for batch_idx, (data, attr) in enumerate(tqdm(train_loader)):
        inp = data.to(genargs.device)
        #attr = to_one_hot(attr)
        #print(attr.shape, data.shape)
        optimizer.zero_grad()

        # Run VAE
        recon, dist_params = model(inp, attr.to(genargs.device))
        # Compute loss
        rec, kl = model.total_loss_function(recon, inp, dist_params['mu_q'], dist_params['logsigma_q'])

        total_loss = rec + kl
        total_loss.backward()
        train_loss += total_loss.item()
        optimizer.step()

        if batch_idx % genargs.log_interval == 0:
            print(f'For Train Epoch: {epoch}, {batch_idx * len(data)}/{len(train_loader.dataset)}, {100. * batch_idx / len(train_loader)}% Over')
            print(f"NLL: {rec.item() / len(data)}, KL: {kl.item() / len(data)}, log_sigma: {model.log_sigma}")
            # Plot reconstructions
            n = min(inp.size(0), 8)
            #print(recon.shape)
            model.eval()
            inp_scaled = (inp+1)/2
            recon_scaled = (recon+1)/2
            comparison = torch.cat([inp_scaled[:n], recon_scaled.view(genargs.batch_size, -1, genargs.img_dim, genargs.img_dim)[:n]])
            comparison = utils.make_grid(comparison)
            print("Trainset Reconstructions: ")
            plt.imshow(comparison.detach().cpu().numpy().transpose(1,2,0))
            plt.show()

            train_loss /=  len(train_loader.dataset)
            print('Average train loss: {:.4f}'.format(train_loss))

            with torch.no_grad():
                model.eval()
                test_recon_loss = 0
                test_kl = 0
                for data_test, attr_test in tqdm(test_loader):
                    inp_test = data_test.to(genargs.device)
                    attr_test = attr_test.to(genargs.device)
                    recon, dist_params = model(inp_test, attr_test)
                    rec, kl = model.total_loss_function(recon, inp_test, dist_params['mu_q'], dist_params['logsigma_q'])

                    test_recon_loss += rec.item()
                    test_kl += kl.item()
                    total_test_loss = test_recon_loss + test_kl
                    test_loss += total_test_loss
                test_loss /= len(test_loader.dataset)
                test_recon_loss /= len(test_loader.dataset)
                test_kl /= len(test_loader.dataset)
                print('Test Total loss: {:.4f}, Recon : {:.4f}, KL: {:.4f}\n'.format(test_loss, test_recon_loss, test_kl))
                if best_model_saver.early_stop(test_recon_loss, epoch, model):
                    break


model = CVAE_nf(args=genargs).to(genargs.device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.0007)
best_model_saver = saveBestModel(tolerance = 2)


