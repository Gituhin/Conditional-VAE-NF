import torch
from torch import nn
import torch.nn.functional as F
import numpy as np


def softclip(tensor, min):
    """ Clips the tensor values at the minimum value min in a softway. Taken from Handful of Trials """
    result_tensor = min + nn.functional.softplus(tensor - min)

    return result_tensor

class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)


class UnFlatten(nn.Module):
    def __init__(self, n_channels):
        super(UnFlatten, self).__init__()
        self.n_channels = n_channels

    def forward(self, input):
        size = int((input.size(1) // self.n_channels) ** 0.5)
        return input.view(input.size(0), self.n_channels, size, size)


# Normalizing Flows (NVP) f(z)

class simpleAffineFlow(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.a = nn.Parameter(torch.zeros(self.dim)).to(genargs.device)  # log_scale
        self.b = nn.Parameter(torch.zeros(self.dim)).to(genargs.device)  # shift

    def forward(self, z):
        fz = torch.exp(self.a) * z + self.b # Simple affine transformation

        det_jacobian = torch.exp(self.a.sum())
        log_det_jacobian = torch.ones(z.shape[0]).to(genargs.device) * torch.log(det_jacobian)
        #log_det_jacobian = torch.log(det_jacobian)

        return fz, log_det_jacobian

    def inverse(self, y):
        x = (y - self.b) / torch.exp(self.a)

        det_jacobian = 1 / torch.exp(self.a.sum())
        inv_log_det_jac = torch.ones(y.shape[0]) * torch.log(det_jacobian)

        return x, inv_log_det_jac


# NVP - Non-Preserving volume taken from  arXiv:1605.08803v3
class NVP(nn.Module):
    def __init__(self, mask, hidden_size):
        super(NVP, self).__init__()
        self.dim = len(mask)
        self.mask = nn.Parameter(mask, requires_grad=False)

        self.s_func = nn.Sequential(nn.Linear(in_features=self.dim, out_features=hidden_size), nn.LeakyReLU(),
                                    nn.Linear(in_features=hidden_size, out_features=hidden_size), nn.LeakyReLU(),
                                    nn.Linear(in_features=hidden_size, out_features=self.dim))

        self.scale = nn.Parameter(torch.Tensor(self.dim))

        self.t_func = nn.Sequential(nn.Linear(in_features=self.dim, out_features=hidden_size), nn.LeakyReLU(),
                                    nn.Linear(in_features=hidden_size, out_features=hidden_size), nn.LeakyReLU(),
                                    nn.Linear(in_features=hidden_size, out_features=self.dim))

    def forward(self, z):
        z_mask = z*self.mask
        #s = self.s_func(z_mask) * self.scale
        s = self.s_func(z_mask)
        t = self.t_func(z_mask)

        fz = z_mask + (1 - self.mask) * (z*torch.exp(s) + t)

        # Sum for -1, since for every batch, and 1-mask, since the log_det_jac is 1 for y1:d = x1:d.
        log_det_jac = ((1 - self.mask) * s).sum(-1)
        return fz, log_det_jac

    def inverse(self, y):
        y_mask = y * self.mask
        s = self.s_func(y_mask) * self.scale
        t = self.t_func(y_mask)

        x = y_mask + (1-self.mask)*(y - t)*torch.exp(-s)

        inv_log_det_jac = ((1 - self.mask) * -s).sum(-1)

        return x, inv_log_det_jac



#CVAE without Normalizing Flow
'''This is built on the objective of minimizing the KL [q(z|x, y), p(z|x)].
It gives rise to ELBO as : Mean: log(p(x|z, y)) + KL [q(z|x, y), p(z)]

q :- encoder taking input x and y, giving z
p :- encoder taking input y and giving posterior mu and sigma
'''
class CVAE_nnf(nn.Module):
    def __init__(self, args=genargs):
        super().__init__()
        self.batch_size = args.batch_size
        self.device = args.device
        self.z_dim = args.z_dim
        self.img_channels = args.img_channels
        self.model = args.model
        self.attr_dim = args.attribute_dim
        self.attr_embedding_dim = args.attr_embedding_dim
        self.img_dim = args.img_dim

        #Encoders
        self.encoder_q = self.get_encoder_q_86dim(self.img_channels)

        #Embeddings for attributes
        self.attr_embedding = nn.Embedding(2, self.attr_embedding_dim)

        # output size depends on input image size, compute the output size
        demo_input = torch.ones([1, self.img_channels, self.img_dim, self.img_dim])
        demo_attr = torch.rand(1, self.attr_dim, self.attr_embedding_dim)

        self.hidden_dim_q = self.encoder_q(demo_input).shape[1]

        # map to latent z (encoder parameters)
        self.fc_mu_q = nn.Linear(self.hidden_dim_q, self.z_dim)
        self.fc_logsigma_q = nn.Linear(self.hidden_dim_q, self.z_dim)

        # decoder parameters
        '''# conditional VAE that adds the attributes with the latent space'''
        self.fc_zdim_to_dec = nn.Linear(self.z_dim+self.attr_embedding_dim*self.attr_dim, self.hidden_dim_q)

        self.decoder = self.get_decoder_86dim()
        self.log_sigma = 0
        if self.model == 'sigma_vae':
            ## Sigma VAE
            self.log_sigma = torch.nn.Parameter(torch.full((1,), 0, dtype=torch.float32)[0], requires_grad=args.model == 'sigma_vae')

    @staticmethod
    def get_encoder_q_86dim(in_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=4, stride=2, padding=1), # -> 32 x 43 x 43
            nn.ReLU(True),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1), # -> 64 x 21 x 21
            nn.ReLU(True),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1), # -> 128 x 10 x 10
            nn.ReLU(True),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1), # -> 256 x 5 x 5
            nn.ReLU(True),
            Flatten()
        )

    @staticmethod
    def get_decoder_86dim():
        return nn.Sequential(
            UnFlatten(256),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1), # -> 128 x 10 x 10
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1, output_padding=1), # -> 64 x 21 x 21
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1, output_padding=1), # -> 32 x 43 x 43
            nn.ReLU(True),
            nn.ConvTranspose2d(32, 3, kernel_size=4, stride=2, padding=1), # -> 3 x 86 x 86
            #nn.Sigmoid().
            nn.Tanh()
        )

    @staticmethod
    def gaussian_nll(mu, log_sigma, x): #works on the parameters of encoder q
        return 0.5 * torch.pow((x - mu) / log_sigma.exp(), 2) + log_sigma + 0.5 * np.log(2 * np.pi)

    def encode(self, x):
        out = self.encoder_q(x)
        return {'mu_q':self.fc_mu_q(out),
                'logsigma_q':self.fc_logsigma_q(out)}

    def reparameterize(self, mu, logsigma):
        std = torch.exp(logsigma)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z, attr):
        attr_embedded = self.attr_embedding(attr)
        sample_size = z.shape[0]
        z_attr = torch.cat([z, attr_embedded.view(sample_size, self.attr_dim*self.attr_embedding_dim)], dim=1)
        return self.decoder(self.fc_zdim_to_dec(z_attr))

    def sample(self, n):
        sample = torch.randn(n, self.z_dim).to(self.device)
        attr_sampled = torch.randint(0, 2, (n, self.attr_dim)).to(self.device)
        return self.decode(sample, attr_sampled)

    def forward(self, x, attr):
        dist_params = self.encode(x)
        z = self.reparameterize(dist_params['mu_q'], dist_params['logsigma_q'])
        reconstructions = self.decode(z, attr)
        return reconstructions, dist_params

    #losses
    def reconstruction_loss(self, x_hat, x):
        """ Computes the likelihood of the data given the latent variable,
        in this case using a Gaussian distribution with mean predicted by the neural network and variance = 1 """

        if self.model == 'gaussian_vae' or self.model == 'mse_vae':
            # Naive gaussian VAE uses a constant variance
            log_sigma = torch.zeros([], device=x_hat.device)
        elif self.model == 'sigma_vae':
            # Sigma VAE learns the variance of the decoder as another parameter
            log_sigma = self.log_sigma
        elif self.model == 'optimal_sigma_vae':
            log_sigma = ((x - x_hat) ** 2).mean([0,1,2,3], keepdim=True).sqrt().log()
            self.log_sigma = log_sigma.item()
        else:
            raise NotImplementedError

        # Learning the variance can become unstable in some cases. Softly limiting log_sigma to a minimum of -6
        # ensures stable training.
        log_sigma = softclip(log_sigma, -6)

        rec = self.gaussian_nll(x_hat, log_sigma, x).sum() # x_hat is mu in the argument of nll
        return rec

    def total_loss_function(self, recon_x, x, mu, logsigma):
        # Important: both reconstruction and KL divergence loss have to be summed over all element!
        # Here we also sum the over batch and divide by the number of elements in the data later
        if self.model == 'mse_vae':
            rec = torch.nn.MSELoss()(recon_x, x)
        else:
            rec = self.reconstruction_loss(recon_x, x)
        logvar = 2*logsigma
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return rec, kl_loss




#CVAE with Normalizing Flow

'''This is built on the objective of minimizing the KL [q(z|x, y), p(z|x, y)].
It gives rise to ELBO as : Mean: log(p(x|z, y)) + KL [q(z|x, y), p(z|y)]

q :- encoder taking input x and y, giving z
p :- encoder taking input y and giving posterior mu and sigma
'''

class CVAE_nf(nn.Module):
    def __init__(self, args=genargs):
        super().__init__()
        self.batch_size = args.batch_size
        self.device = args.device
        self.z_dim = args.z_dim
        self.img_channels = args.img_channels
        self.model = args.model
        self.attr_dim = args.attribute_dim
        self.attr_embedding_dim = args.attr_embedding_dim
        self.img_dim = args.img_dim
        self.filters_m = 32
        self.filters_y = 32

        #Encoders
        self.encoder_q = self.get_encoder_q_86dim(self.img_channels+1)
        self.encoder_p = self.get_encoder_p(self.attr_embedding_dim, self.filters_y)

        #Embeddings for attributes
        self.attr_embedding = nn.Embedding(2, self.attr_embedding_dim)
        #self.flow = simpleAffineFlow(self.z_dim).to(genargs.device)
        cut = int(self.z_dim/2)
        self.bin_mask = torch.tensor([1 if i<cut else 0 for i in range(self.z_dim)])
        self.nvp_hidden_dim = 16
        self.flow = NVP(self.bin_mask, self.nvp_hidden_dim).to(args.device)

        # output size depends on input image size, compute the output size
        demo_input = torch.ones([1, self.img_channels+1, self.img_dim, self.img_dim])
        demo_attr = torch.rand(1, self.attr_dim, self.attr_embedding_dim)

        self.hidden_dim_p = self.encoder_p(demo_attr.view(1, self.attr_embedding_dim, self.attr_dim)).shape[1]
        self.hidden_dim_q = self.encoder_q(demo_input).shape[1]

        # map to latent z (encoder parameters)
        self.fc_mu_q = nn.Linear(self.hidden_dim_q, self.z_dim)
        self.fc_logsigma_q = nn.Linear(self.hidden_dim_q, self.z_dim)
        self.fc_mu_p = nn.Linear(self.hidden_dim_p, self.z_dim)
        self.fc_logsigma_p = nn.Linear(self.hidden_dim_p, self.z_dim)
        self.fc_attr_img_concat = nn.Linear(self.attr_embedding_dim*self.attr_dim, self.img_dim*self.img_dim)


        # decoder parameters
        '''# conditional VAE that adds the attributes with the latent space'''
        self.fc_zdim_to_dec = nn.Linear(self.z_dim+self.attr_embedding_dim*self.attr_dim, self.hidden_dim_q)

        self.decoder = self.get_decoder_86dim()
        self.log_sigma = 0
        if self.model == 'sigma_vae':
            ## Sigma VAE
            self.log_sigma = torch.nn.Parameter(torch.full((1,), 0, dtype=torch.float32)[0], requires_grad=args.model == 'sigma_vae')


    @staticmethod
    def get_encoder_q_86dim(in_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=4, stride=2, padding=1), # -> 32 x 43 x 43
            nn.ReLU(True),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1), # -> 64 x 21 x 21
            nn.ReLU(True),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1), # -> 128 x 10 x 10
            nn.ReLU(True),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1), # -> 256 x 5 x 5
            nn.ReLU(True),
            Flatten()
        )

    def get_encoder_p(self, in_channels, filters_y):
        assert in_channels == self.attr_embedding_dim
        return nn.Sequential(
            nn.Conv1d(in_channels, filters_y, stride = 2, kernel_size = 3, padding = 1),
            nn.ReLU(),
            nn.Conv1d(filters_y, 2*filters_y, stride = 2, kernel_size = 5, padding = 1),
            nn.ReLU(),
            Flatten()
        )


    @staticmethod
    def get_decoder_86dim():
        return nn.Sequential(
            UnFlatten(256),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1), # -> 128 x 10 x 10
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1, output_padding=1), # -> 64 x 21 x 21
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1, output_padding=1), # -> 32 x 43 x 43
            nn.ReLU(True),
            nn.ConvTranspose2d(32, 3, kernel_size=4, stride=2, padding=1), # -> 3 x 86 x 86
            #nn.Sigmoid()
            nn.Tanh()
        )


    @staticmethod
    def gaussian_nll(mu, log_sigma, x): #works on the parameters of encoder q
        return 0.5 * torch.pow((x - mu) / log_sigma.exp(), 2) + log_sigma + 0.5 * np.log(2 * np.pi)

    def encode(self, x, attr): # should include y (labels)
        attr_embedded = self.attr_embedding(attr) #embedding the attributes
        attr_embedded_transformed = self.fc_attr_img_concat(
            attr_embedded.view(self.batch_size, 1, self.attr_embedding_dim*self.attr_dim)) #doing an affine t on emb attrs

        input = torch.cat([x, attr_embedded_transformed.view(self.batch_size, 1, self.img_dim, self.img_dim)], dim = 1)
        #print(attr_embedded.view(self.batch_size, self.attr_embedding_dim, self.attr_dim).shape)
        hq = self.encoder_q(input)
        hp = self.encoder_p(attr_embedded.view(self.batch_size, self.attr_embedding_dim, self.attr_dim))
        return {'mu_p':self.fc_mu_p(hp),
                'logsigma_p':self.fc_logsigma_p(hp),
                'mu_q':self.fc_mu_q(hq),
                'logsigma_q':self.fc_logsigma_q(hq)}


    def reparameterize(self, mu, logsigma):
        std = torch.exp(logsigma)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z, attr):
        attr_embedded = self.attr_embedding(attr)
        sample_size = z.shape[0]
        z_attr = torch.cat([z, attr_embedded.view(sample_size, self.attr_dim*self.attr_embedding_dim)], dim=1)
        return self.decoder(self.fc_zdim_to_dec(z_attr))

    def get_sample_z(self, attr, batch_size):
        attr_embedded = self.attr_embedding(attr)
        hp = self.encoder_p(attr_embedded.view(batch_size, self.attr_embedding_dim, self.attr_dim))
        mu = self.fc_mu_p(hp)
        logsigma = self.fc_logsigma_p(hp)
        z = self.reparameterize(mu, logsigma)
        #fz, _ = self.flow(z)
        return z

    def sample(self, n):
        sample = torch.randn(n, self.z_dim).to(self.device)
        attr_sampled = torch.randint(0, 2, (n, self.attr_dim)).to(self.device)
        return self.decode(sample, attr_sampled)

    def forward(self, x, attr):
        dist_params = self.encode(x, attr)
        z = self.reparameterize(dist_params['mu_q'], dist_params['logsigma_q'])
        fz, log_det_jacobian = self.flow(z)
        reconstructions = self.decode(z, attr)
        return reconstructions, fz, log_det_jacobian, dist_params

    #losses
    def reconstruction_loss(self, x_hat, x):
        """ Computes the likelihood of the data given the latent variable,
        in this case using a Gaussian distribution with mean predicted by the neural network and variance = 1 """

        if self.model == 'gaussian_vae' or self.model == 'mse_vae':
            # Naive gaussian VAE uses a constant variance
            log_sigma = torch.zeros([], device=x_hat.device)
        elif self.model == 'sigma_vae':
            # Sigma VAE learns the variance of the decoder as another parameter
            log_sigma = self.log_sigma
        elif self.model == 'optimal_sigma_vae':
            log_sigma = ((x - x_hat) ** 2).mean([0,1,2,3], keepdim=True).sqrt().log()
            self.log_sigma = log_sigma.item()
        else:
            raise NotImplementedError

        # Learning the variance can become unstable in some cases. Softly limiting log_sigma to a minimum of -6
        # ensures stable training.
        log_sigma = softclip(log_sigma, -6)

        rec = self.gaussian_nll(x_hat, log_sigma, x).sum() # x_hat is mu in the argument of nll
        return rec

    @staticmethod
    def kl_loss(mu_p, logs_p, logs_q, fz, logdetjac):
        kl = logs_p - logs_q - 0.5 + 0.5 * ((fz - mu_p)**2) * torch.exp(-2. * logs_p)
        kl = kl.sum(dim=1) - logdetjac
        return kl.sum()

    def total_loss_function(self, recon_x, x, mu_p, logs_p, logs_q, fz, logdetjac):
        # Important: both reconstruction and KL divergence loss have to be summed over all element!
        # Here we also sum the over batch and divide by the number of elements in the data later
        if self.model == 'mse_vae':
            rec = torch.nn.MSELoss()(recon_x, x)
        else:
            rec = self.reconstruction_loss(recon_x, x)
        kl_loss = self.kl_loss(mu_p, logs_p, logs_q, fz, logdetjac)
        return rec, kl_loss