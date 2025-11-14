import torch

class General_Arguments:
  def __init__(self, batch_size, model_type, z_dim, attr_embedding_dim, epochs, 
               log_interval, attr_embedding_proj_dim, attribute_dim, img_dim, img_channels) -> None:
    self.batch_size = batch_size
    self.model = model_type  # Which model to use:  mse_vae,  gaussian_vae, or sigma_vae or optimal_sigma_vae
    self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    self.z_dim = z_dim
    self.attr_embedding_dim = attr_embedding_dim #dimension of the conditional's embedding
    self.epochs = epochs
    self.log_interval = log_interval # to log every nth batch
    self.attr_embedded_proj_dim = attr_embedding_proj_dim

    self.attribute_dim = attribute_dim #dimension of the conditionals, here y vector
    self.img_dim = img_dim #Assuming hxh
    self.img_channels = img_channels

genargs = General_Arguments(batch_size = 128,
  model_type= 'optimal_sigma_vae',
  z_dim = 64,
  attr_embedding_dim = 8,
  epochs = 3,
  log_interval = 400,
  attr_embedding_proj_dim = 16,
  attribute_dim = 30,
  img_dim = 86,
  img_channels = 3)