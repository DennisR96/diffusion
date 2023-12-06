import torch
import math

class SinusoidalPositionEmbeddings(torch.nn.Module):
    '''
    Sinusoidal Position Embeddings are used to 
    embed timestemps into the Diffusion Model
    '''
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time: torch.Tensor):
        # Retrieve the Device of torch.Tensor
        device = time.device

        # Half Dim because of Sinus / Cosinus Split
        half_dim = self.dim // 2

        # Scaling Factor of Sinusoidal Functions
        embeddings = math.log(10000) / (half_dim - 1)

        # Exponential Values for Sinus and Cosinus Values
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)

        # Expan time Tensor and element wise multiply with embeddings
        embeddings = time[:, None] * embeddings[None, :]

        # Concatenate Sinus and Cosinus Embeddings 
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings