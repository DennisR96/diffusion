# Denoising Diffusion Probalistic Models
import torch
from tqdm.auto import tqdm

def scheduler(beta_schedule, timesteps, *, s=0.008):
    if beta_schedule == "cosine_beta_schedule":
        steps = timesteps + 1
        x = torch.linspace(0, timesteps, steps)
        alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        return torch.clip(betas, 0.0001, 0.9999)


class DDPM():
    def __init__(self, config):
        self.config = config
        self.timesteps = self.config.scheduler.num_diffusion_timestemps

        # Beta Schedule
        self.betas = scheduler(beta_schedule=self.config.scheduler.beta_schedule, timesteps=self.timesteps)

        # Alphas
        self.alphas = 1. - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, axis=0)
        self.alphas_cumprod_prev = torch.nn.functional.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)
        self.sqrt_recip_alphas = torch.sqrt(1.0 / self.alphas)

        # q(x_t | x_{t-1}) 
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - self.alphas_cumprod)

        # q(x_{t-1} | x_t, x_0)
        self.posterior_variance = self.betas * (1. - self.alphas_cumprod_prev) / (1. - self.alphas_cumprod)
        
    def extract(self, a, t, x_shape):
        batch_size = t.shape[0]
        out = a.gather(-1, t.cpu())
        return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)
    
    # Forward Diffusion
    def q_sample(self, x_start, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start)

        self.sqrt_alphas_cumprod_t = self.extract(self.sqrt_alphas_cumprod, t, x_start.shape)
        self.sqrt_one_minus_alphas_cumprod_t = self.extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape)

        return self.sqrt_alphas_cumprod_t * x_start + self.sqrt_one_minus_alphas_cumprod_t * noise
    
    def p_losses(self, denoise_model, x_start, t, noise=None, loss_type="l1"):
        if noise is None:
            noise = torch.randn_like(x_start)

        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
        predicted_noise = denoise_model(x_noisy, t)

        if loss_type == 'l1':
            loss = torch.nn.functional.l1_loss(noise, predicted_noise)
        elif loss_type == 'l2':
            loss = torch.nn.functional.mse_loss(noise, predicted_noise)
        elif loss_type == "huber":
            loss = torch.nn.functional.smooth_l1_loss(noise, predicted_noise)
        else:
            raise NotImplementedError()

        return loss
    
    @torch.no_grad()
    def p_sample(self, model):
        '''
        p Sampling Function
        return: List[timestemp, torch.Tensor[batch_size, channel, width, height]]'''
        shape = [self.config.data.batch_size, self.config.data.channel, self.config.data.image_size, self.config.data.image_size]

        device = next(model.parameters()).device

        b = shape[0]
        # start from pure noise (for each example in the batch)
        img = torch.randn(shape, device=device)
        imgs = []

        for i in tqdm(reversed(range(0, self.timesteps)), desc='sampling loop time step', total=self.timesteps):
            t = torch.full((b,), i, device=device, dtype=torch.long)
            x = img

            self.betas_t = self.extract(self.betas, t, x.shape)
            self.sqrt_one_minus_alphas_cumprod_t = self.extract(self.sqrt_one_minus_alphas_cumprod, t, x.shape)
            self.sqrt_recip_alphas_t = self.extract(self.sqrt_recip_alphas, t, x.shape)

            self.model_mean = self.sqrt_recip_alphas_t * (x - self.betas_t * model(x, t) / self.sqrt_one_minus_alphas_cumprod_t)
            if i == 0:
                img =  self.model_mean
            else:
                self.posterior_variance_t = self.extract(self.posterior_variance, t, x.shape)
                noise = torch.randn_like(x)
                img = self.model_mean + torch.sqrt(self.posterior_variance_t) * noise 
            imgs.append(img)
        return imgs


        


