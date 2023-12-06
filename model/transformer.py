import torch
import einops
from model.helpers import default, exists

_ATTN_PRECISION = "fp32"

class Attention(torch.nn.Module):
    def __init__(self, dim, heads=4, dim_head=32):
        super().__init__()

        # Scaling Factor for Attention weights
        self.scale = dim_head**-0.5

        # Number of Attention Heads with Dimension per Head
        self.heads = heads
        hidden_dim = dim_head * heads

        # Linear Transformation -> Query, Key, Value
        self.to_qkv = torch.nn.Conv2d(dim, hidden_dim * 3, 1, bias=False)

        # Linear Transformation -> Output
        self.to_out = torch.nn.Conv2d(hidden_dim, dim, 1)

    def forward(self, x):
        # Input Tensor Shape [batch_size, channels, height, width]
        b, c, h, w = x.shape

        # Step 1: Linear Transformation
        # Project x into Query, Key, Value and split into 3 Chunks along dim=1
        # Rearrange Tensor: [batch_size, (height, channel), x, y] -> [batch_size, height, channel, (x, y)]
        qkv = self.to_qkv(x).chunk(3, dim=1)
        q, k, v = map(lambda t: einops.rearrange(t, "b (h c) x y -> b h c (x y)", h=self.heads), qkv)


        # Step 2: Compute Scaled Dot Product
        q = q * self.scale
        sim = torch.einsum("b h d i, b h d j -> b h i j", q, k)


        # Normalize Attention Scores
        sim = sim - sim.amax(dim=-1, keepdim=True).detach()
        attn = sim.softmax(dim=-1)

        # Compute Output from Softmax 
        out = torch.einsum("b h i j, b h d j -> b h i d", attn, v)

        # Step 3: Rearrange Output Tensor and Apply Output Transformation
        out = einops.rearrange(out, "b h (x y) d -> b (h d) x y", x=h, y=w)
        return self.to_out(out)
    

class CrossAttention(torch.nn.Module):
    '''
    https://github.com/Stability-AI/stablediffusion/blob/main/ldm/modules/attention.py
    '''
    def __init__(self, query_dim, context_dim=None, heads=8, dim_head=64, dropout=0.):
        super().__init__()

        context_dim = default(context_dim, query_dim)
        
        # Scaling Factor for Attention Weight
        self.scale = dim_head ** -0.5

        # Number of Attention Heads with Dimension per Head
        inner_dim = dim_head * heads
        self.heads = heads

        # Linear Transformation -> Query, Key, Value
        self.to_q = torch.nn.Linear(query_dim, inner_dim, bias=False)
        self.to_k = torch.nn.Linear(context_dim, inner_dim, bias=False)
        self.to_v = torch.nn.Linear(context_dim, inner_dim, bias=False)

        # Output
        self.to_out = torch.nn.Sequential(
            torch.nn.Linear(inner_dim, query_dim),
            torch.nn.Dropout(dropout)
        )

    def forward(self, x, context=None, mask=None):
        h = self.heads

        # Cross-Attention: q -> x: torch.Tensor; k, y -> context: torch.Tensor
        q = self.to_q(x)
        context = default(context, x)
        k = self.to_k(context)
        v = self.to_v(context)

        # Rearrange Tensor
        q, k, v = map(lambda t: einops.rearrange(t, 'b n (h d) -> (b h) n d', h=h), (q, k, v))

        # Compute Scaled Dot Product q, k
        if _ATTN_PRECISION =="fp32":
            with torch.autocast(enabled=False, device_type = 'cuda'):
                q, k = q.float(), k.float()
                sim = einops.einsum('b i d, b j d -> b i j', q, k) * self.scale
        else:
            sim = einops.einsum('b i d, b j d -> b i j', q, k) * self.scale
        
        del q, k
    
        if exists(mask):
            mask = einops.rearrange(mask, 'b ... -> b (...)')
            max_neg_value = -torch.finfo(sim.dtype).max
            mask = einops.repeat(mask, 'b j -> (b h) () j', h=h)
            sim.masked_fill_(~mask, max_neg_value)

        # Apply Softmax
        sim = sim.softmax(dim=-1)

        # Compute Output -> Einsum & Rearrange
        out = einops.einsum('b i j, b j d -> b i d', sim, v)
        out = einops.rearrange(out, '(b h) n d -> b n (h d)', h=h)
        return self.to_out(out)
