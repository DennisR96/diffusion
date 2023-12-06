import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import scipy

# Parameters
dt = 0.001                              # Time step
T = 2.0                                 # Total time
n = int(T / dt)                         # Number of time steps
t = np.linspace(0., T, n)               # Vector of times
x = np.zeros((n, 256, 256))             # Position Vector

# x0 – Initialize Position
x0_PIL = Image.open("playground/NS-VR-LINER-IMAGES copy 2.webp").resize((256, 256)).convert("L")
x0_Numpy = np.array(x0_PIL) / 255
x[0] = x0_Numpy

def beta(t):
        ''' Initialize Beta Schedule'''
        return 0.9 + 0.1 * t / T

# Forward Process
for i in range(n - 1):
    # 1. Generate Wiener increments
    dW = np.sqrt(dt) * np.random.randn(*(256,256))

    # 2. Compute the drift term
    drift = -0.5 * beta(t[i]) * x[i] * dt

    # 3. Compute the diffusion term
    diffusion = np.sqrt(beta(t[i])) * dW

    # 4. Update the position
    x[i + 1] = x[i] + drift + diffusion


# Reverse Process
def score(x):
    x = x.flatten()
    print(x[0])
    print(x0_Numpy.shape)
    images = [x0_Numpy]
    pixel_values = np.concatenate([image.flatten() for image in images])
    kde = scipy.stats.gaussian_kde(x)
    log_pdf_values = kde.logpdf(x)
    score = np.gradient(log_pdf_values)
    score = score.reshape((256, 256))
    return score

# Initialize Position Vector
x = np.zeros((n, 256, 256))             # Position Vector
x[-1] = np.random.normal(0, 25, size=(1, 256, 256))

for i in reversed(range(n-1)):
    # 1. Generate Wiener increments
    dW = np.sqrt(dt) * np.random.randn(*(256,256))

    # 2. Compute Drift Term
    drift = (-0.5 * beta(t[i]) * x[i] - beta(t[i]) * score(x[i], t[i])) * dt

    # 3. Compute the diffusion term
    diffusion = np.sqrt(beta(t[i])) * dW

    # 4. Update the position
    x[i + 1] = x[i] + drift + diffusion

    
plt.imshow(x[0] * 255, cmap="gray")
plt.show()