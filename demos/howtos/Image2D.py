"""
The use of Image2D
==================

This script demonstrates the use of :class:`~cuqi.geometry.Image2D` for domain and range geometries in CUQIpy forward models. The settings of this geometry, in particular the so-called 'visual_only' flag, depend on how the forward model input is operated on in the forward model. `visual_only` flag tells the geometry to interpret the variable as a 2D image only when visualizing the variable but keep it as a vector when being operated on in the forward model. Generally, if your forward operator expects a vector input, set `visual_only` to `True` in the domain geometry. If it expects an image input, set `visual_only` to `False`, which is the default. In the same way, the range geometry `visual_only` setting should be consistent with your forward operator output.
"""

# %%
# Import necessary libraries
import cuqi
import numpy as np
import matplotlib.pyplot as plt

# %%
# Define dimensions
dim_x, dim_y = 2, 4

# %%
# Example with `visual_only=False`
# With `visual_only=False`, the underlying structure of a CUQI array or sample will be reshaped to the shape of the specified geometry, i.e., an image of size (dim_x, dim_y), before being passed to the model.
# Here we intend to define a forward operator that downsamples the input image by a factor of 2. It's easier to define such a forward operator in terms of an image, so we set `visual_only` to `False`.
domain_geom_a = cuqi.geometry.Image2D((dim_x, dim_y), visual_only=False)
range_geom = cuqi.geometry.Image2D((dim_x // 2, dim_y // 2))

def forward_func_a(image):
    assert image.shape == (dim_x, dim_y)
    return image[::2, ::2]

model_A = cuqi.model.LinearModel(forward_func_a, domain_geometry=domain_geom_a, range_geometry=range_geom)
print(model_A)

x_sample = cuqi.array.CUQIarray(np.linspace(0, 1, dim_x * dim_y), geometry=domain_geom_a)

# Plot the original sample
plt.figure()
x_sample.plot(cmap='Greens')
plt.title("X image")

# Apply the model and plot the result
plt.figure()
y_sample = model_A @ x_sample
y_sample.plot(cmap='Greens')
plt.title("Y image (after Model A)")

# %%
# Example with `visual_only=True`
# With `visual_only=True`, the underlying structure of a cuqi array or sample will not be changed before being passed to the model. This is useful when the model expects a vector input. Here we define a forward operator that reverses the input vector, which can be easily defined in terms of a vector.
def forward_func_b(x):
    assert x.shape == (dim_x * dim_y, )
    return x[::-1]

model_B = cuqi.model.LinearModel(forward_func_b, domain_geometry=cuqi.geometry.Image2D((dim_x, dim_y), visual_only=True), range_geometry=cuqi.geometry.Image2D((dim_x, dim_y), visual_only=True))
print(model_B)

# Apply the model and plot the result
plt.figure()
y_sample = model_B @ x_sample
y_sample.plot(cmap='Greens')
plt.title("Y image (after Model B)")