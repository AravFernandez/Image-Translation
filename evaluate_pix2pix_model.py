from tensorflow.keras.models import load_model, Model
from tensorflow.keras import backend as K
from numpy import load, vstack
from matplotlib import pyplot
from numpy.random import randint
import numpy as np
import tensorflow as tf
import cv2

# Load and prepare training images
def load_real_samples(filename):
    data = load(filename)
    X1, X2 = data['arr_0'], data['arr_1']
    X1 = (X1 - 127.5) / 127.5
    X2 = (X2 - 127.5) / 127.5
    return [X1, X2]

# Plot source, generated, target, and Grad-CAM heatmap + caption
def plot_images(src_img, gen_img, tar_img, heatmap=None, explanation=""):
    images = vstack((src_img, gen_img, tar_img))
    images = (images + 1) / 2.0
    titles = ['Satellite (Input)', 'Generated Map', 'Actual Map']

    pyplot.figure(figsize=(14, 6))
    for i in range(len(images)):
        pyplot.subplot(2, 4, 1 + i)
        pyplot.axis('off')
        pyplot.imshow(images[i])
        pyplot.title(titles[i])
    
    if heatmap is not None:
        pyplot.subplot(2, 4, 4)
        pyplot.axis('off')
        pyplot.imshow(heatmap)
        pyplot.title('Grad-CAM Heatmap')

    # Add textual explanation below
    pyplot.subplot(2, 1, 2)
    pyplot.axis('off')
    pyplot.text(0.01, 0.5, explanation, wrap=True, fontsize=10)
    pyplot.tight_layout()
    pyplot.show()

# Grad-CAM Implementation
def compute_gradcam(model, image, layer_name='conv2d_12'):
    grad_model = Model([model.inputs], [model.get_layer(layer_name).output, model.output])
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(image)
        loss = tf.reduce_mean(predictions)

    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = np.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    heatmap = heatmap.numpy()

    # Convert to RGB and resize
    heatmap = cv2.resize(heatmap, (256, 256))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    # Overlay heatmap on original
    src_img_disp = ((image[0] + 1) / 2.0).numpy()
    src_img_disp = np.uint8(255 * src_img_disp)
    overlayed = cv2.addWeighted(src_img_disp, 0.6, heatmap, 0.4, 0)
    return overlayed

# Load dataset
[X1, X2] = load_real_samples('maps_256.npz')
print('Loaded:', X1.shape, X2.shape)

# Load model
model = load_model(r'C:\Users\A\Satellite-images-to-real-maps-with-Deep-Learning\checkpoints\model.h5')

# Select random sample
ix = randint(0, len(X1), 1)
src_image, tar_image = X1[ix], X2[ix]

# Predict
gen_image = model.predict(src_image)

# Compute Grad-CAM
heatmap = compute_gradcam(model, tf.convert_to_tensor(src_image), layer_name='conv2d_12')

# Explanation Text
explanation = (
    "🔴 Red and orange regions in the Grad-CAM heatmap indicate high attention by the generator model, "
    "meaning those areas significantly influenced the map generation.\n"
    "🔵 Blue and green areas represent low attention.\n\n"
    "💡 The generator is focusing on structured features like roads, coastline boundaries, or high-contrast zones in the satellite image, "
    "as these regions provide important spatial cues for generating accurate and context-aware maps."
)

# Plot all
plot_images(src_image, gen_image, tar_image, heatmap, explanation)
