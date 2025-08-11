import numpy as np
import matplotlib.pyplot as plt
from numpy.random import randint
from skimage.metrics import structural_similarity as ssim
from skimage.segmentation import mark_boundaries
from lime import lime_image
import tensorflow as tf
from tensorflow.keras.models import load_model
from numpy import load, vstack
import cv2

# Load and prepare training images
def load_real_samples(filename):
    data = load(filename)
    X1, X2 = data['arr_0'], data['arr_1']
    X1 = (X1 - 127.5) / 127.5
    X2 = (X2 - 127.5) / 127.5
    return [X1, X2]

# Compute SSIM between two generated and real images
def compute_ssim(img1, img2):
    img1 = np.uint8((img1 + 1) * 127.5)
    img2 = np.uint8((img2 + 1) * 127.5)
    s = 0
    for i in range(3):
        s += ssim(img1[:, :, i], img2[:, :, i], data_range=255)
    return s / 3

# Define prediction function for LIME (returns similarity to real map)
def predict_fn(perturbed_imgs):
    scores = []
    for img in perturbed_imgs:
        img = cv2.resize(img, (256, 256))
        img_input = (img / 127.5) - 1.0  # Normalize to [-1, 1]
        img_input = np.expand_dims(img_input, axis=0)
        pred_map = generator.predict(img_input)[0]
        score = compute_ssim(pred_map, tar_image[0])
        scores.append(score)
    return np.array(scores).reshape(-1, 1)

# Explain with LIME
def explain_with_lime(image):
    explainer = lime_image.LimeImageExplainer()
    explanation = explainer.explain_instance(
        np.uint8((image + 1) * 127.5),
        classifier_fn=predict_fn,
        top_labels=1,
        hide_color=0,
        num_samples=1000,
        batch_size=10
    )
    temp, mask = explanation.get_image_and_mask(
        explanation.top_labels[0],
        positive_only=True,
        num_features=10,
        hide_rest=False
    )
    lime_img = mark_boundaries(temp / 255.0, mask)
    return lime_img

# Plot with LIME explanation
def plot_images(src_img, gen_img, tar_img, lime_overlay, explanation=""):
    images = vstack((src_img, gen_img, tar_img))
    images = (images + 1) / 2.0
    titles = ['Satellite (Input)', 'Generated Map', 'Actual Map']

    plt.figure(figsize=(14, 6))
    for i in range(len(images)):
        plt.subplot(2, 4, 1 + i)
        plt.axis('off')
        plt.imshow(images[i])
        plt.title(titles[i])

    plt.subplot(2, 4, 4)
    plt.axis('off')
    plt.imshow(lime_overlay)
    plt.title("LIME Influence Zones")

    plt.subplot(2, 1, 2)
    plt.axis('off')
    plt.text(0.01, 0.5, explanation, wrap=True, fontsize=10)
    plt.tight_layout()
    plt.show()

# Load dataset and model
[X1, X2] = load_real_samples('maps_256.npz')
generator = load_model(r'C:\Users\A\Satellite-images-to-real-maps-with-Deep-Learning\checkpoints\model.h5')

# Pick a sample
ix = randint(0, len(X1), 1)
src_image, tar_image = X1[ix], X2[ix]
gen_image = generator.predict(src_image)

# LIME
lime_overlay = explain_with_lime(src_image[0])

# Explanation Text
explanation = (
    "🧠 This LIME visualization highlights regions in the satellite image that most positively influenced the generator's output.\n"
    "✅ We compute SSIM between the generated map and ground truth for each perturbed input to quantify importance.\n"
    "💡 Bright areas indicate higher influence — helping interpret which parts of the satellite image guide map generation the most."
)

# Display
plot_images(src_image, gen_image, tar_image, lime_overlay, explanation)
