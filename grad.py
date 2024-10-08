import numpy as np
import cv2
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import tensorflow as tf

# Load your trained model
model = load_model('best_mri_model.h5')

# Print model summary
model.summary()

# Load and preprocess the image
def load_and_preprocess_image(image_path):
    img = load_img(image_path, target_size=(128, 128))
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0  # Normalize to [0, 1]
    return img_array

# Define Grad-CAM function
def grad_cam(model, img_array, pred_index=None):
    resnet = model.get_layer('resnet50')
    last_conv_layer = resnet.get_layer('conv5_block3_out')
    
    grad_model = tf.keras.models.Model(
        [model.inputs],
        [last_conv_layer.output, model.output]
    )
    
    with tf.GradientTape() as tape:
        conv_output, predictions = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(predictions[0])
        class_channel = predictions[:, pred_index]
    
    grads = tape.gradient(class_channel, conv_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    
    conv_output = conv_output[0]
    heatmap = conv_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    
    heatmap = tf.maximum(heatmap, 0) / tf.reduce_max(heatmap)
    return heatmap.numpy()

# Example usage
image_path = "C:/Users/Medha Agarwal/Desktop/GANs/brain-tumor-mri-dataset/Training/glioma/Tr-gl_0011.jpg"
img_array = load_and_preprocess_image(image_path)

# Generate class activation heatmap
heatmap = grad_cam(model, img_array)

# Display the heatmap
plt.matshow(heatmap)
plt.title('Grad-CAM Heatmap')
plt.colorbar()
plt.show()

# Load the original image
img = load_img(image_path, target_size=(128, 128))
img = img_to_array(img)

# Resize heatmap to match original image dimensions
heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))

# Convert heatmap to RGB
heatmap = np.uint8(255 * heatmap)
heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

# Superimpose the heatmap on original image
superimposed_img = heatmap * 0.4 + img
superimposed_img = np.clip(superimposed_img, 0, 255).astype('uint8')

# Display the superimposed image
plt.imshow(superimposed_img / 255)
plt.title('Superimposed Grad-CAM')
plt.axis('off')
plt.show()

# Get the predicted class
predictions = model.predict(img_array)
predicted_class = np.argmax(predictions[0])
class_names = ['glioma', 'meningioma', 'no tumor', 'pituitary']
print(f"Predicted class: {class_names[predicted_class]}")
print(f"Confidence: {predictions[0][predicted_class]:.2f}")