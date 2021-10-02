import matplotlib.pyplot as plt
import numpy as np
import PIL.Image
import tensorflow as tf
from tf_keras_vis.activation_maximization import ActivationMaximization
from tf_keras_vis.gradcam import GradcamPlusPlus
from tf_keras_vis.utils import normalize
from tf_keras_vis.utils.callbacks import Print


preprocess_dict = {'VGG16': tf.keras.applications.vgg16.preprocess_input,
        'VGG19': tf.keras.applications.vgg19.preprocess_input,
        'ResNet50': tf.keras.applications.resnet50.preprocess_input,
        'EfficientNet': tf.keras.applications.efficientnet.preprocess_input}


def activation_maximization(model_file: str, target: int, output_file: str):
    """
    target:
        0 for the most EBV+MSI-like image
        1 for the most other-like image
    """
    model = tf.keras.models.load_model(model_file)

    def model_modifier(m):
        m.layers[-1].activation = tf.keras.activations.linear

    am = ActivationMaximization(model, model_modifier, clone=False)

    def loss(output):
        if target == 0:
            return -output
        else:
            return output

    activation = am(loss, steps=512, callbacks=[Print(interval=50)])
    image = activation[0].astype(np.uint8)

    image = PIL.Image.fromarray(image)
    image.save(output_file)


def gradcam(model_file: str, base_model: str, source_file: str):
    """
    base_model: 'VGG16', 'VGG19', 'ResNet50', 'EfficientNet'
    """
    model = tf.keras.models.load_model(model_file)

    image = tf.keras.preprocessing.image.load_img(source_file)
    x = tf.keras.preprocessing.image.img_to_array(image)
    preprocess = preprocess_dict[base_model]
    x = preprocess(x)

    def model_modifier(m):
        m.layers[-1].activation = tf.keras.activations.linear

    def loss(output):
        return -output

    gradcam = GradcamPlusPlus(model, model_modifier=model_modifier, clone=False)
    cam = gradcam(loss, x, penultimate_layer=-1)
    cam = normalize(cam)
    subplot_args = {'nrows': 1, 'ncols': 1, 'figsize': (20, 20),
            'subplot_kw': {'xticks': [], 'yticks': []}}
    _, ax = plt.subplots(**subplot_args)
    hm = np.uint8(plt.cm.jet(cam[0])[..., :3] * 255)
    ax.imshow(image)
    ax.imshow(hm, cmap='jet', alpha=0.25)
    plt.tight_layout()
    plt.show()
