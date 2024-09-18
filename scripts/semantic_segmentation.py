import os
import requests
from io import BytesIO
import tarfile
from six.moves import urllib
from matplotlib import gridspec
from matplotlib import pyplot as plt
import numpy as np
from PIL import Image
import tensorflow as tf

#local file paths
base_dir = os.path.join(os.path.expanduser('~'), 'Desktop/ml/image_processing')
model_dir = os.path.join(base_dir, 'models')

#create the model directory if it doesn't exist
if not os.path.exists(model_dir):
    os.makedirs(model_dir)

#function to download model files if they don't exist
def download_file(url, file_name):
    if not os.path.exists(file_name):
        print(f"Downloading {file_name}...")
        response = requests.get(url)
        with open(file_name, 'wb') as file:
            file.write(response.content)
        print(f"{file_name} downloaded successfully.")
    else:
        print(f"{file_name} already exists, skipping download.")

#class to load DeepLab model from frozen graph and run inference
class DeepLabModel(object):

    INPUT_SIZE = 513
    INPUT_TENSOR_NAME = 'ImageTensor:0'
    OUTPUT_TENSOR_NAME = 'SemanticPredictions:0'

    def __init__(self, model_dir):
        #loads a frozen DeepLab model from a .pb file
        self.graph = tf.Graph()

        #search for the frozen_inference_graph.pb in model_dir and subdirectories
        frozen_graph_filename = self.find_frozen_graph(model_dir)
        if not frozen_graph_filename:
            raise IOError(f"frozen graph file not found in {model_dir}")

        print(f"Loading frozen graph from {frozen_graph_filename}...")
        with tf.io.gfile.GFile(frozen_graph_filename, "rb") as f:
            graph_def = tf.compat.v1.GraphDef()
            graph_def.ParseFromString(f.read())

        #import the graph to the current graph
        with self.graph.as_default():
            tf.import_graph_def(graph_def, name="")

        #create a session for running inference
        self.sess = tf.compat.v1.Session(graph=self.graph)
        print("Model loaded successfully!")

    def find_frozen_graph(self, model_dir):
        #search for frozen_inference_graph.pb in the model directory and subdirectories
        for root, dirs, files in os.walk(model_dir):
            for file in files:
                if file == 'frozen_inference_graph.pb':
                    return os.path.join(root, file)
        return None

    #runs inference on a single image, returns a RGB image resized
    #from original input imageand a segmentation map of `resized_image`.
    def run(self, image):

        width, height = image.size
        resize_ratio = 1.0 * self.INPUT_SIZE / max(width, height)
        target_size = (int(resize_ratio * width), int(resize_ratio * height))
        resized_image = image.convert('RGB').resize(target_size, Image.Resampling.LANCZOS)

        #convert the image to a tensor
        input_tensor = np.asarray(resized_image)
        input_tensor = np.expand_dims(input_tensor, axis=0)  #add batch dimension

        #run inference
        seg_map = self.sess.run(self.OUTPUT_TENSOR_NAME, feed_dict={self.INPUT_TENSOR_NAME: input_tensor})

        return resized_image, seg_map[0]

#creates a label colormap used in PASCAL VOC segmentation benchmark
def create_pascal_label_colormap():
    colormap = np.zeros((256, 3), dtype=int)
    ind = np.arange(256, dtype=int)

    for shift in reversed(range(8)):
        for channel in range(3):
            colormap[:, channel] |= ((ind >> channel) & 1) << shift
        ind >>= 3

    return colormap

#adds color defined by the dataset colormap to the label
def label_to_color_image(label):
    if label.ndim != 2:
        raise ValueError('expect 2-D input label')

    colormap = create_pascal_label_colormap()

    if np.max(label) >= len(colormap):
        raise ValueError('label value too large.')

    return colormap[label]

#visualizes input image, segmentation map and overlay view
def vis_segmentation(image, seg_map):
    plt.figure(figsize=(15, 5))
    grid_spec = gridspec.GridSpec(1, 4, width_ratios=[6, 6, 6, 1])

    plt.subplot(grid_spec[0])
    plt.imshow(image)
    plt.axis('off')
    plt.title('input image')

    plt.subplot(grid_spec[1])
    seg_image = label_to_color_image(seg_map).astype(np.uint8)
    plt.imshow(seg_image)
    plt.axis('off')
    plt.title('segmentation map')

    plt.subplot(grid_spec[2])
    plt.imshow(image)
    plt.imshow(seg_image, alpha=0.7)
    plt.axis('off')
    plt.title('segmentation overlay')

    unique_labels = np.unique(seg_map)
    ax = plt.subplot(grid_spec[3])
    plt.imshow(
        FULL_COLOR_MAP[unique_labels].astype(np.uint8), interpolation='nearest')
    ax.yaxis.tick_right()
    plt.yticks(range(len(unique_labels)), LABEL_NAMES[unique_labels])
    plt.xticks([], [])
    ax.tick_params(width=0.0)
    plt.grid(False)
    plt.show()

#inferences DeepLab model and visualizes result
def run_visualization(url):
    try:
        f = urllib.request.urlopen(url)
        jpeg_str = f.read()
        original_im = Image.open(BytesIO(jpeg_str))
    except IOError:
        print('cannot retrieve image. please check url: ' + url)
        return

    print('running deeplab on image %s...' % url)
    resized_im, seg_map = MODEL.run(original_im)

    vis_segmentation(resized_im, seg_map)


LABEL_NAMES = np.asarray([
    'background', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus',
    'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike',
    'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tv'
])

FULL_LABEL_MAP = np.arange(len(LABEL_NAMES)).reshape(len(LABEL_NAMES), 1)
FULL_COLOR_MAP = label_to_color_image(FULL_LABEL_MAP)

MODEL_NAME = 'mobilenetv2_coco_voctrainaug'

_DOWNLOAD_URL_PREFIX = 'http://download.tensorflow.org/models/'
_MODEL_URLS = {
    'mobilenetv2_coco_voctrainaug':
        'deeplabv3_mnv2_pascal_train_aug_2018_01_29.tar.gz',
    'mobilenetv2_coco_voctrainval':
        'deeplabv3_mnv2_pascal_trainval_2018_01_29.tar.gz',
    'xception_coco_voctrainaug':
        'deeplabv3_pascal_train_aug_2018_01_04.tar.gz',
    'xception_coco_voctrainval':
        'deeplabv3_pascal_trainval_2018_01_04.tar.gz',
}
_TARBALL_NAME = 'deeplab_model.tar.gz'

#download the model to the local model_dir
download_path = os.path.join(model_dir, _TARBALL_NAME)
download_file(_DOWNLOAD_URL_PREFIX + _MODEL_URLS[MODEL_NAME], download_path)

#extract the model tarball into model_dir
print('extracting model...')
with tarfile.open(download_path) as tar:
    tar.extractall(path=model_dir)

#initialize DeepLabModel with the extracted frozen graph
MODEL = DeepLabModel(model_dir)

SAMPLE_IMAGE = 'image1'
IMAGE_URL = ''

_SAMPLE_URL = ('https://github.com/tensorflow/models/blob/master/research/'
               'deeplab/g3doc/img/%s.jpg?raw=true')

image_url = IMAGE_URL or _SAMPLE_URL % SAMPLE_IMAGE
run_visualization(image_url)