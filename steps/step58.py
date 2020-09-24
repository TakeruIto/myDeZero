import sys
sys.path.append('..')
import numpy as np
from PIL import Image
import dezero
from dezero.models import VGG16

model = VGG16(pretrained=True)

url = 'https://github.com/oreilly-japan/deep-learning-from-scratch-3/raw/images/zebra.jpg'

img_path = dezero.utils.get_file(url)
img = Image.open(img_path)
x = VGG16.preprocess(img)
x = x[np.newaxis]

with dezero.test_mode():
    y = model(x)
predict_id = np.argmax(y.data)

labels = dezero.datasets.ImageNet.labels()
print(labels[predict_id])
