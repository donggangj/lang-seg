import numpy as np
from PIL import Image
from torchvision import transforms as transforms


def prepare_image(image_path='samples/cat1.png'):
    image = Image.open(image_path)
    image = np.array(image)
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ]
    )
    image = transform(image).unsqueeze(0)
    return image


def load_ref_data(data_path='original_result.npz'):
    try:
        f = np.load(data_path)
        ref = f['output']
    except KeyError:
        f = np.load(data_path)
        ref = f[list(f.keys())[0]]
    except ValueError:
        ref = np.array([])
    return ref
