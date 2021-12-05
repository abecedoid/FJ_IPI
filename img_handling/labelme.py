import json
import numpy as np
from img_handling import img_transformations
from img_handling.droplets import DropletLabel


def load_labelme_image(path2json: str) -> np.ndarray:
    """Loads image from string data contained in the LabelMe's JSON"""
    with open(path2json, 'r') as f:
        data = json.load(f)

    image_data = data.get('imageData')

    img = img_transformations.img_b64_to_arr(image_data)

    return img


def load_labelme_droplet_labels(path2json: str) -> list:
    """Loads a list of labelme's droplet labels"""
    with open(path2json, 'r') as f:
        data = json.load(f)

    shapes_data = data.get('shapes')

    droplets = []
    for shape in shapes_data:
        try:
            drop = DropletLabel.init_labelme(labelme_struct=shape, labelme_json_path=path2json)
            droplets.append(drop)
            # droplets.append(DropletLabel.init_labelme(labelme_struct=shape))
        except Exception as e:
            droplets.append(DropletLabel(center_pt=[0, 0], radius=0, name='fail'))
            print('Failed to load DropletLabel from {} because {}'.format(shape, e))

    return droplets

