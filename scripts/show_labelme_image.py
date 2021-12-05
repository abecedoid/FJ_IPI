import os
import argparse
import sys
sys.path.append('..')
sys.path.append('.')
from img_handling.labelme import load_labelme_image
from img_handling.plotters import plot_image


"""This is just a convenience script to show either labelme image or multiple labelme images in a dir"""


parser = argparse.ArgumentParser(description='Displays the image in the labelme\' json file')
parser.add_argument('json_path', help='path to the labelme json file or a dir containing labelme json files')
args = parser.parse_args()


try:
    path2json = os.path.abspath(args.json_path)
except Exception as e:
    print('Invalid path, {}'.format(e))
    sys.exit()

if os.path.isdir(path2json):
    json_filenames = os.listdir(path2json)
    for filename in json_filenames:
        path = os.path.join(path2json, filename)
        print('Loading {} ...'.format(path2json))
        img = load_labelme_image(path)
        plot_image(img)
else:
    print('Loading {} ...'.format(path2json))
    img = load_labelme_image(path2json)
    plot_image(img)


