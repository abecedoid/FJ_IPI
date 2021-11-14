import os
import argparse
import sys
sys.path.append('../')
from helpers.labeled_jsons import load_labelme_image, plot_image

parser = argparse.ArgumentParser(description='Displays the image in the labelme\' json file')
parser.add_argument('json_path', help='path to the json file')
args = parser.parse_args()

try:
    path2json = os.path.abspath(args.json_path)
except Exception as e:
    print('Invalid path, {}'.format(e))
    sys.exit()

print('Loading {} ...'.format(path2json))
img = load_labelme_image(path2json)
plot_image(img)


