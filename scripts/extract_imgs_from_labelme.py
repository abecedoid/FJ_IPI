import os
from img_handling.labelme import *
import cv2
from typing import *
import pathlib

"""This script takes all the labelme jsons found in the given directory tree,
extracts the images and saves them to output directory in the same file structure
as .png files"""


# todo - put into img handling
def get_all_files_with_ext_in_tree(dirpath: str, extension: str) -> List[str]:
    dirpath = os.path.abspath(dirpath)
    paths = []

    dir_content = os.listdir(dirpath)

    for ent in dir_content:
        ent_path = os.path.join(dirpath, ent)
        if os.path.isdir(ent_path):
            part_paths = get_all_files_with_ext_in_tree(ent_path, extension)
            paths.extend(part_paths)
        else:
            ext = ent.split('.')[-1]
            if ext == extension:
                paths.append(ent_path)

    return paths


if __name__ == '__main__':
    DIR_IN = '../resources/105mm_60deg.6mxcodhz.000000'
    DIR_OUT = '../resources/imgs_105mm_60deg.6mxcodhz.000000'
    DIR_IN = os.path.abspath(DIR_IN)
    DIR_OUT = os.path.abspath(DIR_OUT)

    json_paths = get_all_files_with_ext_in_tree(DIR_IN, extension='json')

    for json_path in json_paths:
        img = load_labelme_image(json_path)

        # cut out the dir_in from the json path and append dir out to it
        part_path = json_path.replace(DIR_IN, '')
        if part_path[0] == os.sep:                  # relative path cannot start with os separator
            part_path = part_path[1:]
        new_path = os.path.join(DIR_OUT, part_path)
        new_path = new_path.split('.')[:-1]         # remove the file extension
        new_path[-1] = new_path[-1] + '.png'        # add desired file extension
        new_path = os.path.join(*new_path)
        new_path_dir = os.path.dirname(new_path)

        # create new path
        pathlib.Path(new_path_dir).mkdir(parents=True, exist_ok=True)

        cv2.imwrite(new_path, img)
        print('Extracted img {} to {}'.format(json_path, new_path))

