import os
import glob
from img_handling.droplets import *
from img_handling.labelme import *
from detector.circle_detector import detect_circles, preprocess_img
from detector.fringe_count import *
from detector.detector_configuration import *
import json
from typing import List
import cv2
import sys
import argparse

"""This script takes directory path as input, finds all the *.png images inside and creates
a detector json output"""


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


def load_img(path: str) -> np.ndarray:
    img = cv2.imread(path, 0)   # want greyscale
    return img


"""Goes through all the resources and creates evaluation for all of them"""
if __name__ == '__main__':

    # ARGUMENT PARSING
    parser = argparse.ArgumentParser(description='Detect droplets and count fringes from IPI images')
    parser.add_argument('-d', '--dir', help='path to directory with *.png images')
    parser.add_argument('-o', '--out', help='path with filename of detector output')
    args = vars(parser.parse_args())

    RESOURCE_DIR = os.path.abspath(args['dir'])
    if not os.path.exists(RESOURCE_DIR):
        print('Please specify the directory containing the *.png images, specified now: {}'.format(RESOURCE_DIR))
        sys.exit()

    OUTPUT_PATH = os.path.abspath(args['out'])

    # check that there is .json if not, append
    if not 'json' in os.path.basename(OUTPUT_PATH).split('.'):
        OUTPUT_PATH += '.json'

    OUTPUT_DIR = os.path.dirname(OUTPUT_PATH)
    if not os.path.exists(OUTPUT_DIR):
        print('Please specify the output to existing directory')
        sys.exit()

    # SCRIPT ITSELF STARTS HERE
    file_paths = get_all_files_with_ext_in_tree(RESOURCE_DIR, 'png')
    settings = get_detector_settings()
    det_json = {}

    for file_path in file_paths:
        # detect
        filename = file_path.split(os.sep)[-1]
        img = load_img(file_path)
        pimg = preprocess_img(img, settings['CLAHE_clip_limit'], settings['CLAHE_grid_size'])

        try:
            coords = detect_circles(pimg, DS_COEFF=settings['ds_coeff'],
                                    circle_mask_rad=settings['circle_mask_rad'],
                                    circle_mask_wdth=settings['circle_mask_wdth'],
                                    circle_mask_radoff_size=settings['circle_mask_radoff_size'],
                                    pxcorr1=settings['pxcorr1'], pxcorr2=settings['pxcorr2'],
                                    peakfind_thr=settings['peakfind_thr'],
                                    peakfind_min_max_nghbr=settings['peakfind_min_max_nghbr'],
                                    debug=settings['debug'])

            det_droplets = coords2droplet_labels_list(coords, circle_radius=settings['circle_mask_rad'],
                                                      img_path=file_path)
        except Exception as e:
            print('Failed detecting droplets in img {}, cause: {}'.format(file_path, e))

        # FRINGE COUNT
        for det_droplet in det_droplets:
            try:
                ds = droplet_slice_from_image(img, det_droplet)
                n_fringes, DS, pk_coords, score = count_fringes(ds)
                ds.score = score
                det_droplet.slice = ds
                det_droplet.fringe_count = n_fringes

            except SliceOutOfBoundsError as se:
                print('Slice is out of bounds, cannot count fringes on incomplete circle')
                continue
            # else:
            except Exception as e:
                print('Fringe count failed on a droplet {}, because {}'.format(det_droplet.name, e))
                continue

        # JSONIFY THE OUTPUTS
        det_droplets_json = [x.json() if x is not None else {} for x in det_droplets]

        det_json[filename] = {'gt': [],
                              'det': det_droplets_json}

    with open(OUTPUT_PATH, 'w') as f:
        json.dump(det_json, f, indent=4)

    print('All done! Output is in {}'.format(OUTPUT_PATH))

