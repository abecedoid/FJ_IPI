import os
import glob
from img_handling.droplets import *
from img_handling.labelme import *
from detector.circle_detector import detect_circles, preprocess_img
from detector.fringe_count import *
from detector.detector_configuration import *
import json
from typing import List


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


"""Goes through all the resources and creates evaluation for all of them"""
if __name__ == '__main__':

    RESOURCE_DIR = '../resources/IPI_labelme/'
    RESOURCE_DIR = os.path.abspath(RESOURCE_DIR)
    file_paths = get_all_files_with_ext_in_tree(RESOURCE_DIR, 'json')

    settings = get_detector_settings()

    det_json = {}

    for file_path in file_paths:
        # detect
        filename = file_path.split(os.sep)[-1]
        img = load_labelme_image(file_path)
        pimg = preprocess_img(img, settings['CLAHE_clip_limit'], settings['CLAHE_grid_size'])
        gt_droplets = load_labelme_droplet_labels(file_path)

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
        gt_droplets_json = [x.json() if x is not None else {} for x in gt_droplets]
        det_droplets_json = [x.json() if x is not None else {} for x in det_droplets]

        det_json[filename] = {'gt': gt_droplets_json,
                              'det': det_droplets_json}

    with open('all_resource_detections.json', 'w') as f:
        json.dump(det_json, f, indent=4)

    print('All done!')

