import os
from evaluation.evaluator import *
import argparse


"""This is just for going across json images, showing detections without any ground truth"""


def load_img(path: str) -> np.ndarray:
    img = cv2.imread(path, 0)   # want greyscale
    return img


if __name__ == '__main__':

    # ARGUMENT PARSING
    parser = argparse.ArgumentParser(description='This is just for going across the detector\'s output images, \
                                                 showing detections without any ground truth'
                                                 'Any key skips to next picture, Escape stops the script"')
    parser.add_argument('-i', '--inp', help='path to detector json output')
    args = vars(parser.parse_args())

    OUTPUT_JSON_FILEPATH = os.path.abspath(args['inp'])
    if not os.path.exists(OUTPUT_JSON_FILEPATH):
        print('The specified file {} does not exists.'.format(OUTPUT_JSON_FILEPATH))

    data = load_detector_output(OUTPUT_JSON_FILEPATH)
    SHOW_IMAGES = True

    for imname, ostruct in data.items():  # across all images
        dets = ostruct['det']

        det_dls = json_det_structs2droplet_list(dets)

        if SHOW_IMAGES:
            # load the image
            for det in dets:
                dl = DropletLabel.init_dict(det)
                if os.path.exists(dl.img_path):
                    impath = dl.img_path
                    break
            img = load_img(impath)

            # show the image
            plot_multiple_droplet_lists_on_image({'dets': det_dls}, img=img, wait_key=False)
            if cv2.waitKey(0) == 27:
                SHOW_IMAGES = False
