import os
from evaluation.evaluator import *


"""This is just for going across images, showing detctions without any ground truth"""


def load_img(path: str) -> np.ndarray:
    img = cv2.imread(path, 0)   # want greyscale
    return img


if __name__ == '__main__':

    OUTPUT_JSON_FILEPATH = 'imgs_det_output.json'
    OUTPUT_JSON_FILEPATH = os.path.abspath(OUTPUT_JSON_FILEPATH)
    SHOW_IMAGES = True

    data = load_detector_output(OUTPUT_JSON_FILEPATH)

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
