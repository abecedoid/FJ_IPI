import sys
sys.path.append('../')
from img_handling.droplets import coords2droplet_labels_list
from img_handling.labelme import load_labelme_image, load_labelme_droplet_labels
from pprint import pprint
from detector.circle_detector import detect_circles
from evaluation.evaluator import DetectionEvaluator
from matplotlib import pyplot as plt


dsettings_normal = {
    'ds_coeff': 2,
    'circle_mask_rad': 30,
    'circle_mask_wdth': None,
    'circle_mask_radoff_size': 5,
    'pxcorr1': 80,
    'pxcorr2': 80,
    'peakfind_thr': 0.1,
    'peakfind_min_max_nghbr': 30,
    'debug': True
}

dsettings_tune = {
    'ds_coeff': 2,
    'circle_mask_rad': 30,
    'circle_mask_wdth': None,
    'circle_mask_radoff_size': 5,
    'pxcorr1': 70,
    'pxcorr2': 70,
    'peakfind_thr': 0.1,
    'peakfind_min_max_nghbr': 30,
    'debug': False
}


if __name__ == '__main__':
    s = dsettings_tune
    IMPATH = '../../resources/105mm_60deg.6mxcodhz.000000/105mm_60deg.6mxcodhz.000000.json'
    img = load_labelme_image(IMPATH)
    gt_droplets = load_labelme_droplet_labels(IMPATH)

    # FIND KNEE
    precisions = []
    recalls = []
    set2pr = {}
    for k in range(30):
        coords = detect_circles(img, DS_COEFF=s['ds_coeff'],
                                circle_mask_rad=s['circle_mask_rad'],
                                circle_mask_wdth=s['circle_mask_wdth'],
                                circle_mask_radoff_size=s['circle_mask_radoff_size'],
                                pxcorr1=s['pxcorr1'], pxcorr2=s['pxcorr2'],
                                peakfind_thr=s['peakfind_thr'],
                                peakfind_min_max_nghbr=s['peakfind_min_max_nghbr'],
                                debug=s['debug'])

        det_droplets = coords2droplet_labels_list(coords, circle_radius=30)

        devaluator = DetectionEvaluator(detections=det_droplets, labeled=gt_droplets)
        devaluator.evaluate()
        precisions.append(devaluator.confusion_mat.precision())
        recalls.append(devaluator.confusion_mat.recall())
        set2pr[s['pxcorr1']] = (precisions[-1], recalls[-1], precisions[-1] + recalls[-1])
        s['pxcorr1'] += 1
        s['pxcorr2'] += 1
        # print(devaluator.confusion_mat)

    # find the
    pprint(set2pr)

    plt.figure()
    plt.plot(recalls, precisions)
    plt.show()

    # SHOW ONE EXAMPLE

    # coords = detect_circles(img, DS_COEFF=s['ds_coeff'],
    #                         circle_mask_rad=s['circle_mask_rad'],
    #                         circle_mask_wdth=s['circle_mask_wdth'],
    #                         circle_mask_radoff_size=s['circle_mask_radoff_size'],
    #                         pxcorr1=s['pxcorr1'], pxcorr2=s['pxcorr2'],
    #                         peakfind_thr=s['peakfind_thr'],
    #                         peakfind_min_max_nghbr=s['peakfind_min_max_nghbr'],
    #                         debug=s['debug'])
    #
    # det_droplets = coords2droplet_labels_list(coords, circle_radius=30)
    #
    # devaluator = DetectionEvaluator(detections=det_droplets, labeled=gt_droplets, image=img)
    # devaluator.evaluate()
    # print(devaluator.confusion_mat)
    #
    # plot_dict = {'gt': gt_droplets, 'det': det_droplets}
    # plot_multiple_droplet_lists_on_image(plot_dict, img)
