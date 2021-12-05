# IPI droplet detection & fringe counting
# General
This is a set of tools for automatic detection and fringe counting of interference patterns in IPI - acquired 
images. Final use of this toolkit is not yet defined, so this will change in the future (based)
on the users feedback etc. etc.)
# Requirements
Everything is written in Python 3.8. To get all the packages needed to run
scripts in the repo:
```
pip install -r requirements.txt
```
# How to use
First you need to append the project's directory to PYTHONPATH
```
export PYTHONPATH="${PYTHONPATH}:/path/to/your/project/"
```

## Use with labelme - ground truths
To detect and count fringes run **/scripts/showcase_detection.py**
```
python scripts/showcase_detection.py
```
At the top of this script you can change the directory (relative to project root)
where labelme's JSONs are stored.

The output will appear in the **./scripts** folder as **det_output.json**

To see some representation of the detected output and comparison with the ground truth,
run: 
```
python scripts/showcase_detection_output.py
```
This will show you the fringe counts histogram and a representation set of 
detected circles with a given fringe count.

## Use on images in general
Did not yet receive any images other than in labelme, so no scripts for that yet

## Evaluation
To see how the detector fares when compared to human labeling, you can run
```
python evaluation/evaluator.py
```
In the script you can change the detection output it loads (default it det_evaluation.json). 
This will show you all the pictures with both sets of detections (human and detector) superimposed 
on the picture. To go across all the pictures, just press any key, if you want to skip to the
end, press escape. At the end master confusion matrix (containing all the detections from the input)
is printed and is saved to det_evaluation.json including individual confusion matrices for each image.

## Output format
There are two types of output files:
* Detector's output
* Evaluation output

Both outputs are in JSON format. The description is bellow.

### Detector output
Output of the detector
```
det_output.json
    * <img_name>
        * gt        # list of ground truth labels
            * name
            * center
            * radius
            * fringe_count
            * shape
            * img_path
            * score
        * det        # list of detected labels
            * name
            * center
            * radius
            * fringe_count
            * shape
            * img_path
            * score
```         

### Evaluation output
Output of the evaluator:
```
det_evaluation.json
    * overall_evaluation
        * img_path      # invalid here, because many paths are used for this evaluation
        * TP
        * FP
        * TN
        * FN
        * SENSITIVITY
        * PRECISION
        * RECALL (Same thing huh, will change)
    * individual_evaluations    # list of the same eval structures for each img
        * img_path
        * TP
        * FP
        * TN
        * FN
        * SENSITIVITY
        * PRECISION
        * RECALL (Same thing huh, will change)
```         


## Detector tuning
To tune the detector, so far, ask martin, later here will be written how to...
