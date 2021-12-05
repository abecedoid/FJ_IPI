import yaml
import os


def load_configuration() -> dict:
    PATH2CONFIG = os.path.abspath('config.yaml')

    try:
        with open(PATH2CONFIG, 'r') as f:
            config = yaml.safe_load(f)
            return config
    except FileNotFoundError as e:
        print('Could not find {} file. Make sure there is a config.yaml file in the root of the project\'s folder, '
              'error: {}'.format(PATH2CONFIG, e))


# todo - these two guys are the same, rewrite
def get_detector_settings() -> dict:
    config = load_configuration()
    try:
        det_settings = config['detector-settings']
        return det_settings
    except ValueError as e:
        print('The config yaml file is broken, could not find \'detector-settings\', error: {}'.format(e))


def get_evaluation_settings() -> dict:
    config = load_configuration()
    try:
        det_settings = config['evaluation-settings']
        return det_settings
    except ValueError as e:
        print('The config yaml file is broken, could not find \'evaluation-settings\', error: {}'.format(e))


