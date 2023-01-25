import json
from pathlib import Path
from typing import Dict
import numpy as np

import click
import cv2
from tqdm import tqdm


def detect(img_path: str) -> Dict[str, int]:
    """Object detection function, according to the project description, to implement.
    ###
    Parameters
    ----------
    img_path : str
        Path to processed image.
    ###
    Returns
    -------
    Dict[str, int]
        Dictionary with quantity of each object.
    """

    lower_yellow = np.array([15, 50, 180])
    upper_yellow = np.array([40, 255, 255])
    def empty_callback(value):
        pass
    OriginalImage = cv2.imread(img_path, cv2.IMREAD_COLOR)
    hsv = cv2.cvtColor(OriginalImage, cv2.COLOR_BGR2HSV)
    cv2.namedWindow('image')
    mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
    result = cv2.bitwise_and(OriginalImage, OriginalImage, mask=mask)
    while True:
        cv2.imshow('image', mask)
        key_code = cv2.waitKey(10)
        if key_code == 27:
            break

    #TODO: Implement detection method.
    
    red = 0
    yellow = 0
    green = 0
    purple = 0

    return {'red': red, 'yellow': yellow, 'green': green, 'purple': purple}


@click.command()
@click.option('-p', '--data_path', help='Path to data directory', type=click.Path(exists=True, file_okay=False, path_type=Path), required=True)
@click.option('-o', '--output_file_path', help='Path to output file', type=click.Path(dir_okay=False, path_type=Path), required=True)
def main(data_path: Path, output_file_path: Path):
    img_list = data_path.glob('*.jpg')

    results = {}

    for img_path in tqdm(sorted(img_list)):
        fruits = detect(str(img_path))
        results[img_path.name] = fruits

    with open(output_file_path, 'w') as ofp:
        json.dump(results, ofp)


if __name__ == '__main__':
    main()
