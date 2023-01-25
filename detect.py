import json
from pathlib import Path
from typing import Dict
import numpy as np

import click
import cv2
from tqdm import tqdm

#MASKS
lower_green=np.array([30, 150, 0])
upper_green=np.array([70, 255, 255])

lower_purple=np.array([60, 30, 0])
upper_purple=np.array([170, 255, 255])

lower_yellow=np.array([0, 160, 120])
upper_yellow=np.array([30, 255, 255])

lower_red = np.array([165,37,110])
upper_red = np.array([180,255,255])

def imageInitialProcessing(image):
    dim=(720,int((image.shape[0])*720/image.shape[1]))
    im = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)
    return im

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

    #CODE
    ogImage= cv2.imread(img_path, cv2.IMREAD_COLOR)
    resized=imageInitialProcessing(ogImage)
    hsv = cv2.cvtColor(resized, cv2.COLOR_BGR2HSV)

    green_mask=cv2.inRange(hsv, lower_green, upper_green)
    purple_mask = cv2.inRange(hsv, lower_purple, upper_purple)
    yellow_mask=cv2.inRange(hsv,lower_yellow,upper_yellow)
    red_mask = cv2.inRange(hsv, lower_red, upper_red)

    resultGreen = cv2.bitwise_and(hsv,hsv,mask=green_mask)
    resultPurple = cv2.bitwise_and(hsv, hsv, mask=purple_mask)
    resultYellow= cv2.bitwise_and(hsv, hsv, mask=yellow_mask)
    resultRed = cv2.bitwise_and(hsv, hsv, mask=red_mask)

    cv2.imshow('base', resized)
    #cv2.imshow('green',resultGreen)
    #cv2.imshow('purple', resultPurple)
    #cv2.imshow('yellow', resultYellow)
    cv2.imshow('red', resultRed)
    cv2.waitKey()


    #END CODE
    #TODO: PROVIDE ANSWEAR
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
