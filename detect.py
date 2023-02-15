import json
from pathlib import Path
from typing import Dict
import numpy as np
import click
import cv2
from tqdm import tqdm

#DEBUG BEGIN
debug_enabled=False
'''
cv2.namedWindow('image')
cv2.namedWindow('mask')
def empty_callback(value):
    pass
cv2.createTrackbar('H-Lower', 'image', 40, 179,empty_callback)
cv2.createTrackbar('S-Lower', 'image', 0, 255,empty_callback)
cv2.createTrackbar('V-Lower', 'image', 9, 255,empty_callback)
cv2.createTrackbar('H-Higher', 'image', 80, 179,empty_callback)
cv2.createTrackbar('S-Higher', 'image', 255, 255,empty_callback)
cv2.createTrackbar('V-Higher', 'image', 255, 255,empty_callback)
'''
#DEBUG END

def brightnessBump(bgr,bump):
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    limiter = 255 - bump
    v[v > limiter] = 255
    v[v <= limiter] += bump
    hsv = cv2.merge((h, s, v))
    img = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return bgr,hsv

def adjust_gamma(image, gamma=1.0):
	invGamma = 1.0 / gamma
	table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
	return cv2.LUT(image, table)

def imageInitialProcessing(image,colorType='null'):
    #RESIZE
    dim=(720,int((image.shape[0])*720/image.shape[1]))
    resized = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)
    if colorType=='Yellow':
        bgr = cv2.normalize(resized, None, alpha=50, beta=200, norm_type=cv2.NORM_MINMAX)
        bgr = cv2.dilate(bgr, np.ones((3, 3), np.uint8))
        bgr = cv2.medianBlur(bgr, 5)
        bgr, hsv=brightnessBump(bgr,250)
        bgr=adjust_gamma(bgr,0.1)
        bgr, hsv = brightnessBump(bgr, 0)
        bgr = adjust_gamma(bgr, 4)
    elif colorType=='Green':
        bgr = cv2.normalize(resized, None, alpha=50, beta=200, norm_type=cv2.NORM_MINMAX)
        bgr = cv2.dilate(bgr, np.ones((3, 3), np.uint8))
        bgr = cv2.medianBlur(bgr, 5)
        bgr, hsv=brightnessBump(bgr,250)
        bgr=adjust_gamma(bgr,0.1)
        bgr, hsv = brightnessBump(bgr, 0)
        bgr = adjust_gamma(bgr, 4)
    elif colorType=='Red':
        bgr = cv2.normalize(resized, None, alpha=50, beta=200, norm_type=cv2.NORM_MINMAX)
        bgr = cv2.dilate(bgr, np.ones((3, 3), np.uint8))
        bgr = cv2.medianBlur(bgr, 5)
        bgr, hsv=brightnessBump(bgr,250)
        bgr=adjust_gamma(bgr,2)
        bgr, hsv = brightnessBump(bgr, 50)
        bgr = adjust_gamma(bgr, 0.1)
    elif colorType=='Purple':
        bgr = cv2.normalize(resized, None, alpha=50, beta=200, norm_type=cv2.NORM_MINMAX)
        bgr = cv2.dilate(bgr, np.ones((3, 3), np.uint8))
        bgr = cv2.medianBlur(bgr, 5)
        bgr, hsv=brightnessBump(bgr,250)
        bgr=adjust_gamma(bgr,2)
        bgr, hsv = brightnessBump(bgr, 50)
        bgr = adjust_gamma(bgr, 0.1)
    else:
        bgr = cv2.normalize(resized, None, alpha=50, beta=200, norm_type=cv2.NORM_MINMAX)
        bgr = cv2.dilate(bgr, np.ones((3, 3), np.uint8))
        bgr = cv2.medianBlur(bgr, 5)
        bgr, hsv=brightnessBump(bgr,250)
        bgr=adjust_gamma(bgr,2)
        bgr, hsv = brightnessBump(bgr, 50)
        bgr = adjust_gamma(bgr, 0.1)
    return bgr,hsv

def createMask(hsv,H_l, S_l, V_l,H_h, S_h, V_h,morphology=False):
    mask = cv2.inRange(hsv, np.array([H_l, S_l, V_l]), np.array([H_h, S_h, V_h]))
    if morphology:
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    masked = cv2.bitwise_and(hsv, hsv, mask=mask)
    return mask,masked

def searchCountours(mask,image):
    cont, hier = cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_TC89_L1)
    count=0
    for contour in cont:
        #DEBUG BEGIN
        '''
        if debug_enabled:
            cv2.drawContours(image,[contour],-1,(0,255,0),2)
        '''
        #DEBUG END
        count+=1
    return count

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
    #READ IMAGE
    ogImage = cv2.imread(img_path, cv2.IMREAD_COLOR)
    #DEBUG BEGIN
    '''
    while debug_enabled:
        rgb, hsv = imageInitialProcessing(ogImage,'Green')
        H_lower = cv2.getTrackbarPos('H-Lower', 'image')
        S_lower = cv2.getTrackbarPos('S-Lower', 'image')
        V_lower = cv2.getTrackbarPos('V-Lower', 'image')
        H_higher = cv2.getTrackbarPos('H-Higher', 'image')
        S_higher = cv2.getTrackbarPos('S-Higher', 'image')
        V_higher = cv2.getTrackbarPos('V-Higher', 'image')
        tempMaskLow=np.array([H_lower,S_lower,V_lower])
        tempMaskHigh = np.array([H_higher, S_higher, V_higher])
        tempMask = cv2.inRange(hsv, tempMaskLow, tempMaskHigh)
        if True:
            kernel = np.ones((5, 5), np.uint8)
            tempMask = cv2.morphologyEx(tempMask, cv2.MORPH_CLOSE, kernel)
            tempMask = cv2.morphologyEx(tempMask, cv2.MORPH_OPEN, kernel)
        tempMasked = cv2.bitwise_and(hsv, hsv, mask=tempMask)
        count = searchCountours(tempMask,rgb)
        cv2.imshow('image', rgb)
        cv2.imshow('mask', tempMask)
        key_code = cv2.waitKey(10)
        if key_code == 27: # escape key pressed
            #print(count)
            break
    '''
    #DEBUG END
    #CODE BEGINS
    # # YELLOW BEGIN # #
    rgbY, hsvY = imageInitialProcessing(ogImage,'Yellow')
    yellowMask,yellowMasked=createMask(hsvY,12,40,20,30,255,255,morphology=True)
    yellowCount = searchCountours(yellowMask, yellowMasked)
    # # YELLOW END   # #
    # # GREEN BEGIN # #
    rgbG, hsvG = imageInitialProcessing(ogImage, 'Green')
    greenMask, greenMasked = createMask(hsvG, 40, 0, 9, 80, 255, 255, morphology=True)
    greenCount = searchCountours(greenMask, greenMasked)
    # # GREEN END   # #
    # # RED BEGIN # #
    rgbR, hsvR = imageInitialProcessing(ogImage, 'Red')
    redMask, redMasked = createMask(hsvR, 172, 15, 231, 179, 255, 255, morphology=True)
    redCount = searchCountours(redMask, redMasked)
    # # RED END   # #
    # # PURPLE BEGIN # #
    rgbP, hsvP = imageInitialProcessing(ogImage, 'Purple')
    purpleMask, purpleMasked = createMask(hsvP, 130, 17, 0, 176, 255, 241, morphology=True)
    purpleCount = searchCountours(purpleMask, purpleMasked)
    # # PURPLE END   # #
    #DEBUG BEGIN
    '''
    print('\nRED=',redCount,'\nYELLOW=',yellowCount,'\nGREEN=',greenCount,'\nPURPLE=',purpleCount)
    '''
    #DEBUG END
    return {'red': redCount, 'yellow': yellowCount, 'green': greenCount, 'purple': purpleCount}

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