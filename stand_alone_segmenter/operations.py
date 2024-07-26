# Logger library compatible with multiprocessing
from loguru import logger

import cv2

__mask_to_remove = None


def adaptative_threshold(segmenter,img):
    """Apply a threshold to a color image to get a mask from it
    Uses an adaptative threshold with a blocksize of 19 and reduction of 4.

    Args:
        img (cv2 img): Image to extract the mask from

    Returns:
        cv2 img: binary mask
    """
    # start = time.monotonic()
    # logger.debug(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)

    logger.debug("Adaptative threshold calc")
    # img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # ret, mask = cv2.threshold(img_gray, 127, 200, cv2.THRESH_OTSU)
    mask = cv2.adaptiveThreshold(
        img_gray,
        maxValue=255,
        adaptiveMethod=cv2.ADAPTIVE_THRESH_MEAN_C,
        thresholdType=cv2.THRESH_BINARY_INV,
        blockSize=19,  # must be odd
        C=4,
    )
    # mask = 255 - img_tmaskhres

    # logger.debug(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    # logger.debug(time.monotonic() - start)
    # logger.success(f"Threshold used was {ret}")
    logger.success("Adaptative threshold is done")
    return mask


def simple_threshold(segmenter,img):
    """Apply a threshold to a color image to get a mask from it

    Args:
        img (cv2 img): Image to extract the mask from

    Returns:
        cv2 img: binary mask
        operation_parameters : dict of parameters for metadata
    """
    # start = time.monotonic()
    # logger.debug(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)

    logger.debug("Simple threshold calc")
    # img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, mask = cv2.threshold(
        img_gray, 127, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_TRIANGLE
    )
    operation_parameters= {
        "type": "simple_threshold",
        "parameters": {"algorithm": "THRESH_TRIANGLE"}
    }
    # logger.debug(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    # logger.debug(time.monotonic() - start)
    logger.info(f"Threshold value used was {ret}")
    logger.success("Simple threshold is done")
    return mask, operation_parameters


def erode(segmenter, mask):
    """Erode the given mask with a rectangular kernel of 2x2

    Args:
        mask (cv2 img): mask to erode

    Returns:
        cv2 img: binary mask after transformation
        operation_parameters : dict of parameters for metadata
    """
    logger.info("Erode calc")
    # start = time.monotonic()
    # logger.debug(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    
    kernel = cv2.getStructuringElement(segmenter.kernel_shape_erode, segmenter.kernel_size_erode)
    mask_erode = cv2.erode(mask, kernel)
    if segmenter.kernel_shape_erode==0:
        shape="rectangle"
    elif segmenter.kernel_shape_erode==2:
        shape="ellipse"
    operation_parameters={
        "type": "erode",
        "parameters": {"kernel_size": segmenter.kernel_size_erode[0], "kernel_shape": shape},
    } 
    # logger.debug(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    # logger.debug(time.monotonic() - start)
    logger.success("Erode calc")
    return mask_erode, operation_parameters


def dilate(segmenter, mask):
    """Apply a dilate operation to the given mask, with an elliptic kernel of 8x8

    Args:
        mask (cv2 img): mask to apply the operation on

    Returns:
        cv2 img: mask after the transformation
        operation_parameters : dict of parameters for metadata
    """
    logger.info("Dilate calc")
    # start = time.monotonic()
    # logger.debug(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    
    kernel = cv2.getStructuringElement(segmenter.kernel_shape_dilate, segmenter.kernel_size_dilate )
    mask_dilate = cv2.dilate(mask, kernel)

    if segmenter.kernel_shape_dilate==2:
        shape="ellipse"
    elif segmenter.kernel_shape_dilate==0:
        shape="rectangle"

        
    operation_parameters={
        "type": "dilate",
        "parameters": {"kernel_size": segmenter.kernel_size_dilate[0], "kernel_shape": shape},
    }
    # logger.debug(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    # logger.debug(time.monotonic() - start)
    logger.success("Dilate calc")
    return mask_dilate, operation_parameters


def close(segmenter, mask):
    """Apply a close operation to the given mask, with an elliptic kernel of 8x8

    Args:
        mask (cv2 img): mask to apply the operation on

    Returns:
        cv2 img: mask after the transformation
        operation_parameters : dict of parameters for metadata
    """
    logger.info("Close calc")
    # start = time.monotonic()
    # logger.debug(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)

    
    kernel = cv2.getStructuringElement(segmenter.kernel_shape_close, segmenter.kernel_size_close)
    mask_close = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    
    if segmenter.kernel_shape_close==2:
        shape="ellipse"     
    elif segmenter.kernel_shape_close==0:
        shape="rectangle"

    operation_parameters={
        "type": "close",
        "parameters": {"kernel_size": segmenter.kernel_size_close[0], "kernel_shape": shape},
    }
    # logger.debug(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    # logger.debug(time.monotonic() - start)
    logger.success("Close calc")
    return mask_close, operation_parameters


def erode2(segmenter, mask):
    """Apply an erode operation to the given mask, with an elliptic kernel of 8x8

    Args:
        mask (cv2 img): mask to apply the operation on

    Returns:
        cv2 img: mask after the transformation
        operation_parameters : dict of parameters for metadata
    """
    logger.info("Erode calc 2")
    # start = time.monotonic()
    # logger.debug(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    
    kernel = cv2.getStructuringElement(segmenter.kernel_shape_erode2, segmenter.kernel_size_erode2)
    mask_erode_2 = cv2.erode(mask, kernel)
    
    if segmenter.kernel_shape_erode2==2:
        shape="ellipse"       
    elif segmenter.kernel_shape_erode2==0:
        shape="rectangle"
    
    operation_parameters= {
        "type": "erode",
        "parameters": {"kernel_size": segmenter.kernel_size_erode2[0], "kernel_shape": shape},
    }
    # logger.debug(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    # logger.debug(time.monotonic() - start)
    logger.success("Erode calc 2")
    return mask_erode_2, operation_parameters


def remove_previous_mask(segmenter, mask):
    """Remove the mask from the previous pass from the given mask
    The given mask is then saved to be applied to the next pass

    Args:
        mask (cv2 img): mask to apply the operation on

    Returns:
        cv2 img: mask after the transformation
        operation_parameters : dict of parameters for metadata
    """
    global __mask_to_remove
    operation_parameters={
        "type": "remove_previous_mask",
        "parameters": {},
        }
    if __mask_to_remove is not None:
        # start = time.monotonic()
        # np.append(__mask_to_remove, img_erode_2)
        # logger.debug(time.monotonic() - start)
        mask_and = mask & __mask_to_remove
        mask_final = mask - mask_and
        logger.success("Done removing the previous mask")
        __mask_to_remove = mask
        
        return mask_final,operation_parameters
    else:
        logger.debug("First mask")
        __mask_to_remove = mask
        return __mask_to_remove, operation_parameters


def reset_previous_mask():
    """Remove the mask from the previous pass from the given mask
    The given mask is then saved to be applied to the next pass

    Args:
        mask (cv2 img): mask to apply the operation on

    Returns:
        cv2 img: mask after the transformation
    """
    global __mask_to_remove
    __mask_to_remove = None
