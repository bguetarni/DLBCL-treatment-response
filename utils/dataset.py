import numpy as np
from skimage import filters, morphology

def scale_coordinates(wsi, p, source_level, target_level):

    if not isinstance(p, np.ndarray):
        p = np.asarray(p).squeeze()

    assert p.ndim < 3 and p.shape[-1] == 2, 'coordinates must be a single point or an array of 2D-cooridnates'

    # source level dimensions
    source_w, source_h = wsi.level_dimensions[source_level]
    
    # target level dimensions
    target_w, target_h = wsi.level_dimensions[target_level]
    
    # scale coordinates
    p = np.array(p)*(target_w/source_w, target_h/source_h)
    
    # round to int64
    return np.floor(p).astype('int64')


def get_tile_mask(wsi, level, mask, mask_level, x, y, patch_size):
    
    # convert coordinates from slide level to mask level
    x_ul, y_ul = scale_coordinates(wsi, (x, y), level, mask_level)
    x_br, y_br = scale_coordinates(wsi, (x + patch_size, y + patch_size), level, mask_level)

    return mask[y_ul:y_br, x_ul:x_br]


def get_tissue_mask_hsv(wsi, mask_level, black_threshold=100):
    """
    Args:
        slide : whole slide file (openslide.OpenSlide)
        mask_level : level from which to build the mask (int)

    Return np.ndarray of bool as mask
    """
    
    # get slide image into PIL.Image
    thumbnail = wsi.read_region((0, 0), mask_level, wsi.level_dimensions[mask_level])

    # convert to HSV
    hsv = np.array(thumbnail.convert('HSV'))
    H, S, V = np.moveaxis(hsv, -1, 0)

    # filter out black pixels
    V_mask = V > black_threshold

    S_filtered = S[V_mask]
    S_threshold = filters.threshold_otsu(S_filtered)
    S_mask = S > S_threshold

    H_filtered = H[V_mask]
    H_threshold = filters.threshold_otsu(H_filtered)
    H_mask = H > H_threshold

    mask = np.logical_and(np.logical_and(H_mask, S_mask), V_mask)
    mask = morphology.binary_dilation(mask)

    return mask
