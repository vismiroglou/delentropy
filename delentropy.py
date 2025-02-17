import numpy as np
import cv2
from matplotlib import pyplot as plt

def crop_and_resize(img, width:int, height:int) -> np.ndarray:
    """
    Crop and resize an image to the specified width and height while maintaining the aspect ratio.

    Parameters:
    img (numpy.ndarray): The input image.
    width (int): The target width of the image. If None, the original width is kept.
    height (int): The target height of the image. If None, the original height is kept.

    Returns:
    numpy.ndarray: The cropped and resized image.
    """
    img_height, img_width = img.shape[:2]

    width = img_width if width is None else width
    height = img_height if height is None else height
    if width == img_width and height == img_height:
        return img
    
    img_aspect_ratio = img_width / img_height
    target_aspect_ratio = width / height

    if img_aspect_ratio > target_aspect_ratio:
        # Image is wider than target aspect ratio
        new_width = int(img_height * target_aspect_ratio)
        start_x = (img_width - new_width) // 2
        cropped_img = img[:, start_x:start_x + new_width]
    elif img_aspect_ratio < target_aspect_ratio:
        # Image is taller than target aspect ratio
        new_height = int(img_width / target_aspect_ratio)
        start_y = (img_height - new_height) // 2
        cropped_img = img[start_y:start_y + new_height, :]
    else:
        cropped_img = img
    
    resized_img = cv2.resize(cropped_img, (width, height)).astype(np.float32)

    return resized_img


DEFAULTKERNEL = np.array([[-1+0j, 0+1j], [0-1j, 1+0j]], dtype=np.complex128)
def calc_grads(src:np.ndarray, kern:np.ndarray) -> np.ndarray:
    """
    Compute the gradient of an image using a specified kernel.

    Parameters:
    src (numpy.ndarray): The input image.
    kern (numpy.ndarray): The kernel to be used for computing the gradient.

    Returns:
    numpy.ndarray: The computed gradient of the image.
    """
    def by_kernel(kern:np.ndarray, pix, right, below, belowRight) -> np.complex128:
        return kern[0,0]*complex(pix, 0) + kern[0,1]*complex(right, 0) + kern[1,0]*complex(below, 0) + kern[1,1]*complex(belowRight, 0)
    
    grad = np.zeros((src.shape[0]-1) * (src.shape[1]-1), dtype=np.complex128)
    dsti = 0
    for y in range(src.shape[0] - 1):
        for x in range(src.shape[1] - 1):
            val = by_kernel(kern, src[y,x], src[y,x+1], src[y+1,x], src[y+1,x+1])
            grad[dsti] = val
            dsti += 1
    return grad


# minSparseExcursion = 1024
def calc_hist(grad:np.ndarray) -> np.ndarray:
    maxExcursion, width, height = compute_hist_size(grad)

    #TODO: Implement sparse histogram
    # flatHistSize = flatSize(width, height)
    # maxSparseSize = max_sparse_hist_size(grad)
    # if (maxExcursion > minSparseExcursion) and (maxSparseSize < flatHistSize):
    #     print("Using sparse histogram")
    #     hist = make_sparse_hist(grad, width, height)
    # else:
    print("Using flat histogram")
    hist = make_flat_hist(grad, width, height)
    return hist


def max_sparse_hist_size(grad:np.ndarray) -> int:
    #TODO: Implement
    pass


def make_sparse_hist(grad:np.ndarray, width:int, height:int) -> np.ndarray:
    #TODO: Implement
    pass


def compute_hist_size(grad:np.ndarray) -> tuple:
    maxRealExcursion = np.max(np.abs(np.real(grad)))
    maxImagExcursion = np.max(np.abs(np.imag(grad)))
    maxExcursion = int(np.maximum(maxRealExcursion, maxImagExcursion))

    width = int(maxRealExcursion) * 2 + 1
    height = int(maxImagExcursion) * 2 + 1
    return maxExcursion, width, height


# def flat_size(width:int, height:int) -> int:
#     return width * height


def make_flat_hist(grad:np.ndarray, width:int, height:int) -> np.ndarray:
    hist = np.zeros((width, height), dtype=int)
    xoff = int((width-1)/2)
    yoff = int((height-1)/2)

    for pixel in grad:
        x = int(np.floor(np.real(pixel)))
        y = int(np.floor(np.imag(pixel)))
        hist[x + xoff, y + yoff] += 1

    return hist


def render_suppressed(hist:np.ndarray):
    def supScale(x, y, centx, centy, maxDist):
        xdist = x - centx
        ydist = y - centy
        hyp = np.hypot(xdist, ydist)
        return (hyp / maxDist)

    suppressed = np.zeros(hist.shape, dtype=np.float32)
    maxSuppressed = 0.0
    centx = (hist.shape[0] - 1) // 2
    centy = (hist.shape[1] - 1) // 2

    for x in range(hist.shape[0]):
        for y in range(hist.shape[1]):
            sscale = supScale(x, y, centx, centy, hist.shape[0]//2)
            suppressed[x,y] = hist[x, y] * sscale
            if suppressed[x,y] > maxSuppressed:
                maxSuppressed = suppressed[x,y]
    
    pixScale = 255.0 / maxSuppressed
    for x in range(hist.shape[0]):
        for y in range(hist.shape[1]):
            suppressed[x,y] *= pixScale

    plt.imshow(suppressed, cmap='gray_r')
    plt.savefig('histogram.png')
    plt.close()


def calc_delentropy(hist, grad):
    maxBinDelentropy = 0.0
    delentropy = 0.0
    numPixels = grad.shape[0] 

    for x in range(hist.shape[0]):
        for y in range(hist.shape[1]):
            if hist[x,y] != 0:
                p = hist[x,y] / numPixels
                binDelentropy = -1.0 * p * np.log2(p)

                if binDelentropy > maxBinDelentropy:
                    maxBinDelentropy = binDelentropy

                delentropy += binDelentropy
    print(f'Delentropy: {delentropy}')
    return delentropy

if __name__ == '__main__':
    from argparse import ArgumentParser
    ap = ArgumentParser()
    ap.add_argument('-i', '--img', required=True, help='Path to the image')
    ap.add_argument('-r', '--render', action='store_true', help='Render the histogram')
    ap.add_argument('--width', type=int, help='Width of the image. Keep original if not specified.')
    ap.add_argument('--height', type=int, help='Height of the image. Keep original if not specified.')
    args = ap.parse_args()
    
    img = cv2.imread(args.img, cv2.IMREAD_GRAYSCALE)
    img = crop_and_resize(img, args.width, args.height)
    #TODO: Currently only supports the defaultkernel
    grad = calc_grads(img, DEFAULTKERNEL)
    hist = calc_hist(grad)
    if args.render:
        render_suppressed(hist)
    delentropy = calc_delentropy(hist, grad)
    
