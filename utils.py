from constants import *
import numpy as np
import cv2


def pixels_to_bits(blocks):
    """
    Converts list of pixel RGB values back to list of bits
    """
    blocks = blocks.flatten()
    blocks = blocks.astype('uint8')[:, np.newaxis]

    nums = np.unpackbits(blocks, axis=1, count=BITS_PER_COLOR)
    #print(blocks[0], nums[0])
    return nums.flatten()



def frame_to_pixels(frame):
    """
    Averages the middle 4 pixels of each pixel block, following CovertCast
    Returns a list of pixel RGB values
    """
    Z = frame.reshape(PHEIGHT,PIXEL_SIZE,PWIDTH,PIXEL_SIZE,3)

    ZZ = Z[:, PIXEL_SIZE//2-2:PIXEL_SIZE//2+2, :, PIXEL_SIZE//2-2:PIXEL_SIZE//2+2, :].mean(axis=(1,3)) # Z.mean(axis=(1,3)) #
    
    return ZZ.reshape(-1, 3)
 

def detectframe(frame):
    """Detects white border of frame, then crops and resizes frame"""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    NROWS = gray.shape[0]
    NCOLS = frame.shape[1]

    #apply threshold
    thresh = cv2.threshold(gray, 250, 255, cv2.THRESH_BINARY)[1]
    cols = thresh[NROWS//2, :]
    rows = thresh[:, NCOLS//2]
    
    L, R = cols[:].argmax(), NCOLS - cols[::-1].argmax()
    U, D = rows[:].argmax(), NROWS - rows[::-1].argmax()
    
    cropped = frame[U:D, L:R]
    resized = cv2.resize(cropped, (VIDEO_WIDTH, VIDEO_HEIGHT-BOTTOM_BUFFER*PIXEL_SIZE))
    unwrapped = resized[PIXEL_SIZE:-PIXEL_SIZE, PIXEL_SIZE:-PIXEL_SIZE]
    return unwrapped

def bits_to_bytes(bits: np.array):
    """Seems to work"""
    bits = bits[:len(bits)//8*8]
    bytesstr = bits.reshape(-1, 8)
    return bytes(np.packbits(bytesstr, axis=-1).flatten())

def bits_to_blocks(bits):
    """
    (# of bits) --> ( {0, 64, 128, 192} values )
    0, 0 --> 0
    0, 1 --> 64
    1, 0 --> 128
    1, 1 --> 192
    """
    
    split = bits.reshape(-1, BITS_PER_COLOR)
    return np.packbits(split, axis=-1).flatten()


def blocks_to_pixels(arr):
    """
    Groups the values into RGB blocks, adds 32 for more accurate decoding
    """
    return arr.reshape(-1, 3) + (1<<(8-BITS_PER_COLOR-1))


def pixels_to_frame(pixels):  
    """
    (# of pixel values , 3) --> (frame_height, frame_width, 3)
    expands each pixel to a PIXEL_SIZE * PIXEL_SIZE block
    """
    pixels2 = pixels.reshape(PHEIGHT, PWIDTH, 3)
    ret = np.repeat(np.repeat(pixels2, PIXEL_SIZE, axis=0), PIXEL_SIZE, axis=1)
    return ret


