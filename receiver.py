import cv2
import numpy as np
from mss import mss
import pygetwindow as gw
import signal
import threading
import time
from queue import Queue, Empty

q = Queue()
exit_event = threading.Event()

BITS_PER_COLOR = 2
PIXEL_SIZE_BITS = 3

PIXEL_SIZE = 1<<PIXEL_SIZE_BITS
VIDEO_HEIGHT = 720
VIDEO_WIDTH = 1280
BOTTOM_BUFFER = 0
AVIDEO_HEIGHT = VIDEO_HEIGHT-(2+BOTTOM_BUFFER)*PIXEL_SIZE
AVIDEO_WIDTH = VIDEO_WIDTH-2*PIXEL_SIZE
PWIDTH = AVIDEO_WIDTH//PIXEL_SIZE
PHEIGHT = AVIDEO_HEIGHT//PIXEL_SIZE
NPIXELS = PWIDTH*PHEIGHT
BITS_PER_FRAME = NPIXELS*3*BITS_PER_COLOR
MAX_SIZE = BITS_PER_FRAME // 8
COLORWIDTH = 256 // (1<<BITS_PER_COLOR)

frames_received = 0
frames_processed = 0
    
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

def bits_to_bytes(bits):
    """Seems to work"""
    bits = bits[:len(bits)//8*8]
    bytesstr = bits.reshape(-1, 8)
    return bytes(np.packbits(bytesstr, axis=-1).flatten())


def process_frames():
    global frames_processed
    prev = -1 # Hacky way to prevent repeated frames
    capturing = 0
    with open("result.bin", "wb") as f:
        
        while not exit_event.is_set():
            try:
                
                frame = q.get(timeout=5)
                frame = detectframe(frame)[:,:,:3]
                #frame = frame[:,::-1,:]
                #print(frame.shape)
                #frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                #frame = frame[:,::-1,:]
                pixels = frame_to_pixels(frame)
                bits = pixels_to_bits(pixels)
                byts = bits_to_bytes(bits)
                if prev == byts:
                        continue
                    
                bits_sum = np.sum(bits)
                if bits_sum > BITS_PER_FRAME - (BITS_PER_FRAME // 1000):
                    print("Start frame detected. Starting capture.")
                    # Likely to be startframe
                    capturing = 1
                    prev = byts
                    continue

                if bits_sum < BITS_PER_FRAME // 1000:
                    print("End frame detected. Stopping capture.")
                    # Likely to be endframe
                    exit_event.set()
                    capturing = 0

                    break
                    
                if capturing:
                    f.write(byts)
                    #cv2.imshow('Computer Vision', frame)
                    #print(pixels[0],pixels[1],bits[:8], byts[:16])
                    frames_processed += 1
    
                    if frames_processed % 100 == 0:
                        print(f"{frames_processed} frames processed.")
                prev = byts
                
            except Empty:
                print("No more frames in queue. Exiting...")
                exit_event.set()
                break


def screenshot():
    global frames_received
    with mss() as sct:
            prev = None
            while not exit_event.is_set():
                window = gw.getWindowsWithTitle("Zoom Meeting")[0]
                monitor = {"top": window.top, "left": window.left, "width": window.width, "height": window.height}
                screenshot = np.array(sct.grab(monitor))
                
                if prev is None or not np.all(screenshot == prev):
                    q.put(screenshot)
                    frames_received += 1
                    if frames_received % 100 == 0:
                        print(f"{frames_received} frames received.")
                    prev = screenshot
                
def signal_handler(signum, frame):
    print("Exiting...")
    exit_event.set()




if __name__ == "__main__":
    window = gw.getWindowsWithTitle("Zoom Meeting")[0]
    window.activate()
    signal.signal(signal.SIGINT, signal_handler)

    x = threading.Thread(target=screenshot, daemon=True)
    y = threading.Thread(target=process_frames, daemon=True)
    x.start()
    y.start()

    while not exit_event.is_set() and (x.is_alive() or y.is_alive()):
        time.sleep(1)
        
    print(f"Execution stopped. {frames_received} frames received, {frames_processed} frames processed")
    x.join()
    y.join()



    


print('Done.')
