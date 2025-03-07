import numpy as np
import time
import pyvirtualcam
import threading
from queue import Queue, Empty
import signal
import sys
import cv2


BITS_PER_COLOR = 2
PIXEL_SIZE_BITS = 3

PIXEL_SIZE = 1<<PIXEL_SIZE_BITS
VIDEO_HEIGHT = 720
VIDEO_WIDTH = 1280
FPS = 2
BOTTOM_BUFFER = 0  # set to 22 if want to avoid errors when Zoom HUD shows up
AVIDEO_HEIGHT = VIDEO_HEIGHT-(2+BOTTOM_BUFFER)*PIXEL_SIZE
AVIDEO_WIDTH = VIDEO_WIDTH-2*PIXEL_SIZE
PWIDTH = AVIDEO_WIDTH//PIXEL_SIZE
PHEIGHT = AVIDEO_HEIGHT//PIXEL_SIZE
NPIXELS = PWIDTH*PHEIGHT
BITS_PER_FRAME = NPIXELS*3*BITS_PER_COLOR
MAX_SIZE = BITS_PER_FRAME//8
COLORWIDTH = 256 // (1<<BITS_PER_COLOR)

frames_sent = 0
frames_generated = 0
ENDFRAME = np.zeros((AVIDEO_HEIGHT, AVIDEO_WIDTH, 3), np.uint8)
STARTFRAME = np.ones((AVIDEO_HEIGHT, AVIDEO_WIDTH, 3), np.uint8)*240


BORDER = np.zeros((PHEIGHT+2+BOTTOM_BUFFER, PWIDTH+2, 3), dtype=np.uint8)
BORDER[:, :] = (255, 255, 255)
BORDER = np.repeat(np.repeat(BORDER, PIXEL_SIZE, axis=0), PIXEL_SIZE, axis=1)



q = Queue()
exit_event = threading.Event()
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

def wrap(frame):  # Adds white border around frame for image detection
    tmp = BORDER[:]
    tmp[PIXEL_SIZE:-(1+BOTTOM_BUFFER)*PIXEL_SIZE, PIXEL_SIZE:-PIXEL_SIZE] = frame
    if BOTTOM_BUFFER:
        tmp[-BOTTOM_BUFFER*PIXEL_SIZE:] = 0
    return tmp

def process_bits(bits, q): # processes the bits into frames and adds them to a shared queue
    global frames_generated
    bits += [0]*(-len(bits)%BITS_PER_FRAME)
    bits = np.array(bits)

    blocks = bits_to_blocks(bits)
    pixels = blocks_to_pixels(blocks)
    
    frame_blocks = pixels.reshape(-1, NPIXELS, 3)
    #print(bits[0:16], blocks[0:6], pixels[0:6])
    frames = [pixels_to_frame(fb) for fb in frame_blocks]
    

    for frame in frames:
        q.put(frame)
        frames_generated += 1
    
        #if frames_generated % 100 == 0:
        #    print(f"{frames_generated} frames written.")

    

def output_to_stream():
    global frames_sent
    with pyvirtualcam.Camera(width=VIDEO_WIDTH, height=VIDEO_HEIGHT, fps=FPS, fmt=pyvirtualcam.PixelFormat.BGR) as cam:
        print(f'Using virtual camera: {cam.device}')
        while not exit_event.is_set():
            try:
                frame = q.get(timeout=5)
                
                #frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                #print(frame[0,0], frame[0, PIXEL_SIZE])
                cam.send(wrap(frame))
                #cv2.imshow('sending', frame)
                #if cv2.waitKey(1) == ord('q'):
                #    cv2.destroyAllWindows()
                #    break
                cam.sleep_until_next_frame()
                frames_sent += 1
                if frames_sent % 100 == 0:
                    print(f"{frames_sent} frames sent.")
                if (np.all(frame == ENDFRAME)):
                    print("End frame received. Exiting...")
                    break
                
            except Empty:
                print("No more frames in queue. Exiting...")
                break


        

def get_data(filename):
    with open(filename, "rb") as f:
        q.put(STARTFRAME)
        while not exit_event.is_set():
            buf = f.read(MAX_SIZE)
            if not buf:
                print("Finished reading file. Exiting...")
                q.put(ENDFRAME)
                break
            bits = list(map(int, bin(int.from_bytes(buf, byteorder='big'))[2:]))
            bits = ([0]*((-len(bits))%8)) + bits
            
            #print(buf[0], bits[:8])
            process_bits(bits, q)
            

def signal_handler(signum, frame):
    exit_event.set()

FILENAME = -1
if __name__ == "__main__":
    if len(sys.argv) <= 1:
        print("Usage: sender.py <filename>")
        exit(0)

    FILENAME = sys.argv[1]
    signal.signal(signal.SIGINT, signal_handler)

    x = threading.Thread(target=get_data, args=(FILENAME,), daemon=True)
    y = threading.Thread(target=output_to_stream, daemon=True)
    x.start()
    y.start()

    while not exit_event.is_set() and (x.is_alive() or y.is_alive()):
        time.sleep(1)
        
    print(f"Execution stopped. {frames_generated} frames written, {frames_sent} frames sent")
    x.join()
    y.join()




"""

print("reading video")
out = video_to_bits('output_video.avi')

for i in range(3, 8):
    k = 10**i
    print(i, np.all(out[:k]==bits[:k]))
"""

