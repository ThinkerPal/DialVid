import cv2
import numpy as np
from mss import mss
import pygetwindow as gw
import signal
import threading
import time
from queue import Queue, Empty
from constants import *
from utils import pixels_to_bits, frame_to_pixels, detectframe, bits_to_bytes

q = Queue()
netq = Queue()
exit_event = threading.Event()

frames_received = 0
frames_processed = 0
    

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
    cv2.destroyAllWindows()
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
