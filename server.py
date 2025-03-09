import numpy as np
import time
import pyvirtualcam
import threading
from queue import Queue, Empty
from mss import mss
import pygetwindow as gw
import signal
import sys
import hashlib
import cv2
import socket
from constants import *
from utils import bits_to_blocks, blocks_to_pixels, pixels_to_frame, pixels_to_bits, frame_to_pixels, detectframe, bits_to_bytes

frames_sent = 0
frames_received = 0
frames_processed = 0
ENDFRAME = np.zeros((AVIDEO_HEIGHT, AVIDEO_WIDTH, 3), np.uint8)
STARTFRAME = np.ones((AVIDEO_HEIGHT, AVIDEO_WIDTH, 3), np.uint8)*240

BORDER = np.zeros((PHEIGHT+2+BOTTOM_BUFFER, PWIDTH+2, 3), dtype=np.uint8)
BORDER[:, :] = (255, 255, 255)
BORDER = np.repeat(np.repeat(BORDER, PIXEL_SIZE, axis=0), PIXEL_SIZE, axis=1)

send_q = Queue(maxsize=1000)
recv_q = Queue()
upstream_dict = {}
exit_event = threading.Event()

def wrap(frame):  # Adds white border around frame for image detection
    tmp = BORDER[:]
    tmp[PIXEL_SIZE:-(1+BOTTOM_BUFFER)*PIXEL_SIZE, PIXEL_SIZE:-PIXEL_SIZE] = frame
    if BOTTOM_BUFFER:
        tmp[-BOTTOM_BUFFER*PIXEL_SIZE:] = 0
    return tmp

def process_bits(bits, q): # processes the bits into frames and adds them to a shared queue
    global frames_sent
    bits += [0]*(-len(bits)%BITS_PER_FRAME)
    bits = np.array(bits)

    blocks = bits_to_blocks(bits)
    pixels = blocks_to_pixels(blocks)
    
    frame_blocks = pixels.reshape(-1, NPIXELS, 3)
    frames = [pixels_to_frame(fb) for fb in frame_blocks]
    
    for frame in frames:
        q.put(frame)
        frames_sent += 1

def maintain_upstream_conn(conn_identifier: str, data: bytes):
    """Creates a connection to upstream server, sends data, and returns response"""
    try:
        upstream_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        upstream_sock.settimeout(UPSTREAM_TIMEOUT)
        upstream_sock.connect((SOCKS_SERVER_IP, SOCKS_SERVER_PORT))
        upstream_sock.sendall(data)

        recv_data = b''
        try:
            while len(recv_data) < MAX_SIZE:
                poll_data = upstream_sock.recv(MAX_SIZE)
                if not poll_data:
                    break
                recv_data += poll_data
        except socket.timeout:
            pass
        
        if len(recv_data) > 0:
            recv_data = recv_data.rstrip(b"\x00")
            # Convert response to frames and queue them
            combined = conn_identifier.encode() + HEADER_SEPARATOR + recv_data
            bits = list(map(int, bin(int.from_bytes(combined, byteorder='big'))[2:]))
            bits = ([0]*((-len(bits))%8)) + bits
            process_bits(bits, send_q)
            
    except Exception as e:
        print(f"Error sending to upstream for {conn_identifier}: {e}")
    finally:
        upstream_sock.close()

def process_frames():
    global frames_processed
    prev = -1 # Prevent repeated frames
    capturing = 0
        
    while not exit_event.is_set():
        try:
            frame = recv_q.get(timeout=5)
            frame = detectframe(frame)[:,:,:3]
            pixels = frame_to_pixels(frame)
            bits = pixels_to_bits(pixels)
            byts = bits_to_bytes(bits)
            bits_sum = np.sum(bits)
            
            if prev == byts:
                continue
            
            if bits_sum > BITS_PER_FRAME - (BITS_PER_FRAME // 1000):
                print("Start frame detected. Starting capture.")
                capturing = 1
                prev = byts
                continue

            if bits_sum < BITS_PER_FRAME // 1000:
                print("End frame detected. Stopping capture.")
                exit_event.set()
                capturing = 0
                break
                
            if capturing and prev != byts:
                try:
                    conn_identifier = byts[:byts.find(HEADER_SEPARATOR)].decode()
                    data = byts[byts.find(HEADER_SEPARATOR)+len(HEADER_SEPARATOR):]
                    
                    # Create thread to handle upstream connection
                    t = threading.Thread(target=maintain_upstream_conn, args=(conn_identifier, data))
                    t.start()
                    
                    frames_processed += 1
                    prev = byts

                    if frames_processed % 100 == 0:
                        print(f"{frames_processed} frames processed.")
                except:
                    continue # invalid frame
            
        except Empty:
            continue
    print("Closing process_frames")

def screenshot():
    global frames_received
    with mss() as sct:
        prev = None
        while not exit_event.is_set():
            window = gw.getWindowsWithTitle("Zoom Meeting")[0]
            monitor = {"top": window.top, "left": window.left, "width": window.width, "height": window.height}
            screenshot = np.array(sct.grab(monitor))
            
            if prev is None or not np.all(screenshot == prev):
                recv_q.put(screenshot)
                frames_received += 1
                if frames_received % 100 == 0:
                    print(f"{frames_received} frames received.")
                prev = screenshot
        print("Closing screenshot")

def output_to_stream():
    with pyvirtualcam.Camera(width=VIDEO_WIDTH, height=VIDEO_HEIGHT, fps=FPS, fmt=pyvirtualcam.PixelFormat.BGR) as cam:
        print(f'Using virtual camera: {cam.device}')
        while not exit_event.is_set():
            try:
                frame = send_q.get()
                cam.send(wrap(frame))
                if cv2.waitKey(1) == ord('q'):
                   cv2.destroyAllWindows()
                   exit_event.set()
                   break
                cam.sleep_until_next_frame()
                if (np.all(frame == ENDFRAME)):
                    print("End frame received. Exiting...")
                    break
                
            except Empty:
                print("No more frames in queue. Exiting...")
                break
    print("Closing output_to_stream")

def signal_handler(signum, frame):
    print("Sending end frame")
    send_q.put(ENDFRAME)
    print("End frame sent")
    cv2.destroyAllWindows()
    exit_event.set()

if __name__ == "__main__":
    window = gw.getWindowsWithTitle("Zoom Meeting")[0]
    window.activate()
    signal.signal(signal.SIGINT, signal_handler)

    t_output_src_frame = threading.Thread(target=output_to_stream, daemon=True)
    t_get_frames = threading.Thread(target=screenshot, daemon=True)
    t_process_incoming = threading.Thread(target=process_frames, daemon=True)
    
    t_get_frames.start()
    t_process_incoming.start()
    t_output_src_frame.start()

    send_q.put(STARTFRAME)

    while not exit_event.is_set() and (t_output_src_frame.is_alive() or t_get_frames.is_alive() or t_process_incoming.is_alive()):
        time.sleep(1)
        
    print(f"Execution stopped. {frames_sent} frames sent, {frames_received} frames received, {frames_processed} frames processed")
    t_output_src_frame.join()
    t_get_frames.join()
    t_process_incoming.join() 
