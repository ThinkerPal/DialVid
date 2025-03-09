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
frames_generated = 0
frames_received = 0
frames_processed = 0
ENDFRAME = np.zeros((AVIDEO_HEIGHT, AVIDEO_WIDTH, 3), np.uint8)
STARTFRAME = np.ones((AVIDEO_HEIGHT, AVIDEO_WIDTH, 3), np.uint8)*240


BORDER = np.zeros((PHEIGHT+2+BOTTOM_BUFFER, PWIDTH+2, 3), dtype=np.uint8)
BORDER[:, :] = (255, 255, 255)
BORDER = np.repeat(np.repeat(BORDER, PIXEL_SIZE, axis=0), PIXEL_SIZE, axis=1)

send_q = Queue(maxsize=1000)
recv_q = Queue()
net_dict = {}
upstream_dict = {}
exit_event = threading.Event()

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
        # print("New Frame")

 
def maintain_upstream_conn(conn_identifier: str):
    upstream_dict[conn_identifier]["keep_alive"] = threading.Event()
    while not exit_event.is_set() and not upstream_dict[conn_identifier]["keep_alive"].is_set():
        try:
            data = upstream_dict[conn_identifier]["send_q"].get()
            upstream_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            upstream_sock.settimeout(UPSTREAM_TIMEOUT)
            upstream_sock.connect((SOCKS_SERVER_IP, SOCKS_SERVER_PORT))
            upstream_sock.sendall(data)

            recv_data = b''
            try:
                while len(recv_data) < MAX_SIZE:
                    # print("polling")
                    poll_data = upstream_sock.recv(MAX_SIZE)
                    if not poll_data:
                        break
                    print(f"poll data: {poll_data}")
                    recv_data += poll_data
            except socket.timeout:
                if len(recv_data) > 0:
                    recv_data = recv_data.rstrip(b"\x00")
                    print(f"Received from upstream {conn_identifier}: {recv_data}")
                    if "last_data" not in upstream_dict[conn_identifier] or upstream_dict[conn_identifier]["last_data"] != recv_data:
                        net_dict[conn_identifier]["conn"].sendall(recv_data)
                        upstream_dict[conn_identifier]["last_data"] = recv_data
                    
                    print(f"Received from upstream {conn_identifier}: {recv_data}")
        except Exception as e:
            print("client connection dict:")
            print(net_dict)
            print("upstream connection dict:")
            print(upstream_dict)
            print(f"Error sending to upstream for {conn_identifier}: {e}")
            upstream_sock.close()
            upstream_dict[conn_identifier]["keep_alive"].set()
    print(f"Closed upstream_dict {conn_identifier}")
    upstream_dict.pop(conn_identifier)


def process_frames():
    global frames_processed
    prev = -1 # Hacky way to prevent repeated frames
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
                
            if capturing and prev != byts:
                try:
                    conn_identifier = byts[:byts.find(HEADER_SEPARATOR)].decode()
                    data = byts[byts.find(HEADER_SEPARATOR)+len(HEADER_SEPARATOR):]
                    try:
                        if conn_identifier in upstream_dict:
                            print(f"found upstream sock for {conn_identifier}")
                            upstream_dict[conn_identifier]["send_q"].put(data)
                        else:
                            upstream_dict[conn_identifier] = { 
                                "thread": threading.Thread(target=maintain_upstream_conn, args=(conn_identifier,)),
                                "send_q": Queue()
                            }
                            upstream_dict[conn_identifier]["thread"].start()
                            upstream_dict[conn_identifier]["send_q"].put(data)
                    except:
                        print(f"Error sending to {conn_identifier}")
                    frames_processed += 1
                    prev = byts

                    if frames_processed % 100 == 0:
                        print(f"{frames_processed} frames processed.")
                except:
                    continue # invalid frame
            
        except Empty:
            # print("No more frames in recv_queue.") 
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
    global frames_sent
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
                frames_sent += 1
                if frames_sent % 100 == 0:
                    print(f"{frames_sent} frames sent.")
                if (np.all(frame == ENDFRAME)):
                    print("End frame received. Exiting...")
                    break
                
            except Empty:
                print("No more frames in queue. Exiting...")
                break
    print("Closing output_to_stream")

def process_recv_data(conn: socket.socket, conn_identifier: str):
    print(f"Connection thread created at {conn_identifier}")
    keep_alive = True
    while not exit_event.is_set() and keep_alive:
        try:
            header = str(conn_identifier).encode() + HEADER_SEPARATOR
            buf = b'\x00\x00\x00\x00'
            while buf[-4:] != b'\r\n\r\n': # all http requests end with 2 \r\n (as per RFC) 
                buf = conn.recv(1024)
            if buf == b'':
                continue
            print(f"Client {conn_identifier} sent: {buf}")
            combined = header + buf
            bits = list(map(int, bin(int.from_bytes(combined, byteorder='big'))[2:]))
            bits = ([0]*((-len(bits))%8)) + bits 
            # print(f"{conn_identifier}: sending")
            process_bits(bits, send_q)
            # print(f"{conn_identifier}: sent!")
        except ConnectionError:
            keep_alive = False
    conn.close()
    print(f"Closed net dict{conn_identifier}")
    if conn_identifier in net_dict:
        net_dict.pop(conn_identifier)
        upstream_dict[conn_identifier]["keep_alive"].set()

        
def recv_data_sock():
    listen_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    listen_sock.bind((CLIENT_LISTEN_SOCK_IP, CLIENT_LISTEN_SOCK_PORT))
    listen_sock.listen(socket.SOMAXCONN)
    
    while not exit_event.is_set():
        conn, addr = listen_sock.accept()
        if exit_event.is_set():
            conn.close()
            break
        try:
            conn_identifier = hashlib.sha1(str(addr[0]+str(addr[1])).encode()).hexdigest()
            # print(f"Conn identifier: {conn_identifier}")
            t = threading.Thread(target=process_recv_data, args=(conn, conn_identifier))
            t.start()
            # print("Thread Started")
            net_dict[conn_identifier] = {
                "conn": conn,
                "addr": addr,
                "thread": t,
                "send_q": Queue()
            }
        except Exception as e:
            print(f"Error in connection: {e}")
            conn.close()
            break
    
    print("Closing recv_data_sock")


def signal_handler(signum, frame):
    print("Sending end frame")
    send_q.put(ENDFRAME)
    print("End frame sent")
    cv2.destroyAllWindows()
    exit_event.set()
    socket.socket().connect((CLIENT_LISTEN_SOCK_IP, CLIENT_LISTEN_SOCK_PORT))

if __name__ == "__main__":

    window = gw.getWindowsWithTitle("Zoom Meeting")[0]
    window.activate()
    signal.signal(signal.SIGINT, signal_handler)

    t_get_src_data = threading.Thread(target=recv_data_sock, daemon=True)
    t_output_src_frame = threading.Thread(target=output_to_stream, daemon=True)
    t_get_frames = threading.Thread(target=screenshot, daemon=True)
    t_process_incoming = threading.Thread(target=process_frames, daemon=True)
    t_get_frames.start()
    t_process_incoming.start()
    t_get_src_data.start()
    t_output_src_frame.start()

    send_q.put(STARTFRAME)

    while not exit_event.is_set() and (t_get_src_data.is_alive() or t_output_src_frame.is_alive() or t_get_frames.is_alive() or t_process_incoming.is_alive()):
        time.sleep(1)
        
    print(f"Execution stopped. {frames_generated} frames written, {frames_sent} frames sent, {frames_received} frames received, {frames_processed} frames processed")
    t_get_src_data.join()
    t_output_src_frame.join()
    t_get_frames.join()
    t_process_incoming.join()
