from constants import *
import socket
import threading
import signal

exit_event = threading.Event()

def signal_handler(signum, frame):
    print("Shutting down proxy...")
    exit_event.set()

def handle_client(client_socket):
    try:
        request = client_socket.recv(MAX_SIZE)
        if not request:
            client_socket.close()
            return
        request = request.rstrip(b'\x00')
        # print(request)

        req_line = request.split(b'\n')[0]
        print(req_line)
        method, url, version = req_line.split()
        method = method.decode()

        if method == 'CONNECT':
            # Handle HTTPS (CONNECT) requests
            host, port = url.decode().split(':')
            port = int(port)

            remote_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            remote_socket.settimeout(UPSTREAM_TIMEOUT)
            try:
                remote_socket.connect((host, port))
            except socket.gaierror:
                print(f"Failed to resolve host: {host}") # server dns blocks this host (eg. adblocking via pihole)
                client_socket.send(b'HTTP/1.1 502 Bad Gateway\r\n\r\n')
                client_socket.close()
                remote_socket.close()
                return

            client_socket.send(b'HTTP/1.1 200 Connection Established\r\n\r\n')

            def forward(src, dst):
                try:
                    while not exit_event.is_set():
                        try:
                            data = src.recv(MAX_SIZE)
                            if not data:
                                break
                            dst.send(data)
                        except socket.timeout:
                            # If we timeout, check if there's any pending data
                            src.setblocking(False)
                            try:
                                data = src.recv(MAX_SIZE)
                                if data:
                                    dst.send(data)
                            except (socket.error, BlockingIOError):
                                # No more data available right now
                                pass
                            finally:
                                src.setblocking(True)
                                src.settimeout(UPSTREAM_TIMEOUT)
                        except (socket.error, ConnectionError):
                            break
                finally:
                    try:
                        src.shutdown(socket.SHUT_RDWR)
                    except:
                        pass
                    try:
                        dst.shutdown(socket.SHUT_RDWR)
                    except:
                        pass
                    src.close()
                    dst.close()

            client_to_remote = threading.Thread(target=forward, args=(client_socket, remote_socket))
            remote_to_client = threading.Thread(target=forward, args=(remote_socket, client_socket))

            client_to_remote.start()
            remote_to_client.start()

            # Wait for either thread to finish
            client_to_remote.join()
            remote_to_client.join()

        else:
            # Handle regular HTTP requests
            headers = request.split(b'\r\n\r\n')[0].split(b'\r\n')
            host = None
            port = 80

            # Extract host from headers
            for header in headers[1:]:
                if header.lower().startswith(b'host:'):
                    host_header = header.split(b':')[1].strip().decode()
                    if ':' in host_header:
                        host, port_str = host_header.split(':')
                        port = int(port_str)
                    else:
                        host = host_header
                    break

            if not host:
                # Try to extract host from URL if absolute URL is provided
                if url.startswith(b'http://'):
                    url_parts = url.decode().replace('http://', '').split('/')
                    host_part = url_parts[0]
                    if ':' in host_part:
                        host, port_str = host_part.split(':')
                        port = int(port_str)
                    else:
                        host = host_part
                else:
                    client_socket.send(b'HTTP/1.1 400 Bad Request\r\n\r\nNo Host header found')
                    client_socket.close()
                    return

            try:
                # Connect to remote server
                remote_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                remote_socket.settimeout(UPSTREAM_TIMEOUT)
                remote_socket.connect((host, port))

                # Forward the original request
                remote_socket.send(request)

                # Get the response and send it back to client
                while True:
                    try:
                        response = remote_socket.recv(MAX_SIZE)
                        if not response:
                            break
                        client_socket.send(response)
                    except socket.timeout:
                        # If we timeout, check if there's any pending data
                        remote_socket.setblocking(False)
                        try:
                            response = remote_socket.recv(MAX_SIZE)
                            if response:
                                client_socket.send(response)
                        except (socket.error, BlockingIOError):
                            # No more data available right now, break the loop
                            break
                        finally:
                            remote_socket.setblocking(True)
                            remote_socket.settimeout(UPSTREAM_TIMEOUT)
                    except (socket.error, ConnectionError):
                        break

            except socket.gaierror:
                print(f"Failed to resolve host: {host}")
                client_socket.send(b'HTTP/1.1 502 Bad Gateway\r\n\r\n')
            finally:
                try:
                    remote_socket.shutdown(socket.SHUT_RDWR)
                except:
                    pass
                finally:
                    remote_socket.close()
                try:
                    client_socket.shutdown(socket.SHUT_RDWR)
                except:
                    pass
                finally:
                    client_socket.close()

    except Exception as e:
        print(f"Error handling connection: {e}")
        try:
            client_socket.shutdown(socket.SHUT_RDWR)
        except:
            pass
        client_socket.close()

def main():
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind((SOCKS_SERVER_IP, SOCKS_SERVER_PORT))
    server.listen(socket.SOMAXCONN)


    signal.signal(signal.SIGINT, signal_handler)

    while not exit_event.is_set():
        try:
            client_socket, address = server.accept()
            if exit_event.is_set():
                client_socket.close()
                break

            # print(f"Connected: {address}")
            client_handler = threading.Thread(target=handle_client, args=(client_socket,))
            client_handler.daemon = True
            client_handler.start()

        except Exception as e:
            print(f"Error: {e}")
            continue

    server.close()
    print("Proxy shutdown complete")

if __name__ == "__main__":
    main()

