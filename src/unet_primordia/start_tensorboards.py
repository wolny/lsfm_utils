import argparse
import atexit
import html
import io
import logging
import shutil
import socket
import subprocess
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, HTTPServer

import os

logging.basicConfig(level=logging.INFO)


def shutdown(process_list):
    """Terminate the given process instances
    Args:
        process_list (list): list of subprocess.Popen instances
    """
    for p in process_list:
        logging.info('Terminating Tensorboard process...')
        p.terminate()


def start_tensorboard_servers(checkpoints_dir, start_port):
    """Search for directories containing the 'logs/' in the given
    'checkpoints_dir' and start a separate Tensorboard servers for each
    of the found 'logs/' directory.
    Tensorboard servers will be exposed on ports 'start_port', 'start_port+1`,...

    Args:
        checkpoints_dir (string): parent directory to look for Tensorboard logdirs
        start_port (int): port to serve the 1st Tensorboard instance on

    Returns:
        A tuple of model_name_to_port mapping and a list of child processes
        (one for each Tensorboard instance)
    """
    model_port_map = {}
    child_processes = []
    for d in os.listdir(checkpoints_dir):
        dir_path = os.path.join(checkpoints_dir, d, 'logs')
        if os.path.isdir(dir_path):
            logging.info(
                f"Starting Tensorboard instance: 'tensorboard --logdir {dir_path} --port {start_port}'")
            # update the model_port_map
            model_port_map[d] = start_port

            # start new Tensorboard server in a child process
            proc = subprocess.Popen(
                ['tensorboard', '--logdir', dir_path, '--port',
                 str(start_port)])
            child_processes.append(proc)

            start_port += 1

    return model_port_map, child_processes


def define_request_handler(model_port_map):
    def get_html():
        r = []
        title = 'Tensorboard servers'
        r.append('<!DOCTYPE HTML PUBLIC "-//W3C//DTD HTML 4.01//EN" '
                 '"http://www.w3.org/TR/html4/strict.dtd">')
        r.append('<html>\n<head>')
        r.append('<meta http-equiv="Content-Type" '
                 'content="text/html; charset=utf-8">')
        r.append(f'<title>{title}</title>\n</head>')
        r.append(f'<body>\n<h1>{title}</h1>')
        r.append('<hr>\n<ul>')

        hostname = socket.gethostname()
        for name, port in model_port_map.items():
            display_name = html.escape(name, quote=False)
            tb_link = f'http://{hostname}:{port}/'
            r.append(f'<li><a href="{tb_link}">{display_name}</a></li>')
        r.append('</ul>\n<hr>\n</body>\n</html>\n')

        encoded = '\n'.join(r).encode()
        f = io.BytesIO()
        f.write(encoded)
        f.seek(0)
        return f

    class GetRequestHandler(BaseHTTPRequestHandler):
        def do_GET(self):
            f = get_html()
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-type", "text/html; charset=utf-8")
            self.end_headers()

            # Display the the html encoded page to the client
            if f:
                try:
                    shutil.copyfileobj(f, self.wfile)
                finally:
                    f.close()

    return GetRequestHandler


def expose_master_server(port, model_port_map):
    logging.info(f'Running master server on port: {port}...')
    server_address = ('', port)
    handler_class = define_request_handler(model_port_map)
    httpd = HTTPServer(server_address, handler_class)
    httpd.serve_forever()


def main():
    parser = argparse.ArgumentParser(description='Tensorboard MasterServer')
    parser.add_argument('--checkpoints-dir', required=True,
                        help='Path to the checkpoints directory')
    parser.add_argument('--port', default=8666, type=int,
                        help='What port to server the master server on')
    parser.add_argument('--tb-start-port', default=7001, type=int,
                        help='Start port assigned to your first Tensorboard instance')

    args = parser.parse_args()

    model_port_map, tb_processes = start_tensorboard_servers(
        args.checkpoints_dir, args.tb_start_port)

    atexit.register(shutdown, tb_processes)

    expose_master_server(args.port, model_port_map)


if __name__ == '__main__':
    main()
