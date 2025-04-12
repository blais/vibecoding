#!/usr/bin/env python3
# server.py (Conceptual Example)

import os
import argparse # Added for command-line arguments
import threading
from flask import Flask, render_template_string, abort, url_for, redirect # Added url_for, redirect
from flask_socketio import SocketIO, emit
import nbformat
from nbconvert import HTMLExporter
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your_secret_key!' # Replace with a real secret key
socketio = SocketIO(app, async_mode='threading') # Use threading or gevent/eventlet

# --- Configuration ---
# NOTEBOOK_DIR will be set via command-line argument
ALLOWED_EXTENSIONS = {'.ipynb'}
# ---------------------

# Global variable to store the notebook directory from args
notebook_dir_global = None

# Dictionary to keep track of monitored files and associated clients
# { 'path/to/notebook.ipynb': {'clients': {sid1, sid2}, 'observer': observer_instance} }
monitored_files = {}
monitor_lock = threading.Lock()

# Removed basedir argument, will use notebook_dir_global
def is_safe_path(path):
    """Ensure the path is within the allowed directory."""
    if not notebook_dir_global:
        # Handle case where global is not set (shouldn't happen if main runs first)
        print("Error: Notebook directory not configured.")
        return False
    abs_path = os.path.abspath(os.path.join(notebook_dir_global, path))
    # Ensure the common path is the configured notebook directory
    return os.path.commonpath([notebook_dir_global, abs_path]) == notebook_dir_global and \
           os.path.splitext(path)[1].lower() in ALLOWED_EXTENSIONS

class NotebookChangeHandler(FileSystemEventHandler):
    def __init__(self, notebook_abs_path):
        self.notebook_abs_path = notebook_abs_path
        print(f"[*] Monitoring changes for: {self.notebook_abs_path}")

    def on_modified(self, event):
        if event.src_path == self.notebook_abs_path:
            print(f"[*] Detected modification in: {self.notebook_abs_path}")
            # Notify relevant clients via SocketIO
            with monitor_lock:
                if self.notebook_abs_path in monitored_files:
                    clients_to_notify = monitored_files[self.notebook_abs_path]['clients']
                    print(f"[*] Notifying clients: {clients_to_notify}")
                    for sid in list(clients_to_notify): # Iterate over a copy
                         socketio.emit('refresh', {'path': self.notebook_abs_path}, room=sid)

def start_monitoring(notebook_abs_path):
    with monitor_lock:
        if notebook_abs_path not in monitored_files:
            print(f"[*] Starting observer for {notebook_abs_path}")
            event_handler = NotebookChangeHandler(notebook_abs_path)
            observer = Observer()
            # Watch the directory containing the file, less resource intensive than watching individual files on some OS
            watch_dir = os.path.dirname(notebook_abs_path)
            observer.schedule(event_handler, watch_dir, recursive=False)
            observer.start()
            monitored_files[notebook_abs_path] = {'clients': set(), 'observer': observer}
        # Note: Does not handle stopping observers when no clients are connected - requires more logic

def stop_monitoring(notebook_abs_path):
     with monitor_lock:
        if notebook_abs_path in monitored_files and not monitored_files[notebook_abs_path]['clients']:
            print(f"[*] Stopping observer for {notebook_abs_path}")
            observer = monitored_files[notebook_abs_path]['observer']
            observer.stop()
            observer.join()
            del monitored_files[notebook_abs_path]


@app.route('/')
def index():
    """Serves the index page listing available notebooks."""
    if not notebook_dir_global:
        abort(500, "Server configuration error: Notebook directory not set.")

    notebooks = []
    try:
        for item in os.listdir(notebook_dir_global):
            item_path = os.path.join(notebook_dir_global, item)
            if os.path.isfile(item_path) and item.lower().endswith('.ipynb'):
                # Use the relative path for the URL
                relative_path = item
                view_url = url_for('view_notebook', notebook_rel_path=relative_path)
                notebooks.append({'name': item, 'url': view_url})
    except OSError as e:
        print(f"Error reading notebook directory {notebook_dir_global}: {e}")
        abort(500, "Error accessing notebook directory.")

    # Simple HTML generation
    html_content = "<h1>Available Notebooks</h1><ul>"
    if notebooks:
        for nb in notebooks:
            html_content += f'<li><a href="{nb["url"]}">{nb["name"]}</a></li>'
    else:
        html_content += "<li>No notebooks found in the specified directory.</li>"
    html_content += "</ul>"

    return render_template_string(html_content)


@app.route('/view/<path:notebook_rel_path>')
def view_notebook(notebook_rel_path):
    # Use the global notebook_dir_global set from args
    if not notebook_dir_global:
         abort(500, "Server configuration error: Notebook directory not set.")
    if not is_safe_path(notebook_rel_path): # Pass only relative path
        abort(403, "Access denied: Path is outside the allowed directory or has an invalid extension.")

    notebook_abs_path = os.path.abspath(os.path.join(notebook_dir_global, notebook_rel_path))

    if not os.path.exists(notebook_abs_path):
        abort(404, "Notebook not found.")

    try:
        # 1. Read the notebook
        with open(notebook_abs_path, 'r', encoding='utf-8') as f:
            nb = nbformat.read(f, as_version=4)

        # 2. Convert to HTML using nbconvert
        # Use 'full' template for CSS/JS, or 'basic' and add your own styling
        html_exporter = HTMLExporter(template_name='full')
        (body, resources) = html_exporter.from_notebook_node(nb)

        # 3. Inject SocketIO JavaScript and refresh logic
        # Note: This is a basic injection. A cleaner way is using Flask templates.
        socketio_js = """
        <script src="https://cdnjs.cloudflare.com/ajax/libs/socket.io/4.0.1/socket.io.js"></script>
        <script>
            // Use ws:// or wss:// depending on your server setup
            const socket = io(window.location.origin);
            const notebookPath = "{notebook_abs_path}"; // Pass the absolute path for server-side matching

            socket.on('connect', () => {{
                console.log('Socket.IO connected');
                // Tell the server which notebook this client is viewing
                socket.emit('join_notebook_view', {{ path: notebookPath }});
            }});

            socket.on('refresh', (data) => {{
                // Optional: Check if the refresh is for *this* notebook if the server logic isn't precise
                // if (data.path === notebookPath) {
                   console.log('Received refresh signal for', data.path, '- Reloading...');
                   window.location.reload();
                // }
            }});

            socket.on('disconnect', () => {{
                console.log('Socket.IO disconnected');
                // Optionally notify server on disconnect if needed for observer cleanup
                // socket.emit('leave_notebook_view', { path: notebookPath });
            }});

             // Clean up connection when page unloads
            window.addEventListener('beforeunload', () => {{
                socket.disconnect();
            }});
        </script>
        """.replace("{notebook_abs_path}", notebook_abs_path)
        # A simple way to add the script tag before </body>
        html_output = body.replace('</body>', socketio_js + '</body>')

        return html_output

    except Exception as e:
        print(f"Error processing notebook {notebook_abs_path}: {e}")
        abort(500, f"Error processing notebook: {e}")

# --- SocketIO Event Handlers ---

from flask import request # Need request context for sid

@socketio.on('join_notebook_view')
def handle_join(data):
    path = data.get('path')
    sid = request.sid # Get the client's session ID
    if path and os.path.exists(path): # Check path validity again
         print(f"Client {sid} joined view for {path}")
         start_monitoring(path) # Ensure monitoring is active
         with monitor_lock:
             if path in monitored_files:
                 monitored_files[path]['clients'].add(sid)
                 # Optional: Add the client to a room for easier targeting
                 # join_room(path) # If using Flask-SocketIO rooms

@socketio.on('disconnect')
def handle_disconnect():
    sid = request.sid
    print(f"Client {sid} disconnected")
    # Remove client from all monitored files it might have been viewing
    with monitor_lock:
        paths_to_check = []
        for path, data in monitored_files.items():
            if sid in data['clients']:
                data['clients'].remove(sid)
                paths_to_check.append(path)

    # Check if observers can be stopped outside the main lock
    # for path in paths_to_check:
    #    stop_monitoring(path) # Add logic to stop if no clients remain

# --- Main Execution ---
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Flask server to view and auto-refresh Jupyter notebooks.")
    parser.add_argument(
        'notebook_dir',
        type=str,
        help='The directory containing the Jupyter notebooks to serve.'
    )
    parser.add_argument(
        '--port',
        type=int,
        default=5001,
        help='Port number to run the server on (default: 5001).'
    )
    parser.add_argument(
        '--host',
        type=str,
        default='127.0.0.1', # Changed default to localhost
        help='Host address to bind the server to (default: 127.0.0.1).'
    )
    parser.add_argument(
        '--debug',
        action='store_true',
        help='Enable Flask debug mode.'
    )

    args = parser.parse_args()

    # Validate and store the notebook directory globally
    notebook_dir_global = os.path.abspath(args.notebook_dir)
    if not os.path.isdir(notebook_dir_global):
        print(f"Error: Notebook directory not found or is not a directory: {notebook_dir_global}")
        exit(1)

    print(f"[*] Serving notebooks from: {notebook_dir_global}")
    # Use threaded=True for Flask dev server with SocketIO, or deploy with Gunicorn/Eventlet/Gevent
    socketio.run(app, debug=args.debug, host=args.host, port=args.port)
    # Proper cleanup of observers on shutdown needed for production
