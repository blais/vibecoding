#!/usr/bin/env python3
"""
Standalone Jupyter-like server that:
1. Listens on port 6060 for Python source code to execute
2. Serves a web interface on port 8080 that displays executed cells and their outputs
3. Updates connected web clients in real-time via WebSockets
"""

import asyncio
import json
import os
import queue
import threading
from typing import Dict, List, Any, Tuple

# Web server and WebSocket libraries
from aiohttp import web
import websockets

# Jupyter kernel management
from jupyter_client import KernelManager
from jupyter_client.kernelspec import get_kernel_spec


class NotebookServer:
    def __init__(self):
        # Initialize the list of cells with their outputs
        self.cells: List[Dict[str, Any]] = []
        self.cell_counter = 0

        # Keep track of connected WebSocket clients
        self.websocket_clients = set()

        # Start a Jupyter kernel for executing Python code
        self.kernel_manager = KernelManager(kernel_name="python3")
        self.kernel_manager.start_kernel()
        self.kernel_client = self.kernel_manager.client()
        self.kernel_client.start_channels()

        # Set up queues for kernel communication
        self.msg_queue = queue.Queue()

        # Start thread for handling kernel messages
        self.kernel_thread = threading.Thread(target=self._process_kernel_messages)
        self.kernel_thread.daemon = True
        self.kernel_thread.start()

        print("Notebook server initialized with Python kernel")

    def _process_kernel_messages(self):
        """Process messages from the kernel in a separate thread."""
        while True:
            try:
                # Get messages from the kernel
                msg = self.kernel_client.get_iopub_msg(timeout=0.1)

                # Put them in the queue for processing
                self.msg_queue.put(msg)
            except queue.Empty:
                pass
            except Exception as e:
                print(f"Error processing kernel messages: {e}")

    async def execute_code(self, code: str) -> Dict[str, Any]:
        """Execute a code cell and collect the outputs."""
        cell_id = self.cell_counter
        self.cell_counter += 1

        print(f"Executing cell {cell_id}: {code[:50]}{'...' if len(code) > 50 else ''}")

        # Create a new cell
        cell = {"id": cell_id, "code": code, "outputs": [], "status": "running"}

        # Add the cell to our list
        self.cells.append(cell)

        # Notify clients about the new cell immediately
        await self.notify_clients(cell)

        # Send the code to the kernel for execution
        msg_id = self.kernel_client.execute(code)
        print(f"Sent to kernel with msg_id: {msg_id}")

        # Process messages until execution is complete
        while True:
            try:
                # Check for new messages
                if not self.msg_queue.empty():
                    msg = self.msg_queue.get(block=False)

                    # Process the message
                    if "content" in msg:
                        msg_type = msg.get("msg_type", "")

                        if msg_type == "execute_result":
                            # Regular output
                            data = msg["content"].get("data", {})
                            cell["outputs"].append(
                                {"type": "execute_result", "data": data}
                            )

                        elif msg_type == "stream":
                            # stdout/stderr
                            text = msg["content"].get("text", "")
                            name = msg["content"].get("name", "stdout")
                            cell["outputs"].append(
                                {"type": "stream", "name": name, "text": text}
                            )

                        elif msg_type == "display_data":
                            # Display data (like plots)
                            data = msg["content"].get("data", {})
                            cell["outputs"].append(
                                {"type": "display_data", "data": data}
                            )

                        elif msg_type == "error":
                            # Error output
                            ename = msg["content"].get("ename", "")
                            evalue = msg["content"].get("evalue", "")
                            traceback = msg["content"].get("traceback", [])
                            cell["outputs"].append(
                                {
                                    "type": "error",
                                    "ename": ename,
                                    "evalue": evalue,
                                    "traceback": traceback,
                                }
                            )

                        # Notify clients whenever we get new output
                        if msg_type in [
                            "execute_result",
                            "stream",
                            "display_data",
                            "error",
                        ]:
                            # Send an update to all clients
                            await self.notify_clients(cell)

                        elif msg_type == "status":
                            # Kernel status update
                            if msg["content"].get("execution_state") == "idle":
                                # Execution completed
                                if msg["parent_header"].get("msg_id") == msg_id:
                                    cell["status"] = "completed"

                                    # Notify all connected clients of the update
                                    await self.notify_clients(cell)
                                    return cell

            except queue.Empty:
                # No messages, wait a bit
                await asyncio.sleep(0.01)
            except Exception as e:
                print(f"Error in execute_code: {e}")
                cell["status"] = "error"
                cell["outputs"].append(
                    {
                        "type": "error",
                        "ename": "ServerError",
                        "evalue": str(e),
                        "traceback": [],
                    }
                )

                # Notify all connected clients of the update
                await self.notify_clients(cell)
                return cell

    async def notify_clients(self, cell: Dict[str, Any]):
        """Notify all connected WebSocket clients of a cell update."""
        if not self.websocket_clients:
            return

        # Prepare the message
        message = json.dumps({"type": "cell_update", "cell": cell})

        # Notify all connected clients
        for websocket in list(self.websocket_clients):
            try:
                await websocket.send(message)
            except Exception as e:
                print(f"Error sending to WebSocket client: {e}")
                self.websocket_clients.discard(websocket)

    async def handle_code_submission(self, request: web.Request) -> web.Response:
        """Handle code submission on port 6060."""
        try:
            data = await request.json()
            code = data.get("code", "")

            if not code:
                return web.json_response({"error": "No code provided"}, status=400)

            # Execute the code asynchronously
            cell = await self.execute_code(code)

            # Return the cell information
            return web.json_response({"cell": cell})

        except Exception as e:
            return web.json_response({"error": str(e)}, status=500)

    async def handle_websocket(self, request: web.Request) -> web.WebSocketResponse:
        """Handle WebSocket connections for real-time updates."""
        ws = web.WebSocketResponse()
        await ws.prepare(request)

        print("New WebSocket client connected")

        # Add this client to our set of connected clients
        self.websocket_clients.add(ws)

        # Send all existing cells to the new client
        for cell in self.cells:
            await ws.send_json({"type": "cell_update", "cell": cell})

        try:
            async for msg in ws:
                # Handle any incoming WebSocket messages (if needed)
                if msg.type == web.WSMsgType.TEXT:
                    # We could handle client commands here if needed
                    print(f"Received message from client: {msg.data}")
                    try:
                        data = json.loads(msg.data)
                        if data.get("type") == "request_cells":
                            # Client is requesting all cells
                            for cell in self.cells:
                                await ws.send_json(
                                    {"type": "cell_update", "cell": cell}
                                )
                    except json.JSONDecodeError:
                        pass
                elif msg.type == web.WSMsgType.ERROR:
                    print(
                        f"WebSocket connection closed with exception {ws.exception()}"
                    )
        finally:
            # Remove this client when the connection is closed
            self.websocket_clients.discard(ws)
            print("WebSocket client disconnected")

        return ws

    def get_index_html(self) -> str:
        """Generate the HTML for the web interface."""
        return """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Simple Jupyter-like Notebook</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            line-height: 1.6;
            margin: 0;
            padding: 20px;
            max-width: 900px;
            margin: 0 auto;
            background-color: #f9f9f9;
        }
        .cell {
            margin-bottom: 20px;
            border: 1px solid #ddd;
            border-radius: 4px;
            overflow: hidden;
            background-color: white;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        }
        .cell-number {
            background-color: #f0f0f0;
            padding: 5px 10px;
            font-family: monospace;
            color: #666;
            border-bottom: 1px solid #ddd;
        }
        .code-area {
            background-color: #f8f8f8;
            padding: 10px;
            font-family: monospace;
            white-space: pre-wrap;
            border-bottom: 1px solid #ddd;
            overflow-x: auto;
        }
        .output-area {
            padding: 10px;
            font-family: monospace;
            white-space: pre-wrap;
            overflow-x: auto;
            min-height: 10px;
        }
        .output-area img {
            max-width: 100%;
            height: auto;
        }
        .error {
            background-color: #ffecec;
            color: #d8000c;
            padding: 5px;
            border-radius: 3px;
            margin-top: 5px;
        }
        .cell-status {
            font-size: 12px;
            color: #666;
            padding: 2px 10px;
            text-align: right;
            border-top: 1px solid #eee;
        }
        h1 {
            color: #333;
            border-bottom: 2px solid #eee;
            padding-bottom: 10px;
        }
        .connection-status {
            position: fixed;
            top: 10px;
            right: 10px;
            padding: 5px 10px;
            border-radius: 4px;
            font-size: 12px;
            color: white;
        }
        .connected {
            background-color: #4CAF50;
        }
        .disconnected {
            background-color: #f44336;
        }
    </style>
</head>
<body>
    <h1>Simple Jupyter-like Notebook</h1>
    <p>This page displays cells executed through the port 6060 API.</p>

    <div id="connection-status" class="connection-status disconnected">Disconnected</div>

    <div id="notebook-container">
        <!-- Cells will be inserted here -->
    </div>

    <div id="no-cells-message" style="text-align: center; margin-top: 50px; color: #666;">
        <p>No cells have been executed yet.</p>
        <p>Use the test_client.py script or send POST requests to port 6060 to execute code.</p>
        <code>curl -X POST -H "Content-Type: application/json" -d '{"code":"print(\"Hello, world!\")"}'
 http://localhost:6060/</code>
    </div>

    <script>
        // Connect to WebSocket for real-time updates
        const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
        const wsUrl = `${protocol}//${window.location.host}/ws`;
        let socket;
        let reconnectAttempts = 0;

        function connectWebSocket() {
            socket = new WebSocket(wsUrl);

            const connectionStatus = document.getElementById('connection-status');

            socket.onmessage = function(event) {
                const data = JSON.parse(event.data);

                if (data.type === 'cell_update') {
                    // Handle cell update
                    updateCell(data.cell);

                    // Hide the "no cells" message if we have cells
                    document.getElementById('no-cells-message').style.display = 'none';
                }
            };

            socket.onopen = function(event) {
                console.log('WebSocket connection established');
                connectionStatus.textContent = 'Connected';
                connectionStatus.className = 'connection-status connected';
                reconnectAttempts = 0;

                // Request all cells when connection is established
                socket.send(JSON.stringify({
                    type: 'request_cells'
                }));
            };

            socket.onclose = function(event) {
                console.log('Connection closed');
                connectionStatus.textContent = 'Disconnected';
                connectionStatus.className = 'connection-status disconnected';

                // Attempt to reconnect after a delay
                reconnectAttempts++;
                const delay = Math.min(30000, 1000 * Math.pow(1.5, reconnectAttempts));
                console.log(`Attempting to reconnect in ${delay/1000} seconds...`);

                setTimeout(connectWebSocket, delay);
            };

            socket.onerror = function(error) {
                console.error('WebSocket error:', error);
                connectionStatus.textContent = 'Connection Error';
                connectionStatus.className = 'connection-status disconnected';
            };
        }

        // Initial connection
        connectWebSocket();

        function updateCell(cell) {
            const container = document.getElementById('notebook-container');

            // Check if the cell already exists
            let cellElement = document.getElementById(`cell-${cell.id}`);

            if (!cellElement) {
                // Create a new cell element
                cellElement = document.createElement('div');
                cellElement.id = `cell-${cell.id}`;
                cellElement.className = 'cell';

                // Create cell structure
                cellElement.innerHTML = `
                    <div class="cell-number">In [${cell.id}]:</div>
                    <div class="code-area">${escapeHtml(cell.code)}</div>
                    <div id="output-${cell.id}" class="output-area"></div>
                    <div class="cell-status" id="status-${cell.id}">${cell.status}</div>
                `;

                // Add the cell to the notebook
                container.appendChild(cellElement);
            }

            // Update the status
            const statusElement = document.getElementById(`status-${cell.id}`);
            if (statusElement) {
                statusElement.textContent = cell.status;
            }

            // Update the outputs
            const outputElement = document.getElementById(`output-${cell.id}`);
            if (outputElement) {
                outputElement.innerHTML = ''; // Clear existing outputs

                if (cell.outputs && cell.outputs.length > 0) {
                    for (const output of cell.outputs) {
                        const outputDiv = document.createElement('div');

                        if (output.type === 'stream') {
                            outputDiv.textContent = output.text;
                        } else if (output.type === 'execute_result') {
                            if (output.data['text/plain']) {
                                outputDiv.textContent = output.data['text/plain'];
                            } else if (output.data['text/html']) {
                                outputDiv.innerHTML = output.data['text/html'];
                            }
                        } else if (output.type === 'display_data') {
                            if (output.data['image/png']) {
                                const img = document.createElement('img');
                                img.src = 'data:image/png;base64,' + output.data['image/png'];
                                outputDiv.appendChild(img);
                            } else if (output.data['text/html']) {
                                outputDiv.innerHTML = output.data['text/html'];
                            } else if (output.data['text/plain']) {
                                outputDiv.textContent = output.data['text/plain'];
                            }
                        } else if (output.type === 'error') {
                            outputDiv.className = 'error';
                            outputDiv.textContent = output.traceback.join('\n');
                        }

                        outputElement.appendChild(outputDiv);
                    }
                }
            }
        }

        function escapeHtml(unsafe) {
            return unsafe
                .replace(/&/g, "&amp;")
                .replace(/</g, "&lt;")
                .replace(/>/g, "&gt;")
                .replace(/"/g, "&quot;")
                .replace(/'/g, "&#039;");
        }
    </script>
</body>
</html>
        """

    async def handle_index(self, request: web.Request) -> web.Response:
        """Serve the index page."""
        return web.Response(text=self.get_index_html(), content_type="text/html")

    async def start(self):
        """Start the server, listening on both ports."""
        # Set up the web server for port 8080
        app = web.Application()
        app.router.add_get("/", self.handle_index)
        app.router.add_get("/ws", self.handle_websocket)

        # Create a task for the web server
        runner = web.AppRunner(app)
        await runner.setup()
        site = web.TCPSite(runner, "0.0.0.0", 8080)
        await site.start()
        print("Web server started on http://localhost:8080")

        # Set up the code submission server on port 6060
        code_app = web.Application()
        code_app.router.add_post("/", self.handle_code_submission)

        # Create a task for the code submission server
        code_runner = web.AppRunner(code_app)
        await code_runner.setup()
        code_site = web.TCPSite(code_runner, "0.0.0.0", 6060)
        await code_site.start()
        print("Code submission server started on http://localhost:6060")

        # Keep the server running
        while True:
            await asyncio.sleep(3600)  # Wait for 1 hour (or any large value)

    def shutdown(self):
        """Clean up resources when shutting down."""
        if self.kernel_client:
            self.kernel_client.stop_channels()
        if self.kernel_manager:
            self.kernel_manager.shutdown_kernel()


async def main():
    server = NotebookServer()
    try:
        await server.start()
    except KeyboardInterrupt:
        print("Shutting down server...")
    finally:
        server.shutdown()


if __name__ == "__main__":
    asyncio.run(main())
