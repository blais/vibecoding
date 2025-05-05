# jupyter_connector.py
import asyncio
import json
import os
import sys
from pathlib import Path
from typing import AsyncGenerator, Dict, List, Optional, Tuple

# Ensure nest_asyncio is applied if running in certain environments
# import nest_asyncio
# nest_asyncio.apply()

# Prefer AsyncKernelClient for better integration with asyncio frameworks like NiceGUI
#from jupyter_client.asynckernelmanager import AsyncKernelManager
from jupyter_client import AsyncKernelManager
#from jupyter_client.client import AsyncKernelClient
from jupyter_client import AsyncKernelClient
from jupyter_client.connect import find_connection_file, jupyter_runtime_dir
from jupyter_client.kernelspec import find_kernel_specs


class JupyterKernelConnector:
    """
    Manages connection and communication with a running Jupyter kernel.

    Provides methods to list available kernels, connect, execute code,
    and stream results back asynchronously.
    """

    def __init__(self, connection_file: str):
        """
        Initializes the connector but does not connect yet.

        Args:
            connection_file: Path to the kernel's JSON connection file.
        """
        if not Path(connection_file).is_file():
            raise FileNotFoundError(f"Connection file not found: {connection_file}")
        self.connection_file = connection_file
        self.client: Optional[AsyncKernelClient] = None
        self._is_connected = False

    @staticmethod
    def find_running_kernels() -> List[Tuple[str, str]]:
        """
        Finds running Jupyter kernels by looking for connection files.

        Returns:
            A list of tuples, where each tuple contains:
            (kernel_id/filename, full_path_to_connection_file)
        """
        runtime_dir = jupyter_runtime_dir()
        connection_files = []
        try:
            for item in os.listdir(runtime_dir):
                if item.startswith("kernel-") and item.endswith(".json"):
                    full_path = os.path.join(runtime_dir, item)
                    try:
                        # Basic validation: try loading json
                        with open(full_path, "r") as f:
                            json.load(f)
                        connection_files.append((item, full_path))
                    except Exception:
                        # Ignore files that are not valid json or unreadable
                        continue
        except FileNotFoundError:
            print(
                f"Jupyter runtime directory not found: {runtime_dir}", file=sys.stderr
            )
            return []
        except Exception as e:
            print(f"Error listing kernel connection files: {e}", file=sys.stderr)
            return []
        return connection_files

    async def connect(self) -> None:
        """Establishes connection to the kernel."""
        if self._is_connected and self.client:
            print("Already connected.")
            return

        print(f"Attempting to connect using: {self.connection_file}")
        self.client = AsyncKernelClient(connection_file=self.connection_file)
        self.client.load_connection_file()

        # Start communication channels
        self.client.start_channels()

        try:
            # Optional: Check if kernel is alive quickly
            # await self.client.wait_for_ready(timeout=5) # Throws TimeoutError
            # Alternative check: Send a simple ping or status request
            await self.client.kernel_info()  # Waits for kernel_info_reply
            print("Kernel connection established and channels started.")
            self._is_connected = True
        except Exception as e:
            print(f"Failed to connect or verify kernel connection: {e}")
            await self.disconnect()  # Clean up if connection failed
            raise ConnectionError(f"Could not connect to kernel: {e}")

    async def disconnect(self) -> None:
        """Stops communication channels."""
        if self.client and self._is_connected:
            try:
                print("Stopping kernel channels...")
                self.client.stop_channels()
                self._is_connected = False
                self.client = None
                print("Kernel channels stopped.")
            except Exception as e:
                print(f"Error stopping channels: {e}", file=sys.stderr)
        else:
            print("Not connected or client not initialized.")

    async def execute(self, code: str) -> AsyncGenerator[Tuple[str, str], None]:
        """
        Executes Python code on the connected kernel and streams outputs.

        Args:
            code: A string containing the Python code to execute.

        Yields:
            Tuples of (output_type, content), where output_type can be
            'status', 'stdout', 'stderr', 'result', 'display', or 'error'.
        """
        if not self.client or not self._is_connected:
            raise ConnectionError("Client is not connected. Call connect() first.")

        yield ("status", "Executing code...")

        # Execute the code, getting the message ID
        msg_id = await self.client.execute(code, store_history=False)
        # print(f"Sent execute request with msg_id: {msg_id}")

        while True:
            try:
                # Wait for *any* message on the IOPub channel (output, status, etc.)
                # Use a reasonable timeout to prevent hanging indefinitely
                msg = await self.client.get_iopub_msg(timeout=60)
                # print(f"Received IOPub message: {msg['msg_type']} (parent: {msg['parent_header'].get('msg_id')})")

                # Ensure the message corresponds to our execution request
                if msg["parent_header"].get("msg_id") != msg_id:
                    # print(f"Skipping message from different parent: {msg['parent_header'].get('msg_id')}")
                    continue

                msg_type = msg["msg_type"]
                content = msg["content"]

                if msg_type == "status":
                    execution_state = content.get("execution_state")
                    yield ("status", f"Kernel status: {execution_state}")
                    if execution_state == "idle":
                        # Idle *might* mean done, but output could still be buffered/in-flight.
                        # We need to wait for the execute_reply on the SHELL channel
                        # for definitive completion. However, streaming IOPub is common.
                        # We'll rely on seeing subsequent messages or the shell reply check below.
                        # Let's break here for simplicity in streaming, assuming idle + no more iopub msgs means done
                        # Check if shell channel has the reply confirming idle status for *this* msg_id
                        shell_msg = await self.client.get_shell_msg(
                            timeout=1
                        )  # Short timeout
                        if (
                            shell_msg["parent_header"].get("msg_id") == msg_id
                            and shell_msg["msg_type"] == "execute_reply"
                        ):
                            # print("Received execute_reply, execution confirmed complete.")
                            if shell_msg["content"]["status"] == "error":
                                ename = shell_msg["content"].get(
                                    "ename", "Unknown Error"
                                )
                                evalue = shell_msg["content"].get("evalue", "")
                                traceback = "\n".join(
                                    shell_msg["content"].get("traceback", [])
                                )
                                yield ("error", f"{ename}: {evalue}\n{traceback}")
                            break  # Definitively finished
                        else:
                            # Put back unexpected shell message if needed or discard
                            # print("Idle status received, but no final execute_reply yet, continuing to listen briefly...")
                            # Continue listening on iopub for a bit longer in case of delayed output
                            pass

                elif msg_type == "stream":
                    stream_name = content.get(
                        "name", "unknown_stream"
                    )  # 'stdout' or 'stderr'
                    text = content.get("text", "")
                    yield (stream_name, text)

                elif msg_type == "execute_result":
                    # Result of the execution (e.g., the value of the last expression)
                    data = content.get("data", {})
                    text_plain = data.get("text/plain", "")
                    yield ("result", text_plain)

                elif msg_type == "display_data":
                    # Data displayed using display() (e.g., plots, rich outputs)
                    data = content.get("data", {})
                    text_plain = data.get(
                        "text/plain", "<<Display data (non-text)>>"
                    )  # Simplification
                    # Could potentially handle other mime types like image/png here
                    yield ("display", text_plain)

                elif msg_type == "error":
                    ename = content.get("ename", "Unknown Error")
                    evalue = content.get("evalue", "")
                    traceback = "\n".join(content.get("traceback", []))
                    yield ("error", f"{ename}: {evalue}\n{traceback}")
                    # An error usually means execution stops, associated execute_reply will also be 'error'
                    # We might want to break here after yielding the error, matching behavior above.
                    # Let's wait for the idle status / execute_reply for consistency.

            except asyncio.TimeoutError:
                yield (
                    "status",
                    "Timeout waiting for kernel message. Execution might be ongoing or finished.",
                )
                break
            except Empty:  # Queue.Empty is raised by get_iopub_msg if timeout=None and queue is empty, but we use timeout
                yield ("status", "Kernel message queue empty.")
                # This might indicate completion if status was idle before.
                break
            except Exception as e:
                yield ("error", f"Error receiving message: {e}")
                break

        yield ("status", "Execution stream finished.")

    # Optional: Add context manager support for cleaner connect/disconnect
    async def __aenter__(self):
        await self.connect()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.disconnect()
