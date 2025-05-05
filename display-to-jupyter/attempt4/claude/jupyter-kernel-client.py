"""
Jupyter Kernel Client Library

This library provides functionality to connect to a running Jupyter kernel,
execute code, and stream the results.
"""

import asyncio
import json
import queue
import threading
from typing import Any, Callable, Dict, Optional, List
from jupyter_client import BlockingKernelClient


class JupyterKernelClient:
    """
    Client for connecting to and executing code on a Jupyter kernel.
    """
    
    def __init__(self, connection_file: str):
        """
        Initialize the kernel client with a connection file.
        
        Args:
            connection_file: Path to the kernel connection file (JSON format)
        """
        self.client = BlockingKernelClient(connection_file=connection_file)
        self.client.load_connection_file()
        self.result_queue = queue.Queue()
        self._execution_thread = None
        self._stop_event = threading.Event()
    
    def connect(self):
        """Establish connection to the kernel."""
        self.client.start_channels()
        try:
            self.client.wait_for_ready()
        except RuntimeError:
            print("Failed to connect to kernel. Are you sure it's running?")
            raise
    
    def disconnect(self):
        """Close all channels and stop the client."""
        self._stop_event.set()
        if self._execution_thread and self._execution_thread.is_alive():
            self._execution_thread.join()
        self.client.stop_channels()
    
    def execute_code(
        self, 
        code: str, 
        on_output: Optional[Callable[[Dict[str, Any]], None]] = None
    ) -> List[Dict[str, Any]]:
        """
        Execute code on the kernel and handle the output.
        
        Args:
            code: Python code to execute
            on_output: Optional callback function called for each output
            
        Returns:
            List of output messages from the kernel
        """
        msg_id = self.client.execute(code)
        results = []
        
        # Poll for messages until execution is complete
        while True:
            try:
                msg = self.client.get_iopub_msg(timeout=1)
                
                if self._stop_event.is_set():
                    break
                
                if msg['parent_header'].get('msg_id') != msg_id:
                    continue
                
                msg_type = msg['msg_type']
                content = msg['content']
                
                # Convert to standard format
                output = self._convert_message(msg_type, content)
                if output:
                    results.append(output)
                    if on_output:
                        on_output(output)
                
                # Check if execution is complete
                if msg_type == 'status' and content['execution_state'] == 'idle':
                    break
                    
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Error processing message: {e}")
                break
        
        return results
    
    def execute_async(
        self, 
        code: str, 
        on_output: Optional[Callable[[Dict[str, Any]], None]] = None
    ):
        """
        Execute code asynchronously in a separate thread.
        
        Args:
            code: Python code to execute
            on_output: Optional callback function called for each output
        """
        def run():
            self.execute_code(code, on_output)
        
        self._execution_thread = threading.Thread(target=run)
        self._execution_thread.daemon = True
        self._execution_thread.start()
    
    def _convert_message(self, msg_type: str, content: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Convert Jupyter message to a standard format.
        
        Args:
            msg_type: Message type from Jupyter
            content: Message content
            
        Returns:
            Standardized output dictionary or None if not a relevant message
        """
        if msg_type == 'stream':
            return {
                'type': 'stream',
                'name': content['name'],
                'text': content['text']
            }
        elif msg_type == 'execute_result':
            return {
                'type': 'execute_result',
                'data': content['data'],
                'execution_count': content['execution_count']
            }
        elif msg_type == 'display_data':
            return {
                'type': 'display_data',
                'data': content['data']
            }
        elif msg_type == 'error':
            return {
                'type': 'error',
                'ename': content['ename'],
                'evalue': content['evalue'],
                'traceback': content['traceback']
            }
        return None
