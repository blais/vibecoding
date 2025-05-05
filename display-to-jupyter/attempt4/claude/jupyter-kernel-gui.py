"""
Jupyter Kernel GUI Example

This example demonstrates how to use the JupyterKernelClient library with NiceGUI
to create a simple interface for executing code on a Jupyter kernel.
"""

import sys
import asyncio
from nicegui import ui
from jupyter_kernel_client import JupyterKernelClient


class JupyterKernelGUI:
    """
    GUI interface for interacting with a Jupyter kernel using NiceGUI.
    """
    
    def __init__(self):
        self.kernel_client = None
        self.output_container = None
        self.connection_status = None
        self.connection_file_input = None
        self.code_editor = None
        
    def create_ui(self):
        """Create the GUI interface."""
        with ui.column().classes('w-full max-w-4xl mx-auto gap-4 p-4'):
            ui.label('Jupyter Kernel Interface').classes('text-2xl font-bold')
            
            # Connection section
            with ui.card().classes('w-full'):
                ui.label('Connection').classes('text-lg font-semibold mb-2')
                with ui.row().classes('w-full gap-2'):
                    self.connection_file_input = ui.input(
                        'Connection File', 
                        placeholder='/path/to/connection_file.json'
                    ).classes('flex-grow')
                    ui.button('Connect', on_click=self.connect_to_kernel).props('icon=link')
                    ui.button('Disconnect', on_click=self.disconnect_from_kernel).props('icon=link_off')
                
                self.connection_status = ui.label('Status: Disconnected').classes('mt-2')
            
            # Code execution section
            with ui.card().classes('w-full'):
                ui.label('Code Execution').classes('text-lg font-semibold mb-2')
                self.code_editor = ui.textarea(
                    'Python Code', 
                    placeholder='Enter Python code to execute...'
                ).classes('w-full font-mono').props('rows=10')
                
                ui.button('Execute', on_click=self.execute_code).props('icon=play_arrow').classes('mt-2')
            
            # Output section
            with ui.card().classes('w-full'):
                ui.label('Output').classes('text-lg font-semibold mb-2')
                self.output_container = ui.column().classes('w-full')
                ui.button('Clear Output', on_click=self.clear_output).props('icon=clear_all').classes('mt-2')
    
    async def connect_to_kernel(self):
        """Connect to the Jupyter kernel."""
        connection_file = self.connection_file_input.value
        if not connection_file:
            ui.notify('Please enter a connection file path', type='negative')
            return
        
        try:
            self.kernel_client = JupyterKernelClient(connection_file)
            self.kernel_client.connect()
            self.connection_status.set_text('Status: Connected')
            self.connection_status.classes('text-green-600', remove='text-red-600')
            ui.notify('Connected to kernel successfully', type='positive')
        except Exception as e:
            self.connection_status.set_text(f'Status: Error - {str(e)}')
            self.connection_status.classes('text-red-600', remove='text-green-600')
            ui.notify(f'Failed to connect: {str(e)}', type='negative')
    
    def disconnect_from_kernel(self):
        """Disconnect from the Jupyter kernel."""
        if self.kernel_client:
            self.kernel_client.disconnect()
            self.kernel_client = None
            self.connection_status.set_text('Status: Disconnected')
            self.connection_status.classes(remove='text-green-600 text-red-600')
            ui.notify('Disconnected from kernel', type='info')
        else:
            ui.notify('No kernel connection exists', type='warning')
    
    def execute_code(self):
        """Execute the code in the editor on the kernel."""
        if not self.kernel_client:
            ui.notify('Please connect to a kernel first', type='negative')
            return
        
        code = self.code_editor.value
        if not code.strip():
            ui.notify('Please enter some code to execute', type='warning')
            return
        
        # Execute code asynchronously with callback for output
        self.kernel_client.execute_async(code, self.handle_output)
        ui.notify('Code execution started', type='info')
    
    def handle_output(self, output: dict):
        """Handle output from the kernel."""
        if output['type'] == 'stream':
            with self.output_container:
                with ui.card().classes('w-full'):
                    ui.label(f"Stream ({output['name']})").classes('font-semibold')
                    ui.markdown(f"```\n{output['text']}\n```").classes('w-full')
        
        elif output['type'] == 'execute_result':
            with self.output_container:
                with ui.card().classes('w-full'):
                    ui.label(f"Result [{output['execution_count']}]").classes('font-semibold')
                    if 'text/plain' in output['data']:
                        ui.markdown(f"```\n{output['data']['text/plain']}\n```").classes('w-full')
                    if 'text/html' in output['data']:
                        ui.html(output['data']['text/html']).classes('w-full')
        
        elif output['type'] == 'display_data':
            with self.output_container:
                with ui.card().classes('w-full'):
                    ui.label("Display Data").classes('font-semibold')
                    if 'text/plain' in output['data']:
                        ui.markdown(f"```\n{output['data']['text/plain']}\n```").classes('w-full')
                    if 'text/html' in output['data']:
                        ui.html(output['data']['text/html']).classes('w-full')
                    if 'image/png' in output['data']:
                        ui.image(f"data:image/png;base64,{output['data']['image/png']}")
        
        elif output['type'] == 'error':
            with self.output_container:
                with ui.card().classes('w-full'):
                    ui.label("Error").classes('font-semibold text-red-600')
                    ui.label(f"{output['ename']}: {output['evalue']}").classes('font-semibold text-red-600')
                    ui.markdown(f"```\n{''.join(output['traceback'])}\n```").classes('w-full')
    
    def clear_output(self):
        """Clear the output display."""
        self.output_container.clear()


def main():
    """Main entry point for the application."""
    app = JupyterKernelGUI()
    app.create_ui()
    
    # Run NiceGUI
    ui.run(title='Jupyter Kernel Interface', reload=False)


if __name__ == '__main__':
    main()
