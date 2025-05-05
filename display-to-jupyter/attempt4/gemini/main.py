# main.py
from pathlib import Path
import asyncio
from typing import Any, Dict, List, Optional, Tuple

from jupyter_connector import JupyterKernelConnector  # Import our library

from nicegui import Client, app, ui

# --- Global State ---
# Store kernel connection info {display_name: full_path}
available_kernels: Dict[str, str] = {}
# Store the active connector instance
active_connector: Optional[JupyterKernelConnector] = None

# --- UI Elements ---
kernel_selector: Optional[ui.select] = None
code_input: Optional[ui.textarea] = None
execute_button: Optional[ui.button] = None
results_log: Optional[ui.log] = None
status_label: Optional[ui.label] = None

# --- Core Logic ---


async def update_kernel_list():
    """Finds running kernels and updates the UI selector."""
    global available_kernels, kernel_selector
    print("Searching for running kernels...")
    kernels = JupyterKernelConnector.find_running_kernels()
    available_kernels = {
        f"Kernel ({f[:12]}...) - {Path(p).name}": p for f, p in kernels
    }

    if kernel_selector is not None:
        kernel_selector.options = list(available_kernels.keys())
        kernel_selector.update()
        count = len(available_kernels)
        status_label.set_text(
            f"Found {count} running kernel(s)."
            if count
            else "No running kernels found. Please start one."
        )
        print(f"Found {count} kernels.")
    else:
        print("Kernel selector UI element not ready.")


async def handle_execute():
    """Handles the button click to execute code on the selected kernel."""
    global active_connector, results_log, status_label

    if not kernel_selector or not code_input or not results_log or not status_label:
        ui.notify("UI elements not initialized.", type="negative")
        return

    selected_kernel_display = kernel_selector.value
    code_to_execute = code_input.value

    if not selected_kernel_display or selected_kernel_display not in available_kernels:
        ui.notify("Please select a running kernel.", type="warning")
        return

    if not code_to_execute.strip():
        ui.notify("Please enter some Python code to execute.", type="warning")
        return

    connection_file = available_kernels[selected_kernel_display]

    # Clear previous results
    results_log.clear()
    results_log.push("--- Starting Execution ---")
    status_label.set_text("Connecting to kernel...")

    try:
        # Disconnect previous connector if exists
        if active_connector:
            await active_connector.disconnect()
            active_connector = None

        # Create and connect the connector
        # Using 'async with' would be cleaner if the connector was always short-lived
        # Here, we might want to keep it active, so manual connect/disconnect.
        active_connector = JupyterKernelConnector(connection_file)
        await active_connector.connect()  # Handles connection logic

        status_label.set_text(
            f"Connected. Executing code on {Path(connection_file).name}..."
        )

        # Execute code and stream results
        async for output_type, content in active_connector.execute(code_to_execute):
            prefix = f"[{output_type.upper()}]"
            # Handle potential multi-line content nicely in the log
            lines = str(content).splitlines()
            if not lines:  # Handle empty content case
                results_log.push(f"{prefix}")
            else:
                results_log.push(f"{prefix} {lines[0]}")
                for line in lines[1:]:
                    results_log.push(
                        f"{' ' * len(prefix)} {line}"
                    )  # Indent subsequent lines

        status_label.set_text("Execution finished.")
        ui.notify("Execution finished.", type="positive")

    except ConnectionError as e:
        status_label.set_text(f"Connection Error: {e}")
        ui.notify(f"Connection Error: {e}", type="negative")
        if active_connector:
            await active_connector.disconnect()  # Attempt cleanup
            active_connector = None
    except Exception as e:
        status_label.set_text(f"An error occurred: {e}")
        results_log.push(f"[ERROR] Client-side error: {e}")
        ui.notify(f"An error occurred: {e}", type="negative")
        if active_connector:
            await active_connector.disconnect()  # Attempt cleanup
            active_connector = None
    finally:
        # Optional: Decide if you want to disconnect automatically after execution
        # await active_connector.disconnect()
        # status_label.set_text("Disconnected.")
        pass  # Keep connected for now


# --- UI Setup ---


@ui.page("/")
async def main_page(client: Client):
    global kernel_selector, code_input, execute_button, results_log, status_label

    ui.label("Jupyter Kernel Remote Executor").classes("text-h4")
    status_label = ui.label("Initializing...")

    with ui.row().classes("w-full items-center"):
        kernel_selector = ui.select(
            options=[],
            label="Select Running Kernel",
            with_input=True,  # Allows searching if many kernels
        ).classes("flex-grow")
        ui.button(icon="refresh", on_click=update_kernel_list).tooltip(
            "Refresh kernel list"
        )

    ui.label("Enter Python code to execute:")
    code_input = ui.textarea(
        placeholder="e.g.,\nimport time\nfor i in range(5):\n  print(f'Iteration {i}')\n  time.sleep(0.5)\n'done'"
    ).classes("w-full")

    execute_button = ui.button("Execute on Kernel", on_click=handle_execute)

    ui.label("Results Stream:").classes("text-lg mt-4")
    # Use ui.log for automatically scrollingappend-only text display
    results_log = ui.log(max_lines=100).classes("w-full h-64 border")

    # Automatically update kernel list when the page is ready
    await client.connected()  # Wait for websocket connection
    await update_kernel_list()  # Initial population


# --- App Entry Point ---
if __name__ in {"__main__", "__mp_main__"}:
    # Ensure asyncio event loop compatibility if needed (often good practice with NiceGUI)
    # import nest_asyncio
    # nest_asyncio.apply()

    # Add handler to clean up kernel connection on shutdown
    async def cleanup():
        global active_connector
        print("Shutting down NiceGUI app...")
        if active_connector:
            print("Disconnecting from kernel...")
            await active_connector.disconnect()

    app.on_shutdown(cleanup)

    ui.run()
