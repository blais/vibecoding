#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "jupyter-client",
#     "nbformat",
#     "ipykernel",
# ]
# ///

from queue import Empty
import argparse
import json
import time
import uuid
import os
import traceback

import jupyter_client
import nbformat
from jupyter_client import KernelManager, find_connection_file
def display_image_in_kernel_and_notebook(image_path, notebook_path=None, kernel_connection_file=None):
    """
    Connects to a running Jupyter kernel, executes code to display an image,
    and optionally adds a cell with that code to a specified notebook file.

    Args:
        image_path (str): The path to the image file.
                          Make sure this path is accessible from the kernel's environment.
        notebook_path (str, optional): Path to the .ipynb notebook file to add the cell to.
                                       If provided, a new cell will be appended to this file.
                                       The live notebook view might need a manual refresh.
        kernel_connection_file (str, optional): The path to the kernel's connection file.
                                                If None, the most recently started kernel will be used.
    """
    kc = None  # Initialize client to None for cleanup
    try:
        if kernel_connection_file is None:
            # Use jupyter_client's find_connection_file
            kernel_connection_file = find_connection_file()
        print(f"Using kernel connection file: {kernel_connection_file}")

        # Use BlockingKernelClient to connect to an existing kernel
        kc = jupyter_client.BlockingKernelClient(connection_file=kernel_connection_file)
        kc.load_connection_file()  # Load connection details from the file
        kc.start_channels()  # Start the communication channels (IOPub, Shell, etc.)

        # Optional but recommended: Check if kernel is alive after starting channels
        # Send a kernel_info_request to confirm the kernel is responsive
        try:
            kc.kernel_info()
            print("Successfully connected to kernel and received kernel info.")
        except Exception as info_error:
            print(f"Failed to get kernel info after starting channels: {info_error}")
            # If it fails here, the connection file might be stale, kernel might be dead,
            # or there might be network issues.
            raise  # Re-raise the exception

        print(f"Client: {kc}")
        # kc.wait_for_ready() is not needed here, channels are started.

        # Ensure the image path is properly escaped (especially for Windows paths)
        # Using repr() often helps create a valid string literal
        escaped_image_path = repr(image_path)

        code = f"""
from IPython.display import Image, display
import os
image_path = {escaped_image_path}
print(f"Attempting to display image: {{image_path}}")
if os.path.exists(image_path):
    try:
        img = Image(image_path)
        display(img)
        print("Image display command executed.")
    except Exception as e:
        print(f"Error displaying image in kernel: {{e}}")
else:
    print(f"Error: Image path does not exist from kernel's perspective: {{image_path}}")

# Keep the simple test for verification
"""
        # Store the code for potential notebook insertion
        cell_code = code.strip()
        print(f"Executing code in kernel:\n---\n{cell_code}\n---")

        # Use the high-level execute method which sends an execute_request
        # and waits for the execute_reply on the shell channel.
        msg_id = kc.execute(
            code,
            silent=False,
            store_history=True,
            allow_stdin=False,
            stop_on_error=True,  # Keep this True if you want the kernel to stop on error
        )
        print(f"Execute request sent with msg_id: {msg_id}")

        # The high-level kc.execute waits for the 'execute_reply' message (status: ok/error)
        # on the SHELL channel. We now need to listen on the IOPub channel for outputs
        # (like display_data, execute_result, stream).

        print("Waiting for execution output on IOPub channel...")
        while True:
            try:
                # Get messages from the IOPub channel (output, intermediate results)
                iopub_msg = kc.get_iopub_msg(timeout=1)  # Check for 1 second

                # Check if the message corresponds to our execution request
                if iopub_msg["parent_header"].get("msg_id") == msg_id:
                    msg_type = iopub_msg["msg_type"]
                    content = iopub_msg["content"]
                    print(f"Received IOPub message type: {msg_type}")

                    if msg_type == "status":
                        # Kernel status updates (busy, idle)
                        if content["execution_state"] == "idle":
                            print("Kernel is now idle. Execution likely complete.")
                            break  # Exit the loop once the kernel is idle after our request
                    elif msg_type == "stream":
                        # Output from print() statements
                        print(
                            f"  Stream ({content['name']}): {content['text'].strip()}"
                        )
                    elif msg_type == "display_data":
                        # Output from display() - this is where the image data would be
                        print("  Display data received:")
                        if "image/png" in content["data"]:
                            print("    (PNG Image data found)")
                        elif "text/plain" in content["data"]:
                            print(f"    Plain text: {content['data']['text/plain']}")
                        else:
                            print(f"    Data keys: {content['data'].keys()}")
                    elif msg_type == "execute_result":
                        # Output of the *last* expression if it wasn't display()
                        print("  Execute result received:")
                        if "text/plain" in content["data"]:
                            print(f"    Plain text: {content['data']['text/plain']}")
                        else:
                            print(f"    Data keys: {content['data'].keys()}")
                    elif msg_type == "error":
                        print(
                            f"  Kernel Error: {content['ename']} - {content['evalue']}"
                        )
                        # print traceback if needed
                        # print("\n".join(content['traceback']))
                        break  # Stop listening if an error occurred in the kernel code

            except Empty:
                # Timeout occurred, no message received. Check if kernel is still alive.
                if not kc.is_alive():
                    print("Kernel died unexpectedly while waiting for output.")
                    break
                # If alive, just continue waiting
                pass
            except Exception as e:
                print(f"An error occurred while processing IOPub messages: {e}")
                break

        print("Finished listening for kernel output.")

        # --- Add cell to notebook file if path provided ---
        if notebook_path:
            print(f"\nAttempting to add cell to notebook file: {notebook_path}")
            if not os.path.exists(notebook_path):
                print(f"Error: Notebook file not found at {notebook_path}")
            else:
                try:
                    # Read the notebook using nbformat
                    with open(notebook_path, 'r', encoding='utf-8') as f:
                        nb = nbformat.read(f, as_version=4)

                    # Create a new code cell with the executed code
                    new_cell = nbformat.v4.new_code_cell(cell_code)

                    # Append the new cell to the notebook
                    nb.cells.append(new_cell)

                    # Write the modified notebook back to the file
                    with open(notebook_path, 'w', encoding='utf-8') as f:
                        nbformat.write(nb, f)
                    print(f"Successfully added cell to {notebook_path}.")
                    print("NOTE: You may need to reload the notebook in your browser/editor to see the added cell.")

                except Exception as nb_error:
                    print(f"Error updating notebook file {notebook_path}: {nb_error}")
                    traceback.print_exc()


    except jupyter_client.KernelNotFoundError:
        print(f"Error: Could not find kernel connection file: {kernel_connection_file}")
    except Exception as e:
        print(f"An error occurred during kernel communication or notebook modification: {e}")
        traceback.print_exc()  # Print detailed traceback for debugging
    finally:
        # Ensure kernel client channels are stopped regardless of success or failure
        if kc and kc.channels_running:
            try:
                print("Stopping client channels...")
                kc.stop_channels()
                print("Client channels stopped.")
            except Exception as stop_exc:
                print(f"Error stopping client channels: {stop_exc}")


# # --- Example Usage ---
# # Make sure 'test.png' exists or change to a valid path
# # Ensure the path is accessible *from the environment where the Jupyter kernel is running*
# try:
#     # Create a dummy file for testing if it doesn't exist
#     with open("test.png", "a") as f:
#         pass
#     # On Windows, paths might look like: 'C:\\Users\\YourUser\\Pictures\\my_image.png'
#     # On Linux/macOS: '/home/youruser/images/my_image.png'
#     # Example with notebook file:
#     # display_image_in_kernel_and_notebook('test.png', notebook_path='MyNotebook.ipynb')
# except Exception as e:
#     print(f"Error in example usage setup: {e}")


def main():
    parser = argparse.ArgumentParser(
        description="Display an image in a Jupyter kernel and optionally add the cell to a notebook file."
    )
    parser.add_argument("image_file", help="Path to the image file.")
    parser.add_argument(
        "-n", "--notebook",
        metavar="NOTEBOOK_PATH",
        help="Path to the .ipynb notebook file to add the cell to (optional)."
    )
    parser.add_argument(
        "-k", "--kernel",
        metavar="CONNECTION_FILE",
        help="Path to the kernel connection file (optional, finds latest if omitted)."
    )
    args = parser.parse_args()

    display_image_in_kernel_and_notebook(args.image_file, args.notebook, args.kernel)


if __name__ == "__main__":
    main()
