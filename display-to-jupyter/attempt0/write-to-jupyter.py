#!/usr/bin/env python3
"""Attempt to write to a running Jupyter notebook.
"""

import nbformat
from nbformat import v4
from jupyter_client import BlockingKernelClient
from jupyter_client import find_connection_file

import argparse
import logging


def main():
    logging.basicConfig(level=logging.DEBUG, format="%(levelname)-8s: %(message)s")
    parser = argparse.ArgumentParser(description=__doc__.strip())
    parser.add_argument("filename", help="Filename of notebook")
    args = parser.parse_args()

    # Connect to the running Jupyter notebook server
    kernel_client = BlockingKernelClient()
    cf = find_connection_file()
    #print(kernel_client.load_connection_info())
    kernel_client.load_connection_file(cf)  # Load connection info from default location
    kernel_client.start_channels()

    # Create a new code cell
    new_cell = v4.new_code_cell(source="2+2")
    new_cell2 = v4.new_code_cell(source="%%javascript\nlocation.reload(true);")

    # Get the current notebook content
    notebook_path = args.filename  # Specify the path to your notebook
    logging.debug(notebook_path)
    with open(notebook_path, "r") as f:
        notebook_content = nbformat.read(f, as_version=4)

    # Insert the new cell at the beginning of the notebook
    notebook_content["cells"].insert(0, new_cell)
    notebook_content["cells"].insert(1, new_cell2)

    # Save the modified notebook content
    with open(notebook_path, "w") as f:
        nbformat.write(notebook_content, f)

    # Execute the new cell
    kernel_client.execute_interactive(new_cell.source)
    #out = kernel_client.execute_interactive("%%javascript\nlocation;")
    kernel_client.execute_interactive("%%javascript\nlocation.reload(true);")

    # Shutdown the kernel client
    kernel_client.stop_channels()

    # Replace "path_to_your_notebook.ipynb" with the path to your Jupyter notebook file. This code assumes that the notebook is running with a kernel that supports execution, such as a Python kernel.


if __name__ == "__main__":
    main()
