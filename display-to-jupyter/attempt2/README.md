Second attempt requesting Claude to write a brand new server and example
client. It decided to execute using a kernel, somehow (which I didn't request
but I like).

    You are a Python expert, well versed in the Jupyter client and server libraries that underpin notebooks.  Write a standalone server that listens to two ports:

    1. port 6060: accepts connections sending Python source code, adds this source code to a list of internal "cells" and executes this code in its running kernel.
    2. port 8080: serves a web page that renders these cells and outputs.

    In particular, the web server should have a long polling connection (via WebSocket) that automatically updates the web clients when a new cell is inserted at the port 6060 connection.

    Produce server code and an example client that will send simple Python example cells to be inserted and rendered.


This does not appear to work either.
