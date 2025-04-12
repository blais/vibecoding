#!/usr/bin/env python3
"""
Simple client to send Python code to the notebook server running on port 6060.
This client demonstrates how to interact with the server API.
"""

import argparse
import json
import requests
import time


def send_code(code, server_url="http://localhost:6060"):
    """Send Python code to the notebook server."""
    payload = {"code": code}
    
    try:
        # Send the code to the server
        response = requests.post(server_url, json=payload)
        
        # Check if the request was successful
        if response.status_code == 200:
            result = response.json()
            print(f"Successfully sent code. Cell ID: {result['cell']['id']}")
            return result
        else:
            print(f"Error: Server returned status code {response.status_code}")
            print(response.text)
            return None
            
    except requests.exceptions.RequestException as e:
        print(f"Connection error: {e}")
        return None


def run_examples(server_url="http://localhost:6060"):
    """Run a series of example code cells."""
    examples = [
        # Simple calculation
        "2 + 2",
        
        # Variable definition
        """x = 42
print(f"The value of x is {x}")""",
        
        # List comprehension
        """squares = [x**2 for x in range(10)]
print(squares)""",
        
        # Creating a function
        """def factorial(n):
    if n <= 1:
        return 1
    return n * factorial(n-1)

print(factorial(5))""",
        
        # Matplotlib plot (if matplotlib is installed in the kernel)
        """try:
    import matplotlib.pyplot as plt
    import numpy as np
    
    # Generate data
    x = np.linspace(0, 10, 100)
    y = np.sin(x)
    
    # Create plot
    plt.figure(figsize=(8, 4))
    plt.plot(x, y, 'b-', label='sin(x)')
    plt.title('Sine Wave')
    plt.xlabel('x')
    plt.ylabel('sin(x)')
    plt.grid(True)
    plt.legend()
    plt.show()
except ImportError:
    print("Matplotlib not available in the kernel")""",
        
        # Error example
        """# This will generate an error
1/0"""
    ]
    
    # Send each example to the server
    for i, code in enumerate(examples):
        print(f"\nSending example {i+1}/{len(examples)}:")
        print("-" * 40)
        print(code)
        print("-" * 40)
        
        result = send_code(code, server_url)
        
        # Wait a moment between requests
        time.sleep(1)
    
    print("\nAll examples have been sent. Check the web interface at http://localhost:8080")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Send Python code to a notebook server")
    parser.add_argument("--server", default="http://localhost:6060", help="Server URL (default: http://localhost:6060)")
    parser.add_argument("--code", help="Python code to send (if not specified, examples will be run)")
    
    args = parser.parse_args()
    
    if args.code:
        # Send the specified code
        send_code(args.code, args.server)
    else:
        # Run examples
        run_examples(args.server)
