#!/usr/bin/env python3
"""
Test client for the notebook server.
Sends Python code to the server and displays the response.
"""

import requests
import json
import sys
import time

SERVER_URL = "http://localhost:6060"

def send_code(code):
    """Send code to the notebook server and print the response."""
    try:
        response = requests.post(
            SERVER_URL,
            json={"code": code},
            headers={"Content-Type": "application/json"}
        )
        
        if response.status_code == 200:
            result = response.json()
            print(f"Code executed successfully!")
            
            # Print cell outputs
            if 'cell' in result and 'outputs' in result['cell']:
                for output in result['cell']['outputs']:
                    if output['type'] == 'stream':
                        print(f"Output: {output['text']}")
                    elif output['type'] == 'execute_result':
                        if 'text/plain' in output['data']:
                            print(f"Result: {output['data']['text/plain']}")
                    elif output['type'] == 'error':
                        print(f"Error: {output['ename']}: {output['evalue']}")
            
            return True
        else:
            print(f"Error: {response.status_code}")
            print(response.text)
            return False
            
    except Exception as e:
        print(f"Exception: {e}")
        return False

def main():
    """Main function to run the test client."""
    if len(sys.argv) > 1:
        # Code provided as command line argument
        code = " ".join(sys.argv[1:])
        send_code(code)
    else:
        # Interactive mode
        print("Enter Python code to execute (Ctrl+D to exit):")
        try:
            while True:
                code_lines = []
                print(">>> ", end="")
                
                # Collect multi-line input
                while True:
                    try:
                        line = input()
                        if line.strip() == "":
                            break
                        code_lines.append(line)
                        print("... ", end="")
                    except EOFError:
                        raise
                
                if not code_lines:
                    continue
                    
                code = "\n".join(code_lines)
                if code.lower() in ["exit", "quit"]:
                    break
                    
                send_code(code)
                print()  # Empty line for readability
                
        except EOFError:
            print("\nExiting...")
        except KeyboardInterrupt:
            print("\nExiting...")

if __name__ == "__main__":
    main()
