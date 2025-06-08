import subprocess
import webbrowser
import time
import os

# Step 1: Start the FastAPI backend server
def start_backend():
    subprocess.Popen(["uvicorn", "backend.main:app", "--reload"])

# Step 2: Open the frontend index.html in the default browser
def open_frontend():
    frontend_path = os.path.abspath("templates/index.html")
    webbrowser.open(f"file://{frontend_path}")

if __name__ == "__main__":
    start_backend()
    time.sleep(2)  # Wait for server to start
    open_frontend()
