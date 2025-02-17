import os
import sys
from pathlib import Path
import logging

# Get project name from user
while True:
    project_name = input('Enter your project name: ')
    if project_name.strip() != "":
        break

# List of files to create
list_of_files = [
    f"{project_name}/__init__.py",
    f"{project_name}/components/__init__.py",
    f"{project_name}/config/__init__.py",
    f"{project_name}/constants/__init__.py",
    f"{project_name}/entity/__init__.py",
    f"{project_name}/exception/__init__.py",
    f"{project_name}/logger/__init__.py",
    f"{project_name}/pipeline/__init__.py",
    f"{project_name}/utils/__init__.py",
    "config/config.yaml",
    "schema.yaml",
    "app.py",
    "main.py",
    "logs.py",
    "exception.py",
    "setup.py"
]

# Create directories and files
for filepth in list_of_files:
    filepath = Path(filepth)
    filedir, filename = os.path.split(filepth)  # ✅ Fixed os.path.split()

    # Create directory if it does not exist
    if filedir != "":
        os.makedirs(filedir, exist_ok=True)  # ✅ Fixed os.makedirs()

    # Create file if it does not exist or is empty
    if not os.path.exists(filepath) or os.path.getsize(filepath) == 0:  # ✅ Fixed os.path.exists()
        with open(filepath, "w") as f:
            pass
    else:
        logging.info(f"File is already present at: {filepath}")  # ✅ Fixed f-string usage
