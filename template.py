import os
from pathlib import Path

import logging

logging.basicConfig(level=logging.INFO, format='[%(asctime)s]: %(message)s:')

# Replace with your project name
project_name = "credit_card_defaulters"


    
    # Define the list of files and folders
list_of_files = [
        ".github/workflows/.gitkeep",
        f"src/{project_name}/__init__.py",
        f"src/{project_name}/artifact/__init__.py",
        f"src/{project_name}/component/__init__.py",
        f"src/{project_name}/utils/__init__.py",
        f"src/{project_name}/util/common.py",
        f"src/{project_name}/config/__init__.py",
        f"src/{project_name}/config/configuration.py",
        f"src/{project_name}/pipeline/__init__.py",
        f"src/{project_name}/entity/__init__.py",
        f"src/{project_name}/entity/config_entity.py",
        f"src/{project_name}/constant/__init__.py",
        f"logs/",
        f"saved_models/",
        "config/config.yaml",
        "config/model.yaml",
        "config/schema.yaml",
        "config/params.yaml",
        "dvc.yaml",
        "main.py",
        "app.py",
        "Dockerfile",
        "requirements.txt",
        "setup.py",
        "research/trials.ipynb",
        "templates/index.html",
]

    

for filepath in list_of_files:
    filepath = Path(filepath)
    filedir, filename = os.path.split(filepath)


    if filedir !="":
        os.makedirs(filedir, exist_ok=True)
        logging.info(f"Creating directory; {filedir} for the file: {filename}")

    if (not os.path.exists(filepath)) or (os.path.getsize(filepath) == 0):
        with open(filepath, "w") as f:
            pass
            logging.info(f"Creating empty file: {filepath}")


    else:
        logging.info(f"{filename} is already exists")
