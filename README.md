# MST O($\log \log {n}$)

Conference PaperPDF Available MST construction in O(log log n) communication rounds June 2003 DOI:10.1145/777412.777428

## Prerequisites

Before you begin, ensure you have met the following requirements:
- You have installed the latest version of [Python](https://www.python.org/downloads/).
- You have a `<Windows/Linux/Mac>` machine.

## Setting Up the Project

To set up the project on your local machine for development and testing purposes, follow these steps.

### Create a Virtual Environment

A virtual environment is a tool that helps to keep dependencies required by different projects in separate places by creating isolated python virtual environments for them. This is one of the most important tools that most of the Python developers use.

```bash
# Create a virtual environment
python -m venv venv

# Activate the virtual environment
.\venv\bin\activate
```

### Install dependencies from requreiments.txt
After setting up the virtual environment, use the package manager pip to install the project dependencies.

```bash
# Ensure you have the latest version of pip
pip install --upgrade pip

# Install project dependencies from requirements.txt file
pip install -r requirements.txt
```

### Run the code
Please check `mst.sh` first to make sure everything is configured correctly. 

```bash
sbatch mst.sh
```