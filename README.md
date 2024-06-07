# OceanStream

[OceanStream](https://oceanstream.io/) is an integrated cloud data platform used for hydroacoustic data collection and analysis in real-time. It is built on [Azure IoT Edge](https://azure.microsoft.com/en-us/products/iot-edge) infrastructure.

<a href="https://oceanstream.io/" target="_blank"><img width="1537" alt="OceanStream.io" src="https://github.com/OceanStreamIO/oceanstream-cli/assets/419506/a1e5ecc6-77d6-45ea-ac4c-51ca1378acd2"></a>

## How to Install

### Prerequisites

- **Python 3.11**: Ensure Python 3.11 is installed on your system.

### Setup

To contribute, clone the OceanStream repository's dev branch, which contains the latest development changes:

    git clone -b dev https://github.com/OceanStreamIO/oceanstream.git

Navigate to the oceanstream directory:

    cd oceanstream

Now, you can install dependencies, run tests, or start development! Direct all pull requests to the dev branch.

### Using Conda/Mamba

- **Create a virtual environment to use the package:**
    ```bash
    conda env create -f environment.yml
    ```
This command creates an environment named `oceanstream`. To activate it:
    ```
    conda activate oceanstream
    ```
- **Create a virtual environment to contribute to the package:**
    ```bash
    conda env create -f environment-dev.yml
    ```
Again, activate the environment with:
    ```
    conda activate oceanstream
    ```
### Using Pip

1. **Create a Virtual Environment:**
    ```bash
    python -m venv .venv
    ```
This command creates a virtual environment in the `.venv` directory using Python 3.9.

- **Example using `pyenv`:**

    - Install Python 3.9 with `pyenv`:
        ```bash
        pyenv install 3.9
        ```
    - Navigate to your project directory:
        ```bash
        cd /path/to/your/project_directory
        ```
    - Create the virtual environment using the Python 3.9 executable managed by `pyenv`:
        ```bash
        ~/.pyenv/versions/3.11/bin/python -m venv .venv
        ```
2. **Activate the Virtual Environment:**

- On Linux and MacOS:
    ```bash
    source .venv/bin/activate
    ```
- On Windows:
    ```bash
    .venv\Scripts\activate.bat  # In cmd.exe
    .venv\Scripts\Activate.ps1  # In PowerShell
    ```
3. **Install the Dependencies:**

- To use the package:
    ```bash
    pip install -r requirements.txt
    ```
- To contribute to the package:
    ```bash
    pip install -r requirements-dev.txt
    ```
## Running Pre-Commit Locally

### Installation

Install the pre-commit tool using pip:

    pip install pre-commit

### Installing the Git Hook Scripts

Navigate to your repository where the `.pre-commit-config.yaml` file is located. Install the Git hook scripts with:

    pre-commit install

### Running Pre-Commit

Pre-commit will now run automatically every time you attempt to make a commit. If any hooks fail, the commit will be blocked, and you'll be prompted to fix the issues before committing again.

To manually run all hooks against all the files, use:

    pre-commit run --all-files

### Updating Hooks

To update your hooks to the latest versions, use:

    pre-commit autoupdate

### Skipping Hooks

To bypass the hooks for a particular commit, use the `-n` or `--no-verify` option:

    git commit -m "Your commit message" -n

### Uninstalling Pre-Commit

To uninstall the Git hook scripts, use:

    pre-commit uninstall

## Building Documentation Locally

### Prerequisites

Ensure Python and pip are installed on your system.

### Steps

1. **Install Sphinx:**

    pip install sphinx

2. **Navigate to Your Project's docs Directory:**

    cd /path/to/your/docs_directory

3. **Build the Documentation:**

- For Linux/Mac:

    make html

- For Windows:

    make.bat html

This command will generate the HTML documentation in the `build/html` directory within your docs folder.
