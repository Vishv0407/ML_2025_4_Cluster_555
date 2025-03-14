# How to setup virtual python environment ?

This project uses a Python virtual environment (`venv`) to manage dependencies. Follow these steps to set up your local environment.

## Prerequisites

- Python 3.x installed (check with `python3 --version` ).
- Git installed to clone the repository.

## Setup Instructions

1. **Clone the Repository**

   ```bash
   git clone https://github.com/Vishv0407/ML_2025_4_Cluster_555
   cd https://github.com/Vishv0407/ML_2025_4_Cluster_555
    ```
  
2. **Enter the `Codes` directory**

   ```bash
   cd Codes
   ```

3. **Create the virutal environment**

   ```bash
   python3 -m venv .venv
   ```

4. **Activate the virtual environment**
  
- On Windows:

   ```bash
   .venv\Scripts\activate
   ```

- On macOS and Linux:

   ```bash
   source .venv/bin/activate
   ```

### Note

If you are not able to setup virtual environment and getting the following error:

```bash
Fatal Python error: Failed to import encodings module
Python runtime state: core initialized
ModuleNotFoundError: No module named 'encodings'

Current thread 0x00007f13bebeff40 (most recent call first):
  <no Python frame>
```

Then, you can try the following solution:

``` bash
unset PYTHONPATH
unset PYTHONHOME
```

and follow from step.3 again.
