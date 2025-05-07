<img src="logo.png" alt="logo" width="200"/>

# PREST

This project allows for the automatic control of an Ismatec Reglo ICC laboratory peristaltic pump.

## Table of Contents

- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [License](#license)
- [Authors](#authors)

## Features

- Control of 4 independent channels
- Time configuration in seconds
- Volume configuration in microliters
- Control of rotation direction

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/cfm-mpc/PREST.git
   ```
2. Navigate to the project directory:
   ```bash
   cd PREST
   ```
3. Create a conda environment using the `environment.yml` file:
   ```bash
   conda env create -f environment.yml
   ```
4. Activate the environment:
   ```bash
   conda activate ismatec
   ```

## Usage

Before running the main script, you need to configure the parameters in `settings.py`. Edit the `settings.py` file to set the following parameters correctly:

- `PORT`: The port to which the pump is connected.
- `SOURCE_DIR`: The folder where the source `.txt` file to read data from is stored.
- `DELAY`: The delay between the different channels.

Example configuration in `settings.py`:

```python
PORT = '/dev/ttyACM0'  # linux
BAUDRATE = 9600
SOURCE_DIR = '/path-to-project/src/'
DELAY = [10, 10, 10, 10]  # delay in seconds for each individual channel
```

Once you have configured the `settings.py` file, you can run the main script:

```python
python3 main.py
```

## License

This project is licensed under the GNU-3.0 license. See the [LICENSE](./LICENSE) file for details.

## Authors

- [Mikel Arocena](https://github.com/marocenae)
