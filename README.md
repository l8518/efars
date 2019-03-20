# EFARS - Evaluation Framework for Architectures of Recommender Systems
This repository is a prottypical implementation the the Evaluation Framework for Architectures of Recommender Systems (EFARS), that was designed during writing my Bachelor Thesis "Performance Benchmark for Recommender System Architectures"

## Installation
EFARS is implemented in Python to leverage on popular data science frameworks, such as numpy, pandas and matplotlib.
For running this EFARS, you will need to:
- install Python 3.6.X on your system (Python 3.6.3 and 3.6.8 were used for development)
    - Linux/Ubuntu XX: `sudo apt-get install python3.6`
- install docker and docker-compose in the most recent version. For development the following builds were used:
    - docker-compose version 1.18.0, build 8dd22a9
    - docker version 18.09.2, build 6247962
- install the framework with `python setup.py install`:
    - you might need to use `python3` instead of `python` due to your system's configuration
    - you might need to `sudo` the command above, e.g., `sudo python3 setup.py install`
- configure EFARS approprieatly

### Trouble Shooting Installation
- macOS High Sierra and above might require you to disable fork safety for python:
    - `export OBJC_DISABLE_INITIALIZE_FORK_SAFETY=YES`
- Linux/macOS/Windows 10:
    - Missing setuptools? Install pip / pip3
- Linux Missing TKinter
    - Install tkinter for Python3: `sudo apt-get install python3-tk`

## Setup / Further Configuration
### Configuration of EFARS
You can configure this framework implementation by using enviroment variables or a .env file in the root folder. The following variables exist and can be used:
- `COMPOSE_FILE` : Allows you to choose which compose file should be used for running the evaluation.
- `DOCKER_SOCK` : Defines the connection to docker daemon, which is required by the Python docker SDK
    - Linux/macOS: `DOCKER_SOCK=unix://var/run/docker.sock`
    - Windows: `DOCKER_SOCK=npipe:////./pipe/docker_engine`

### Configuration of Docker for macOS/Windows
Please note, that macOS and Windows need additionall configuration, you have to:
- Allocate enough memory and CPUs 
- Windows requires you to share drive

## Hardware Requirements for Each Datasets
- The respository is configured to run the scientific MovieLens 20m dataset or the much smaller MovieLens Latest Dataset (check [GroupLens website]( https://grouplens.org/datasets/movielens/) for more information ), thus you might need to fulfill, depending on which dataset you use, the following hardware specs:
    - MovieLens Latest Dataset:
        - 16 GB of RAM (tested)
        - A suffiecient CPU (tested on a AMD Ryzen 5 2600, 6x 3.40GHz)
    - MovieLens 20m Dataset:
        - 80 GB of RAM (tested)
        - Sufficient CPU Cores

# Evaluation Enviroment for MovieLens 20m
## RAM
| RAM           | Size           | Total Width | Data Width  |
| ------------- |:--------------:| -----------:| ----------: |
| DIMM 0        |       16384 MB | 64 bits     | 64 bits     |
| DIMM 1        |       16384 MB | 64 bits     | 64 bits     |
| DIMM 2        |       16384 MB | 64 bits     | 64 bits     |
| DIMM 3        |       16384 MB | 64 bits     | 64 bits     |
| DIMM 4        |       14488 MB | 64 bits     | 64 bits     |
|               |                |             |             |
| Total:        | 80024MB        | -           | -           |

## CPU
| RAM           | Size           | Total Width | Data Width  |
| ------------- |:--------------:| -----------:| ----------: |
| DIMM 0        |       16384 MB | 64 bits     | 64 bits     |
| DIMM 1        |       16384 MB | 64 bits     | 64 bits     |
| DIMM 2        |       16384 MB | 64 bits     | 64 bits     |
| DIMM 3        |       16384 MB | 64 bits     | 64 bits     |
| DIMM 4        |       14488 MB | 64 bits     | 64 bits     |
|               |                |             |             |
| Total:        | 80024MB        | -           | -           |

## Docker Resources


## Enviroment Vars
# Configuration for Running MovieLens 20m
```console
# Either two compose files
# COMPOSE_FILE=docker-compose.mysql.yml
# COMPOSE_FILE=docker-compose.yml
DOCKER_CONTAINERS="basic-recommender,database"
RUN_PROVISIONS_PER_TICK=10000
RUN_CONCURRENT_FETCHES=40
RUN_CONCURRENT_PROVISIONS=40
RUN_MEASURE_SKIP_STEPS=20
RUN_FETCH_SKIP_STEPS=20
RUN_TICK_DELAY=2
RUN_WARM_UP_DELAY=40
RUN_N_REPETITION=2
```

# Misc
