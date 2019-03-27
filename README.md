# EFARS - Evaluation Framework for Architectures of Recommender Systems
This repository is a prototypical implementation the Evaluation Framework for Architectures of Recommender Systems (EFARS), that was designed during writing my Bachelor Thesis "Performance Benchmark for Recommender System Architectures"

## Installation
EFARS is implemented in Python to leverage popular data science frameworks, such as numpy, pandas, and matplotlib.
For running this EFARS, you will need to:
- Install Python 3.6.X on your system (Python 3.6.3 and 3.6.8 were used for development)
    - Linux/Ubuntu XX: `sudo apt-get install python3.6`
- Install docker and docker-compose in the most recent version. For development the following builds were used:
    - docker-compose version 1.18.0, build 8dd22a9
    - docker version 18.09.2, build 6247962
- Install the framework with `python setup.py install`:
    - You might need to use `python3` instead of `python` due to your system's configuration
    - You might need to `sudo` the command above, e.g., `sudo python3 setup.py install`
- Configure EFARS appropriately

### Trouble Shooting Installation
- macOS High Sierra and above might require you to disable fork safety for python:
    - `export OBJC_DISABLE_INITIALIZE_FORK_SAFETY=YES`
- Linux/macOS/Windows 10:
    - Missing setuptools? Install pip / pip3
- Linux Missing Tkinter
    - Install Tkinter for Python3: `sudo apt-get install python3-tk`

## Setup / Further Configuration
### Configuration of EFARS
You can configure this framework implementation by using environment variables or a .env file in the root folder. The following variables exist and can be used:
- `COMPOSE_FILE` : Allows you to choose which compose file should be used for running the evaluation.
    - Default: `COMPOSE_FILE=docker-compose.yml`
- `DOCKER_SOCK` : Defines the connection to docker daemon, which is required by the Python docker SDK. Default are set as follows:
    - Default for Linux/macOS: `DOCKER_SOCK=unix://var/run/docker.sock`
    - Default for Windows: `DOCKER_SOCK=npipe:////./pipe/docker_engine`
- `DOCKER_CONTAINERS` : Specifies the containers, that are being monitored during an evaluation.
    - Default: `DOCKER_CONTAINERS=basic-recommender,database`
- `DATASET_ADAPTER` : Specifies which Adapter is used to split the dataset into test and training data.
    - Default: `DATASET_ADAPTER=MovieLensDatasetAdapter`
 - `RUN_WARM_UP_DELAY` : Defines the duration in seconds that will be waited until the first ticks are initiated.
    - Default: `RUN_WARM_UP_DELAY=30`
 - `RUN_TRAINING_FILE` : Defines the path of the training file being used of the provider component.
    - Default: `RUN_TRAINING_FILE=./data/ratings/train.csv`
 - `RUN_N_REPETITION` : Defines the number of repetitions for each configuration/run
    - Default: `RUN_N_REPETITION=1`
 - `RUN_PROVISIONS_PER_TICK` : Defines the number of events streamed to the recommender system.
    - Default: `RUN_PROVISIONS_PER_TICK=2000`
 - `RUN_PROVIDER_ADAPTER` : Defines the adapter for streaming events to the recommender system.
    - Default: `RUN_PROVIDER_ADAPTER=BasicRecommenderProviderAdapter`
 - `RUN_RECEIVER_ADAPTER` : Defines the adapter for querying the recommendations of the recommender system.
    - Default: `RUN_RECEIVER_ADAPTER=BasicRecommenderReceiverAdapter`
- `RUN_MEASURE_SKIP_STEPS` : Defines the number of ticks being skipped until hardware measurement is taken.
    - Default: `RUN_MEASURE_SKIP_STEPS=10`
- `RUN_FETCH_SKIP_STEPS` : Defines the number of ticks being skipped until recommendations of the recommender system are fetched.
    - Default: `RUN_FETCH_SKIP_STEPS=10`
- `RUN_TICK_DELAY` : Defines the tick delay in seconds, which is the minimal duration of a tick. Can be longer if the operations per tick take longer than the duration specified with this parameter.
    - Default: `RUN_TICK_DELAY=2`
- `RUN_CONCURRENT_FETCHES` : Defines the number of processes being spawned to query for recommendations.
    - Default: `RUN_CONCURRENT_FETCHES=[Calculated Based on the CPU count]`
- `RUN_CONCURRENT_PROVISIONS` : Defines the number of processes being spawned to stream the data.
    - Default: `RUN_CONCURRENT_PROVISIONS=[Calculated Based on the CPU count]`
- `RUN_FETCHES_RATING_N` : Defines the number of recommended items which are used for querying the recommender system.
    - Default: `RUN_FETCHES_RATING_N=20`

### Configuration of Docker for macOS/Windows
Please note, that macOS and Windows need additional configuration of Docker, you have to permit Docker to:
- Allocate enough memory and CPUs 
- Access the shared drive on Windows

## Hardware Requirements for MovieLens Datasets
- The repository is configured to run the scientific MovieLens 20m dataset or the much smaller MovieLens Latest Dataset (check [GroupLens website]( https://grouplens.org/datasets/movielens/) for more information ). Thus, you might need to fulfill, depending on which dataset you use, the following hardware specs:
    - MovieLens Latest Dataset:
        - 16 GB of RAM (tested)
        - A sufficient CPU (tested on a AMD Ryzen 5 2600, 6x 3.40GHz)
    - MovieLens 20m Dataset:
        - 80 GB of RAM (tested)
        - Sufficient CPU Cores

# Running an Evaluation
1. Start the application with `python efars`
2. Plan the run and select the appropriate dataset
3. Run the current docker-compose configuration
    - The EFARSPrompt will ask to provide a name
4. Wait till configuration is evaluated, and uses assessment
    - Output of diagrams and CSV files will be stored under `./data/evaluation_plots`.

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
