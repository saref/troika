# Troika Algorithm

Troika is an algorithm for solving the Clique Partitioning problem, inspired by the Bayan algorithm which was originally designed to solve Modularity Maximization. This new algorithm, is capable of providing an approximation of the maximum objective value with a guarantee of proximity. This algorithm is theoretically grounded by the Integer Programming (IP) formulation of the Clique Partitioning problem and relies on an exact branch-and-cut scheme for solving the NP-complete optimization problem.

## Project Structure

Below is the file and directory structure of this project along with a brief description of each item:

```
troika/
│
├── troika/ - Source code of Troika
│
├── corr40-7.gml - A sample benchmark instance from CP-Lib
│
├── EstimateUB.so - Python wrapper for the Best Partition algorithm implementation
│
├── example.ipynb - Sample usage of the Troika algorithm
│
├── README.md - Project overview and setup instructions
│
└── requirements.txt - The list of project dependencies
```

## Dependencies

This project has the following dependencies:

- networkx>=3.4.2
- pycombo>=0.1.8
- gurobipy>=11.0.3
- numpy>=2.2.6
- joblib>=1.5.0
- scipy>=1.15.3

Note that pycombo 0.1.8 requires Python version >= 3.7, < 4.0 and therefore the algorithm has the same python version requirement.

The dependencies above can be installed using the following command.
```
# Navigate to the project directory
cd path/to/the/project

python -m pip install -r requirements.txt
```

### Gurobi Installation with Free Academic License

The algorithm requires Gurobi Optimizer. Gurobi is a commercial software, but it can be registered with a free academic license if the user is affiliated with an academic institution. Due to the restrictions of Gurobi, Troika requires a (free academic) Gurobi license for processing any graph with more than sqrt(2000) ≈ 44 nodes.

Follow these five steps to install Gurobi with a free academic license:

1. Download and install Python from [the official Python website](https://www.python.org/downloads/).
2. Register for an account on [the Gurobi registration page](https://pages.gurobi.com/registration) to be able to request a free academic license for using Gurobi (if you are affiliated with an academic institution).
3. Install Gurobi (version >= 11.0.3, latest version is recommended) into Python ([using either conda or pip](https://support.gurobi.com/hc/en-us/articles/360044290292)) using the following commands:
- Using Conda (recommended for Anaconda users):
```
conda config --add channels http://conda.anaconda.org/gurobi
conda install -c gurobi gurobi
```
- Using pip (alternative method):
```
python -m pip install gurobipy
```
4. Request a Named-User Academic License from [the Gurobi academic license page](https://www.gurobi.com/downloads/end-user-license-agreement-academic/) after reading and agreeing to Gurobi's End User License Agreement.
5. Install the license on your computer following the [instructions given on the Gurobi license page](https://support.gurobi.com/hc/en-us/articles/360059842732).

For detailed installation instructions, refer to the [Gurobi's Quick Start Guides](https://support.gurobi.com/hc/en-us/articles/14799677517585-Getting-Started-with-Gurobi-Optimizer).

## Usage

You can use Troika directly in the project directory as follows:

```python
import networkx as nx
from troika import troika

# Create or load your networkx (undirected and weighted) graph 
graph = nx.Graph([
    (0, 1, {'weight': 0.5}),
    (0, 2, {'weight': 2.2}),
    (0, 3, {'weight': -0.2}),
    (1, 2, {'weight': -0.8}),
    (2, 3, {'weight': 1.7})])

# Run the Troika algorithm
cp_objective_value, optimality_gap, partition, modeling_time, solve_time = troika(graph, global_threshold=0.001, time_allowed=60)
```

#### Parameters and acceptable input
- `graph`: Input graph should be an undirected weighted networkx graph. The graph must have edge attribute "weight" to store edge weights.
- `global_threshold` (optional, default = 0.001): The acceptable global optimality gap for the algorithm to terminate. If Troika finds a solution with an objective value within the specified threshold of the optimal solution, it stops the search and returns the found solution. For example, setting the threshold to 0.01 means Troika will stop when it finds a global solution within 1% of the maximum objective value for that graph.
- `time_allowed` (optional, default = 600 seconds): The maximum allowed execution time in seconds for Troika to search for a solution after the optimization model is built (formulated). Shortly after this time limit is reached, the algorithm will terminate and returns the best solution found so far, even if the optimality gap threshold is not met.

#### Returns
- `objective_value`: The objective value of the returned partition.
- `optimality_gap`: The upper bound for the percentage difference between the objective value of the returned partition and the maximum objective value.
- `partition`: A nested list describing the communities.
- `modeling_time`: The seconds taken for pre-processing the input and formulating the optimization model.
- `solve_time`: The seconds taken for solving the optimization model using Troika.

## References
- [Bayan code](https://github.com/saref/bayan)
- [Bayan project](https://bayanproject.github.io/)
- [CP-Lib Benchmark Instances](https://github.com/MMSorensen/CP-Lib)
- [Combo code](https://github.com/Alexander-Belyi/Combo)
- [Best partition](https://github.com/Alexander-Belyi/best-partition)
