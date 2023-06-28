
# Search Algorithms Used in Artificial Intelligence  

## Overview

Search is an integral part of AI. Here is implemented several graph search algorithms with the goal of solving bi-directional and tri-directional search.


### The Files

| File | Description |
| ----:| :-----------|
|**__search_algorithms.py__** | Search algorithm such as _PriorityQueue_, _Breadth First Search_, _Uniform Cost Search_, _A* Search_, _Bi-directional Search_, _Tri-directional Search_ |
|**_search_tests.py_** | Simple unit tests to validate your searches validity and number of nodes explored |
|**_search_unit_tests.py_** | More detailed tests that run searches from all possible pairs of nodes in the graph |
|**_romania_graph.pickle_** | Serialized graph files for Romania. |


### Implemented Search Amgorithms
Some methods implemented in search_algorithm.py to run optimization and search algorithms.

#### Priority Queue (Class)

A data structure representing a queue where elements are served based on their priority. Higher priority elements are served before lower priority elements. If two elements have the same priority, they are served in the order they were added to the queue.

#### breadth_first_search (method)

Returns a list of nodes representing the path from a given start node to a given end node.

#### uniform_cost_search (method)

Returns the best path as a list of nodes from the start node to the goal node, considering the cost of each path.

#### A* search (method)

Uses the A* algorithm with Euclidean distance as the heuristic to find the best path from the start node to the goal node.

#### bidirectional_ucs (method)

Performs a bidirectional uniform-cost search, starting from both the start and end states, and expands nodes until the two searches meet.

#### bidirectional_a_star (method)

Calculates a heuristic for both the start-to-goal search and the goal-to-start search, and finds the best path using the bidirectional A* algorithm.

#### tridirectional_search (method)

Implements a tridirectional UCS search by starting from each goal node and expanding nodes until two of the three searches meet.

#### tridirectional_upgraded (method)

An upgraded version of tridirectional search designed to improve performance by exploring fewer nodes during the search, resulting in reduced runtime.
