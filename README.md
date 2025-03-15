# ENPM 661 Project 1 
To implement BFS and Dijkstra on a custom map.\
Submission by:- Anirudh Swarankar 121150653

## Implementation Notes
Both the algorithms BFS and Dijkstra are implemented in same file having different functions. Better would be to create a base graphsearch class and then inherit but that's out of scope for this. 
Custom data containers are implemented to aid the graph search for example Node container. The Node container represents each node in graph it holds cost to come, visited status and parent node in it. By encapsulating it backtracking is easier. Since Heapq is used for min priority queue a custom functions are implemented for comparison and hash creation. 

## Map

![map](map.png)

## How to run ?
The program takes multiple inputs from the used like the algorithm to run either BFS or DIJKSTRA and then start and goal point. Note for visualization all the dimensions are internally multiplied by a multiplier(default 10), but the user input has to be in original canvas size i.e. 180x50.\

To run the code 
```bash
python3 graphsearch_Anirudh_Swarankar.py
```

Coordinated of start and end point in videos 
start(100,250) and goal(1750,450).

## External libraries used
- Numpy
- Heapq : for min priority queue in Dijkstra
- Deque : Double queue for BFS. 
- Time : to time program sections
- OpenCV : CV used for visualization and mat