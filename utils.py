"""
utils.py
Purpose: Contains the Dijkstra algorithm implementation for finding the shortest path.
This module handles all pathfinding logic for the traffic predictor.
"""

import heapq


def dijkstra_shortest_path(graph, start, end):
    """
    Implements Dijkstra's algorithm to find the shortest path between two locations.
    
    Args:
        graph: Dictionary representing the road network (adjacency list)
        start: Starting location (string)
        end: Destination location (string)
    
    Returns:
        tuple: (path as list of locations, total weight/distance)
               Returns (None, None) if no path exists
    
    Algorithm:
        1. Initialize distances to infinity for all nodes except start (0)
        2. Use a priority queue to always process the node with minimum distance
        3. For each node, update distances to its neighbors
        4. Track the path by storing predecessors
        5. Reconstruct path from end to start using predecessors
    """
    
    # Validate inputs
    if start not in graph:
        return None, None
    if end not in graph:
        return None, None
    if start == end:
        return [start], 0
    
    # Initialize data structures
    distances = {node: float('infinity') for node in graph}
    distances[start] = 0
    
    # Priority queue: (distance, node)
    pq = [(0, start)]
    
    # Track predecessors to reconstruct path
    predecessors = {node: None for node in graph}
    
    # Track visited nodes
    visited = set()
    
    # Main Dijkstra loop
    while pq:
        current_distance, current_node = heapq.heappop(pq)
        
        # Skip if already visited
        if current_node in visited:
            continue
        
        visited.add(current_node)
        
        # If we reached the destination, we can stop
        if current_node == end:
            break
        
        # Skip if we found a better path already
        if current_distance > distances[current_node]:
            continue
        
        # Check all neighbors
        for neighbor, weight in graph[current_node].items():
            distance = current_distance + weight
            
            # If we found a shorter path to the neighbor
            if distance < distances[neighbor]:
                distances[neighbor] = distance
                predecessors[neighbor] = current_node
                heapq.heappush(pq, (distance, neighbor))
    
    # Check if destination is reachable
    if distances[end] == float('infinity'):
        return None, None
    
    # Reconstruct path from end to start
    path = []
    current = end
    while current is not None:
        path.append(current)
        current = predecessors[current]
    
    # Reverse to get path from start to end
    path.reverse()
    
    return path, distances[end]


def get_path_edges(path):
    """
    Converts a path (list of nodes) into a list of edges.
    
    Args:
        path: List of locations representing the path
    
    Returns:
        List of tuples representing edges: [(node1, node2), ...]
    """
    if not path or len(path) < 2:
        return []
    
    edges = []
    for i in range(len(path) - 1):
        edges.append((path[i], path[i + 1]))
    
    return edges


def format_path_display(path, total_weight):
    """
    Formats the path and weight for display in the UI.
    
    Args:
        path: List of locations
        total_weight: Total travel time/distance
    
    Returns:
        Formatted string for display
    """
    if path is None:
        return "❌ No route available between selected locations."
    
    path_str = " → ".join(path)
    return f"🛣 *Route:* {path_str}\n\n⏱ *Estimated Travel Time:* {total_weight:.1f} minutes"


def calculate_alternative_routes(graph, start, end, k=3):
    """
    Placeholder for future enhancement: Calculate k alternative routes.
    This could be implemented using Yen's K-shortest paths algorithm.
    
    Args:
        graph: Road network
        start: Starting location
        end: Destination location
        k: Number of alternative routes to find
    
    Returns:
        List of (path, weight) tuples
    """
    # For now, just return the single shortest path
    path, weight = dijkstra_shortest_path(graph, start, end)
    if path:
        return [(path, weight)]
    return []