"""
data_loader.py
Purpose: Contains predefined road network data with locations and traffic-weighted edges.
This module provides the graph structure and location list for the traffic predictor.
"""

def get_road_network():
    """
    Returns a predefined road network as a weighted adjacency list graph.
    
    Structure:
    {
        'Location': {
            'Neighbor': weight (traffic congestion score)
        }
    }
    
    Weight interpretation:
    - Lower weight = Less traffic, faster route
    - Higher weight = More traffic, slower route
    - Weight represents approximate travel time in minutes
    """
    
    # Define the road network with 10 major city locations
    road_network = {
        'Downtown': {
            'Central Park': 5,
            'Shopping Mall': 8,
            'Airport': 25
        },
        'Central Park': {
            'Downtown': 5,
            'Shopping Mall': 3,
            'University': 7,
            'Hospital': 6
        },
        'Shopping Mall': {
            'Downtown': 8,
            'Central Park': 3,
            'Business District': 4,
            'University': 5
        },
        'University': {
            'Central Park': 7,
            'Shopping Mall': 5,
            'Hospital': 4,
            'Tech Hub': 6
        },
        'Hospital': {
            'Central Park': 6,
            'University': 4,
            'Residential Area': 5,
            'Airport': 15
        },
        'Business District': {
            'Shopping Mall': 4,
            'Tech Hub': 3,
            'Station': 7
        },
        'Tech Hub': {
            'University': 6,
            'Business District': 3,
            'Station': 5,
            'Industrial Zone': 8
        },
        'Station': {
            'Business District': 7,
            'Tech Hub': 5,
            'Airport': 12,
            'Industrial Zone': 6
        },
        'Industrial Zone': {
            'Tech Hub': 8,
            'Station': 6,
            'Residential Area': 9
        },
        'Residential Area': {
            'Hospital': 5,
            'Industrial Zone': 9,
            'Airport': 10
        },
        'Airport': {
            'Downtown': 25,
            'Hospital': 15,
            'Station': 12,
            'Residential Area': 10
        }
    }
    
    return road_network


def get_locations():
    """
    Returns a sorted list of all available locations in the network.
    Used for populating dropdown menus in the UI.
    """
    road_network = get_road_network()
    locations = sorted(list(road_network.keys()))
    return locations


def get_edge_list():
    """
    Returns all edges in the graph as a list of tuples.
    Format: [(source, destination, weight), ...]
    Useful for visualization purposes.
    """
    road_network = get_road_network()
    edges = []
    
    # Avoid duplicates by tracking processed edges
    processed = set()
    
    for source, neighbors in road_network.items():
        for destination, weight in neighbors.items():
            # Create a normalized edge identifier (bidirectional)
            edge_id = tuple(sorted([source, destination]))
            if edge_id not in processed:
                edges.append((source, destination, weight))
                processed.add(edge_id)
    
    return edges


def update_traffic_weight(road_network, source, destination, new_weight):
    """
    Updates the traffic weight for a specific edge in the network.
    Useful for simulating real-time traffic changes.
    
    Args:
        road_network: The current road network dictionary
        source: Starting location
        destination: Ending location
        new_weight: New traffic weight value
    
    Returns:
        Updated road network
    """
    if source in road_network and destination in road_network[source]:
        road_network[source][destination] = new_weight
    
    # Update reverse direction if it exists (bidirectional road)
    if destination in road_network and source in road_network[destination]:
        road_network[destination][source] = new_weight
    
    return road_network