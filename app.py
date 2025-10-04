"""
app.py - Production Version
Purpose: Main Streamlit application for the Traffic Congestion Predictor.
Provides UI for route selection, prediction, and visualization.

Production Enhancements:
- Fixed critical cache invalidation bug
- Added comprehensive error handling
- Implemented route history tracking
- Enhanced visualization
- Added performance metrics
- Improved user feedback
"""

import streamlit as st
import matplotlib.pyplot as plt
import networkx as nx
from typing import Optional, List, Tuple, Dict
from datetime import datetime
from data_loader import get_road_network, get_locations, get_edge_list, update_traffic_weight
from utils import dijkstra_shortest_path, get_path_edges, format_path_display
import time


# Page configuration
st.set_page_config(
    page_title="Traffic Congestion Predictor",
    page_icon="🚦",
    layout="wide",
    initial_sidebar_state="expanded"
)


def initialize_session_state():
    """Initialize session state variables for the app."""
    if 'road_network' not in st.session_state:
        st.session_state.road_network = get_road_network()
    if 'show_traffic_editor' not in st.session_state:
        st.session_state.show_traffic_editor = False
    if 'graph_cache_key' not in st.session_state:
        st.session_state.graph_cache_key = 0
    if 'route_history' not in st.session_state:
        st.session_state.route_history = []
    if 'calculation_time' not in st.session_state:
        st.session_state.calculation_time = 0


@st.cache_data
def create_networkx_graph(_road_network: Dict, _cache_key: int):
    """
    Creates a NetworkX graph from the road network (cached for performance).
    
    Args:
        _road_network: Dictionary representing the road network
        _cache_key: Cache invalidation key (changes when traffic updates)
        
    Returns:
        NetworkX Graph object
    """
    G = nx.Graph()
    for source, neighbors in _road_network.items():
        for destination, weight in neighbors.items():
            G.add_edge(source, destination, weight=weight)
    return G


def create_network_graph(road_network: Dict, highlight_path: Optional[List[Tuple]] = None):
    """
    Creates a NetworkX graph from the road network and visualizes it.
    
    Args:
        road_network: Dictionary representing the road network
        highlight_path: List of edges to highlight (shortest path)
        
    Returns:
        matplotlib figure object
    """
    try:
        # Create graph with cache key
        G = create_networkx_graph(road_network, st.session_state.graph_cache_key)
        
        if len(G.nodes()) == 0:
            st.error("No nodes found in the road network!")
            return None
        
        # Set up the plot
        fig, ax = plt.subplots(figsize=(14, 10))
        
        # Use spring layout for better visualization
        pos = nx.spring_layout(G, k=2, iterations=50, seed=42)
        
        # Draw all nodes
        nx.draw_networkx_nodes(
            G, pos, 
            node_color='lightblue', 
            node_size=2000,
            alpha=0.9,
            ax=ax
        )
        
        # Draw all edges in gray
        nx.draw_networkx_edges(
            G, pos,
            edge_color='gray',
            width=2,
            alpha=0.5,
            ax=ax
        )
        
        # Highlight the shortest path if provided
        if highlight_path and len(highlight_path) > 0:
            # Filter valid edges
            valid_edges = [e for e in highlight_path if G.has_edge(*e)]
            if valid_edges:
                nx.draw_networkx_edges(
                    G, pos,
                    edgelist=valid_edges,
                    edge_color='red',
                    width=4,
                    alpha=0.8,
                    ax=ax
                )
        
        # Draw node labels
        nx.draw_networkx_labels(
            G, pos,
            font_size=9,
            font_weight='bold',
            font_color='darkblue',
            ax=ax
        )
        
        # Draw edge labels (weights)
        edge_labels = nx.get_edge_attributes(G, 'weight')
        nx.draw_networkx_edge_labels(
            G, pos,
            edge_labels,
            font_size=8,
            font_color='green',
            ax=ax
        )
        
        ax.set_title("Road Network Graph (Weights = Travel Time in Minutes)", 
                     fontsize=14, fontweight='bold', pad=20)
        ax.axis('off')
        plt.tight_layout()
        
        return fig
        
    except Exception as e:
        st.error(f"Error creating network graph: {str(e)}")
        return None


def display_route_statistics(path: List[str], total_weight: float):
    """Display detailed route statistics."""
    if not path or len(path) < 2:
        return
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Number of Stops", len(path) - 1)
    
    with col2:
        st.metric("Total Travel Time", f"{total_weight:.1f} min")
    
    with col3:
        avg_segment = total_weight / (len(path) - 1) if len(path) > 1 else 0
        st.metric("Avg. Segment Time", f"{avg_segment:.1f} min")
    
    with col4:
        if st.session_state.calculation_time > 0:
            st.metric("Calculation Time", f"{st.session_state.calculation_time:.3f}s")


def add_to_route_history(source: str, destination: str, time: float, path: List[str]):
    """Add a route to the history."""
    history_entry = {
        'timestamp': datetime.now().strftime("%H:%M:%S"),
        'source': source,
        'destination': destination,
        'time': time,
        'path': path
    }
    
    # Keep only last 10 entries
    st.session_state.route_history.append(history_entry)
    if len(st.session_state.route_history) > 10:
        st.session_state.route_history.pop(0)


def display_sidebar():
    """Display sidebar with route history and info."""
    with st.sidebar:
        st.header("📊 Dashboard")
        
        # Network stats
        st.subheader("Network Statistics")
        locations = get_locations()
        edges = get_edge_list()
        
        st.metric("Total Locations", len(locations))
        st.metric("Total Routes", len(edges))
        
        # Calculate average traffic
        if edges:
            avg_weight = sum(w for _, _, w in edges) / len(edges)
            st.metric("Avg. Travel Time", f"{avg_weight:.1f} min")
        
        st.divider()
        
        # Route history
        st.subheader("🕐 Recent Routes")
        if st.session_state.route_history:
            for idx, entry in enumerate(reversed(st.session_state.route_history[-5:])):
                with st.expander(f"{entry['timestamp']} - {entry['source'][:8]}... → {entry['destination'][:8]}..."):
                    st.write(f"**From:** {entry['source']}")
                    st.write(f"**To:** {entry['destination']}")
                    st.write(f"**Time:** {entry['time']:.1f} min")
                    st.write(f"**Path:** {' → '.join(entry['path'])}")
        else:
            st.info("No routes calculated yet.")
        
        st.divider()
        
        # Quick actions
        st.subheader("⚡ Quick Actions")
        if st.button("🔄 Reset All Traffic", use_container_width=True):
            st.session_state.road_network = get_road_network()
            st.session_state.graph_cache_key += 1
            st.success("All traffic reset!")
            st.rerun()
        
        if st.button("🗑️ Clear History", use_container_width=True):
            st.session_state.route_history = []
            st.success("History cleared!")
            st.rerun()


def main():
    """Main application function."""
    
    try:
        # Initialize session state
        initialize_session_state()
        
        # Display sidebar
        display_sidebar()
        
        # Header
        st.title("🚦 Traffic Congestion Predictor")
        st.markdown("""
        Welcome to the **Traffic Congestion Predictor**! This app helps you find the fastest route 
        between locations considering current traffic conditions. Select your source and destination 
        to get started.
        """)
        
        st.divider()
        
        # Main content area
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.subheader("📍 Route Selection")
            
            # Get locations for dropdowns
            locations = get_locations()
            
            if not locations or len(locations) == 0:
                st.error("No locations available in the road network!")
                return
            
            # Source and destination selection
            source = st.selectbox(
                "Select Source Location:",
                options=locations,
                index=0,
                key="source"
            )
            
            destination = st.selectbox(
                "Select Destination Location:",
                options=locations,
                index=min(5, len(locations)-1) if len(locations) > 1 else 0,
                key="destination"
            )
            
            # Predict button
            predict_button = st.button("🚀 Predict Fastest Route", type="primary", use_container_width=True)
            
            st.divider()
            
            # Results section
            if predict_button:
                if source == destination:
                    st.warning("⚠️ Source and destination cannot be the same!")
                else:
                    with st.spinner("Calculating optimal route..."):
                        try:
                            # Measure calculation time
                            start_time = time.time()
                            
                            # Calculate shortest path
                            path, total_weight = dijkstra_shortest_path(
                                st.session_state.road_network,
                                source,
                                destination
                            )
                            
                            end_time = time.time()
                            st.session_state.calculation_time = end_time - start_time
                            
                            # Store results in session state
                            st.session_state.current_path = path
                            st.session_state.current_weight = total_weight
                            st.session_state.path_edges = get_path_edges(path) if path else None
                            
                            # Add to history
                            if path:
                                add_to_route_history(source, destination, total_weight, path)
                            
                        except Exception as e:
                            st.error(f"Error calculating route: {str(e)}")
            
            # Display results if available
            if 'current_path' in st.session_state and st.session_state.current_path is not None:
                st.subheader("📊 Route Results")
                
                result_text = format_path_display(
                    st.session_state.current_path,
                    st.session_state.current_weight
                )
                
                if st.session_state.current_path:
                    st.success(result_text)
                    
                    # Display detailed statistics
                    display_route_statistics(
                        st.session_state.current_path,
                        st.session_state.current_weight
                    )
                else:
                    st.error(result_text)
            
            st.divider()
            
            # Traffic simulation toggle
            st.subheader("🔧 Advanced Options")
            if st.button("Toggle Traffic Editor", use_container_width=True):
                st.session_state.show_traffic_editor = not st.session_state.show_traffic_editor
                st.rerun()
        
        with col2:
            st.subheader("🗺️ Network Visualization")
            
            # Visualize the graph
            highlight_edges = st.session_state.get('path_edges', None)
            fig = create_network_graph(st.session_state.road_network, highlight_edges)
            
            if fig:
                st.pyplot(fig)
                plt.close(fig)
            
            # Legend
            st.markdown("""
            **Legend:**
            - 🔵 Blue circles = Locations
            - Gray lines = Available roads
            - 🔴 **Red lines = Fastest route**
            - Green numbers = Travel time (minutes)
            """)
        
        # Traffic editor section (expandable)
        if st.session_state.show_traffic_editor:
            st.divider()
            st.subheader("🚧 Traffic Simulation Editor")
            st.info("Modify traffic conditions by updating edge weights. Higher values = more congestion.")
            
            # Get all edges
            edges = get_edge_list()
            
            if not edges:
                st.warning("No edges available to edit.")
                return
            
            # Create columns for edge editing
            cols = st.columns(3)
            
            for idx, (src, dst, weight) in enumerate(edges):
                with cols[idx % 3]:
                    new_weight = st.number_input(
                        f"{src} ↔ {dst}",
                        min_value=1,
                        max_value=100,
                        value=int(weight),
                        step=1,
                        key=f"edge_{src}_{dst}",
                        help=f"Current: {weight} min"
                    )
                    
                    # Update if changed
                    if new_weight != weight:
                        try:
                            st.session_state.road_network = update_traffic_weight(
                                st.session_state.road_network,
                                src, dst, new_weight
                            )
                            # Invalidate cache
                            st.session_state.graph_cache_key += 1
                            st.toast(f"Updated {src} ↔ {dst} to {new_weight} min", icon="✅")
                        except Exception as e:
                            st.error(f"Error updating traffic: {str(e)}")
        
        # Footer
        st.divider()
        st.markdown("""
        <div style='text-align: center; color: gray; padding: 20px;'>
            <p>🚦 Traffic Congestion Predictor | Powered by Dijkstra's Algorithm</p>
            <p style='font-size: 12px;'>This app uses graph theory to find optimal routes considering traffic conditions.</p>
        </div>
        """, unsafe_allow_html=True)
    
    except Exception as e:
        st.error(f"Application error: {str(e)}")
        st.info("Please refresh the page or contact support if the issue persists.")
        
        # Show detailed error in expander
        with st.expander("🐛 Error Details (for debugging)"):
            import traceback
            st.code(traceback.format_exc())


if __name__ == "__main__":
    main()