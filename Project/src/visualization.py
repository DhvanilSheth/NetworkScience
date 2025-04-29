"""
Visualization module for community detection results.

This module provides functions for visualizing network structure,
communities, and comparison between different community detection methods.
"""

import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import plotly.graph_objects as go
from typing import Dict, List, Tuple, Set, Any, Optional, Union
import random
import math


def set_node_community(G: nx.Graph, 
                     communities: List[Set[int]]) -> Dict[int, int]:
    """
    Create a mapping of nodes to their community ID.
    
    Parameters:
    -----------
    G : nx.Graph
        The input graph
    communities : List[Set[int]]
        Communities as a list of sets of node ids
        
    Returns:
    --------
    Dict[int, int] : Mapping of node ID to community ID
    """
    node_to_community = {}
    for comm_id, comm_nodes in enumerate(communities):
        for node in comm_nodes:
            node_to_community[node] = comm_id
            
    # Set community as a node attribute
    nx.set_node_attributes(G, node_to_community, 'community')
    
    return node_to_community


def visualize_communities(G: nx.Graph, 
                        communities: List[Set[int]], 
                        title: str = "Community Structure",
                        figsize: Tuple[int, int] = (10, 8),
                        node_size: int = 100,
                        with_labels: bool = False,
                        layout: Optional[Dict] = None) -> plt.Figure:
    """
    Visualize communities in a network.
    
    Parameters:
    -----------
    G : nx.Graph
        The input graph
    communities : List[Set[int]]
        Communities as a list of sets of node ids
    title : str
        Plot title
    figsize : Tuple[int, int]
        Figure size
    node_size : int
        Size of nodes in the plot
    with_labels : bool
        Whether to show node labels
    layout : Optional[Dict]
        Pre-computed layout for node positions
        
    Returns:
    --------
    plt.Figure : The matplotlib figure
    """
    # Set community as a node attribute
    node_to_community = set_node_community(G, communities)
    
    # Create color palette
    num_communities = max(node_to_community.values()) + 1
    colors = plt.cm.rainbow(np.linspace(0, 1, num_communities))
    
    # Get or compute layout
    if layout is None:
        layout = nx.spring_layout(G, seed=42)
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Create a list of colors by node
    node_colors = [colors[node_to_community[node]] for node in G.nodes()]
    
    # Draw the graph
    nx.draw_networkx_nodes(G, layout, node_size=node_size, node_color=node_colors, alpha=0.8)
    nx.draw_networkx_edges(G, layout, alpha=0.5)
    
    if with_labels:
        nx.draw_networkx_labels(G, layout, font_size=8)
    
    # Add title and remove axes
    plt.title(title)
    plt.axis('off')
    
    return fig


def visualize_community_comparison(G: nx.Graph, 
                                 baseline_communities: List[Set[int]],
                                 enhanced_communities: List[Set[int]],
                                 figsize: Tuple[int, int] = (15, 7)) -> plt.Figure:
    """
    Compare two community detection results visually.
    
    Parameters:
    -----------
    G : nx.Graph
        The input graph
    baseline_communities : List[Set[int]]
        Baseline communities
    enhanced_communities : List[Set[int]]
        Enhanced communities
    figsize : Tuple[int, int]
        Figure size
        
    Returns:
    --------
    plt.Figure : The matplotlib figure
    """
    # Create layout for consistent positioning
    layout = nx.spring_layout(G, seed=42)
    
    # Create figure
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    # Set community as node attributes
    baseline_map = set_node_community(G.copy(), baseline_communities)
    enhanced_map = set_node_community(G.copy(), enhanced_communities)
    
    # Get number of communities
    num_baseline_communities = max(baseline_map.values()) + 1
    num_enhanced_communities = max(enhanced_map.values()) + 1
    
    # Get colormap for each partition
    baseline_colors = plt.cm.rainbow(np.linspace(0, 1, num_baseline_communities))
    enhanced_colors = plt.cm.rainbow(np.linspace(0, 1, num_enhanced_communities))
    
    # Draw baseline communities
    axes[0].set_title(f"Baseline Communities ({num_baseline_communities} communities)")
    node_colors = [baseline_colors[baseline_map[node]] for node in G.nodes()]
    nx.draw_networkx_nodes(G, layout, node_size=80, node_color=node_colors, alpha=0.8, ax=axes[0])
    nx.draw_networkx_edges(G, layout, alpha=0.4, ax=axes[0])
    axes[0].axis('off')
    
    # Draw enhanced communities
    axes[1].set_title(f"Enhanced Communities ({num_enhanced_communities} communities)")
    node_colors = [enhanced_colors[enhanced_map[node]] for node in G.nodes()]
    nx.draw_networkx_nodes(G, layout, node_size=80, node_color=node_colors, alpha=0.8, ax=axes[1])
    nx.draw_networkx_edges(G, layout, alpha=0.4, ax=axes[1])
    axes[1].axis('off')
    
    plt.tight_layout()
    return fig


def create_interactive_network(G: nx.Graph, 
                             communities: List[Set[int]] = None, 
                             community_map: Dict[int, int] = None) -> go.Figure:
    """
    Create an interactive plotly visualization of a network with communities.
    
    Parameters:
    -----------
    G : nx.Graph
        The input graph
    communities : List[Set[int]], optional
        Communities as a list of sets of node ids
    community_map : Dict[int, int], optional
        Pre-computed mapping of nodes to communities
        
    Returns:
    --------
    go.Figure : Plotly figure object
    """
    # Set community as a node attribute
    if community_map:
        node_to_community = community_map
    elif communities:
        node_to_community = set_node_community(G, communities)
    else:
        # If no communities provided, assign all to the same community
        node_to_community = {node: 0 for node in G.nodes()}
    
    # Create layout
    layout = nx.spring_layout(G, seed=42, dim=3)
    
    # Create colormap
    num_communities = max(node_to_community.values()) + 1
    colorscale = plt.cm.rainbow(np.linspace(0, 1, num_communities))
    
    # Convert to hex colors
    colors = ['#%02x%02x%02x' % (int(255*r), int(255*g), int(255*b)) for r, g, b, _ in colorscale]
    
    # Create edge trace
    edge_x = []
    edge_y = []
    edge_z = []
    
    for edge in G.edges():
        x0, y0, z0 = layout[edge[0]]
        x1, y1, z1 = layout[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])
        edge_z.extend([z0, z1, None])
        
    edge_trace = go.Scatter3d(
        x=edge_x, y=edge_y, z=edge_z,
        line=dict(width=1, color='#888'),
        hoverinfo='none',
        mode='lines')
    
    # Create node trace
    node_x = []
    node_y = []
    node_z = []
    node_colors = []
    node_text = []
    
    for node in G.nodes():
        x, y, z = layout[node]
        node_x.append(x)
        node_y.append(y)
        node_z.append(z)
        node_colors.append(colors[node_to_community[node]])
        node_text.append(f'Node {node}<br>Community {node_to_community[node]}')
    
    node_trace = go.Scatter3d(
        x=node_x, y=node_y, z=node_z,
        mode='markers',
        hoverinfo='text',
        text=node_text,
        marker=dict(
            showscale=False,
            color=node_colors,
            size=8,
            line=dict(width=2)))
    
    # Create figure
    fig = go.Figure(data=[edge_trace, node_trace],
                 layout=go.Layout(
                    title=f'Network with {num_communities} Communities',
                    showlegend=False,
                    margin=dict(b=0, l=0, r=0, t=40),
                    hovermode='closest',
                    scene=dict(
                        xaxis=dict(showticklabels=False),
                        yaxis=dict(showticklabels=False),
                        zaxis=dict(showticklabels=False)
                    )
                ))
    
    return fig


def visualize_node_reassignments(G: nx.Graph, 
                              baseline_communities: List[Set[int]], 
                              enhanced_communities: List[Set[int]],
                              figsize: Tuple[int, int] = (10, 8)) -> plt.Figure:
    """
    Visualize nodes that were reassigned to different communities.
    
    Parameters:
    -----------
    G : nx.Graph
        The input graph
    baseline_communities : List[Set[int]]
        Baseline communities
    enhanced_communities : List[Set[int]]
        Enhanced communities
    figsize : Tuple[int, int]
        Figure size
        
    Returns:
    --------
    plt.Figure : The matplotlib figure
    """
    # Create layout for consistent positioning
    layout = nx.spring_layout(G, seed=42)
    
    # Create maps from node to community
    baseline_map = {}
    for comm_id, comm in enumerate(baseline_communities):
        for node in comm:
            baseline_map[node] = comm_id
            
    enhanced_map = {}
    for comm_id, comm in enumerate(enhanced_communities):
        for node in comm:
            enhanced_map[node] = comm_id
    
    # Find reassigned nodes
    reassigned_nodes = []
    for node in baseline_map:
        if node in enhanced_map and baseline_map[node] != enhanced_map[node]:
            reassigned_nodes.append(node)
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Draw edges
    nx.draw_networkx_edges(G, layout, alpha=0.3)
    
    # Draw non-reassigned nodes
    non_reassigned = [n for n in G.nodes() if n not in reassigned_nodes]
    nx.draw_networkx_nodes(G, layout, nodelist=non_reassigned, node_size=80, 
                          node_color='lightgray', alpha=0.7)
    
    # Draw reassigned nodes
    nx.draw_networkx_nodes(G, layout, nodelist=reassigned_nodes, node_size=120, 
                          node_color='red', alpha=0.9)
    
    # Add labels to reassigned nodes
    reassigned_labels = {node: str(node) for node in reassigned_nodes}
    nx.draw_networkx_labels(G, layout, labels=reassigned_labels, font_size=8)
    
    plt.title(f'Nodes Reassigned to New Communities ({len(reassigned_nodes)} nodes)')
    plt.axis('off')
    
    return fig
