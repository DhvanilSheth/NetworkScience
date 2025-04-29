"""
Data utility functions for loading and preprocessing network datasets.
"""

import networkx as nx
import pandas as pd
import numpy as np
import os
from typing import Tuple, Dict, Any, Optional, Union


def load_network(file_path: str, directed: bool = False, weighted: bool = False) -> nx.Graph:
    """
    Load a network from various file formats based on extension.
    
    Parameters:
    -----------
    file_path : str
        Path to the network file
    directed : bool
        Whether to create a directed graph
    weighted : bool
        Whether edges have weights
        
    Returns:
    --------
    nx.Graph or nx.DiGraph : The loaded network
    """
    # Determine file extension
    _, ext = os.path.splitext(file_path)
    
    # Initialize the appropriate graph type
    G = nx.DiGraph() if directed else nx.Graph()
    
    # Load based on file type
    if ext.lower() in ['.edgelist', '.txt']:
        if weighted:
            G = nx.read_weighted_edgelist(file_path, create_using=G)
        else:
            G = nx.read_edgelist(file_path, create_using=G)
    elif ext.lower() == '.csv':
        df = pd.read_csv(file_path)
        
        # Check column names to determine format
        if 'source' in df.columns and 'target' in df.columns:
            source_col, target_col = 'source', 'target'
        elif 'from' in df.columns and 'to' in df.columns:
            source_col, target_col = 'from', 'to'
        else:
            # Assume first two columns are source and target
            source_col, target_col = df.columns[0], df.columns[1]
            
        # Check if weight column exists
        weight_col = None
        for col in ['weight', 'value', 'strength']:
            if col in df.columns:
                weight_col = col
                break
                
        # Add edges to graph
        for _, row in df.iterrows():
            u = row[source_col]
            v = row[target_col]
            
            if weight_col is not None and weighted:
                w = row[weight_col]
                G.add_edge(u, v, weight=w)
            else:
                G.add_edge(u, v)
    elif ext.lower() == '.graphml':
        G = nx.read_graphml(file_path)
        if not directed:
            G = G.to_undirected()
    elif ext.lower() == '.gml':
        G = nx.read_gml(file_path)
        if not directed:
            G = G.to_undirected()
    elif ext.lower() == '.gexf':
        G = nx.read_gexf(file_path)
        if not directed:
            G = G.to_undirected()
    else:
        raise ValueError(f"Unsupported file format: {ext}")
    
    return G


def generate_synthetic_network(network_type: str, 
                              n: int, 
                              **params) -> nx.Graph:
    """
    Generate synthetic networks for testing.
    
    Parameters:
    -----------
    network_type : str
        Type of network to generate ('barabasi_albert', 'watts_strogatz', 'random', etc.)
    n : int
        Number of nodes
    params : dict
        Additional parameters for the specific network model
        
    Returns:
    --------
    nx.Graph : Generated synthetic network
    """
    if network_type == 'barabasi_albert':
        m = params.get('m', 3)  # Number of edges to attach from a new node
        G = nx.barabasi_albert_graph(n, m)
    elif network_type == 'watts_strogatz':
        k = params.get('k', 4)  # Each node is connected to k nearest neighbors
        p = params.get('p', 0.1)  # Probability of rewiring each edge
        G = nx.watts_strogatz_graph(n, k, p)
    elif network_type == 'random':
        p = params.get('p', 0.1)  # Probability of edge creation
        G = nx.erdos_renyi_graph(n, p)
    elif network_type == 'complete':
        G = nx.complete_graph(n)
    elif network_type == 'star':
        G = nx.star_graph(n-1)
    elif network_type == 'path':
        G = nx.path_graph(n)
    elif network_type == 'cycle':
        G = nx.cycle_graph(n)
    elif network_type == 'grid':
        m = int(np.sqrt(n))
        G = nx.grid_2d_graph(m, m)
    elif network_type == 'regular':
        d = params.get('d', 3)  # Degree of each node
        G = nx.random_regular_graph(d, n)
    else:
        raise ValueError(f"Unknown network type: {network_type}")
    
    return G


def largest_connected_component(G: nx.Graph) -> nx.Graph:
    """
    Extract the largest connected component of a graph.
    
    Parameters:
    -----------
    G : nx.Graph
        Input graph
        
    Returns:
    --------
    nx.Graph : Largest connected component subgraph
    """
    if not nx.is_connected(G):
        largest_cc = max(nx.connected_components(G), key=len)
        return G.subgraph(largest_cc).copy()
    return G.copy()


def add_metadata_to_network(G: nx.Graph, 
                           node_metadata: Optional[Dict] = None,
                           edge_metadata: Optional[Dict] = None) -> nx.Graph:
    """
    Add metadata to nodes and/or edges.
    
    Parameters:
    -----------
    G : nx.Graph
        Input graph
    node_metadata : dict, optional
        Dictionary mapping node IDs to attribute dictionaries
    edge_metadata : dict, optional
        Dictionary mapping edge tuples (u, v) to attribute dictionaries
        
    Returns:
    --------
    nx.Graph : Graph with added metadata
    """
    H = G.copy()
    
    # Add node metadata
    if node_metadata:
        for node, attrs in node_metadata.items():
            if node in H:
                for key, value in attrs.items():
                    H.nodes[node][key] = value
    
    # Add edge metadata
    if edge_metadata:
        for edge, attrs in edge_metadata.items():
            u, v = edge
            if H.has_edge(u, v):
                for key, value in attrs.items():
                    H.edges[u, v][key] = value
    
    return H


def standardize_network(G: nx.Graph) -> nx.Graph:
    """
    Standardize a network to ensure consistent node IDs and attributes.
    
    Parameters:
    -----------
    G : nx.Graph
        Input graph
        
    Returns:
    --------
    nx.Graph : Standardized graph with integer node IDs
    """
    H = nx.Graph()
    
    # Create a mapping from original node IDs to integers
    node_mapping = {node: i for i, node in enumerate(G.nodes())}
    
    # Add nodes with attributes
    for node, attrs in G.nodes(data=True):
        H.add_node(node_mapping[node], **attrs)
    
    # Add edges with attributes
    for u, v, attrs in G.edges(data=True):
        H.add_edge(node_mapping[u], node_mapping[v], **attrs)
    
    return H


def sample_network(G: nx.Graph, sample_size: int, 
                  method: str = 'random') -> nx.Graph:
    """
    Sample a subgraph from a larger network.
    
    Parameters:
    -----------
    G : nx.Graph
        Input graph
    sample_size : int
        Number of nodes to include in the sample
    method : str
        Sampling method ('random', 'snowball', 'forest_fire')
        
    Returns:
    --------
    nx.Graph : Sampled subgraph
    """
    if sample_size >= G.number_of_nodes():
        return G.copy()
        
    if method == 'random':
        # Random node sampling
        nodes = list(G.nodes())
        sampled_nodes = np.random.choice(nodes, size=sample_size, replace=False)
        return G.subgraph(sampled_nodes).copy()
    
    elif method == 'snowball':
        # Snowball sampling (BFS-based)
        sampled_nodes = set()
        start_node = np.random.choice(list(G.nodes()))
        
        # Use BFS to expand from start_node
        queue = [start_node]
        while len(sampled_nodes) < sample_size and queue:
            current = queue.pop(0)
            if current not in sampled_nodes:
                sampled_nodes.add(current)
                queue.extend(list(G.neighbors(current)))
                
                if len(sampled_nodes) >= sample_size:
                    break
        
        return G.subgraph(sampled_nodes).copy()
    
    elif method == 'forest_fire':
        # Forest fire sampling
        sampled_nodes = set()
        start_node = np.random.choice(list(G.nodes()))
        sampled_nodes.add(start_node)
        
        p_forward = 0.5  # Forward burning probability
        
        # Initialize the fire front
        fire_front = [start_node]
        
        while len(sampled_nodes) < sample_size and fire_front:
            current = fire_front.pop(0)
            neighbors = list(G.neighbors(current))
            
            # For each neighbor, decide whether to "burn" it
            for neighbor in neighbors:
                if neighbor not in sampled_nodes and np.random.random() < p_forward:
                    sampled_nodes.add(neighbor)
                    fire_front.append(neighbor)
                    
                    if len(sampled_nodes) >= sample_size:
                        break
        
        return G.subgraph(sampled_nodes).copy()
    
    else:
        raise ValueError(f"Unknown sampling method: {method}")


def save_network(G: nx.Graph, file_path: str) -> None:
    """
    Save a network to a file.
    
    Parameters:
    -----------
    G : nx.Graph
        Network to save
    file_path : str
        Path where to save the network
    """
    # Determine file extension
    _, ext = os.path.splitext(file_path)
    
    if ext.lower() == '.graphml':
        nx.write_graphml(G, file_path)
    elif ext.lower() == '.gexf':
        nx.write_gexf(G, file_path)
    elif ext.lower() == '.gml':
        nx.write_gml(G, file_path)
    elif ext.lower() in ['.edgelist', '.txt']:
        nx.write_edgelist(G, file_path)
    elif ext.lower() == '.csv':
        df = nx.to_pandas_edgelist(G)
        df.to_csv(file_path, index=False)
    else:
        raise ValueError(f"Unsupported file format: {ext}")
