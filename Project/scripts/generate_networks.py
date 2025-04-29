"""
Generate synthetic networks with planted communities.
"""

import networkx as nx
import pandas as pd
import os
import random
import csv

def generate_network_with_communities(num_communities=5, nodes_per_community=20, 
                                     p_in=0.3, p_out=0.01, seed=42):
    """
    Generate a network with planted community structure.
    
    Parameters:
    -----------
    num_communities : int
        Number of communities
    nodes_per_community : int
        Number of nodes per community
    p_in : float
        Probability of connection within communities
    p_out : float
        Probability of connection between communities
    seed : int
        Random seed
        
    Returns:
    --------
    G : nx.Graph
        Generated network
    communities : dict
        Dict mapping community IDs to node sets
    """
    random.seed(seed)
    
    # Create empty graph
    G = nx.Graph()
    
    # Create communities
    communities = {}
    node_id = 0
    
    for comm_id in range(num_communities):
        community = set()
        
        # Add nodes for this community
        for _ in range(nodes_per_community):
            G.add_node(node_id, community=comm_id)
            community.add(node_id)
            node_id += 1
            
        communities[comm_id] = community
    
    # Add edges within communities
    for comm_id, nodes in communities.items():
        node_list = list(nodes)
        for i in range(len(node_list)):
            for j in range(i+1, len(node_list)):
                if random.random() < p_in:
                    G.add_edge(node_list[i], node_list[j])
    
    # Add edges between communities
    for comm1 in range(num_communities):
        for comm2 in range(comm1 + 1, num_communities):
            nodes1 = list(communities[comm1])
            nodes2 = list(communities[comm2])
            
            for node1 in nodes1:
                for node2 in nodes2:
                    if random.random() < p_out:
                        G.add_edge(node1, node2)
    
    return G, communities

def save_network(G, communities, base_path):
    """Save network and community data"""
    # Create the base directory if it doesn't exist
    if not os.path.exists(os.path.dirname(base_path)):
        os.makedirs(os.path.dirname(base_path))
    
    # Save network as edgelist
    nx.write_edgelist(G, f"{base_path}.edgelist", data=False)
    
    # Save communities
    with open(f"{base_path}_communities.csv", 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Node', 'Community'])
        
        for comm_id, nodes in communities.items():
            for node in nodes:
                writer.writerow([node, comm_id])
    
    print(f"Network saved to {base_path}.edgelist")
    print(f"Communities saved to {base_path}_communities.csv")

if __name__ == "__main__":
    # Get the project root directory
    project_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    
    # Create output directory if it doesn't exist
    output_dir = os.path.join(project_dir, "data", "synthetic")
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Creating synthetic networks in {output_dir}")
    
    # Generate networks with different community structures
    
    # 1. Clear community structure
    G1, comm1 = generate_network_with_communities(
        num_communities=5, 
        nodes_per_community=20,
        p_in=0.3, 
        p_out=0.01, 
        seed=42
    )
    save_network(G1, comm1, os.path.join(output_dir, "clear_communities"))
    
    # 2. Medium community structure
    G2, comm2 = generate_network_with_communities(
        num_communities=5, 
        nodes_per_community=20,
        p_in=0.2, 
        p_out=0.05, 
        seed=43
    )
    save_network(G2, comm2, os.path.join(output_dir, "medium_communities"))
    
    # 3. Ambiguous community structure
    G3, comm3 = generate_network_with_communities(
        num_communities=5, 
        nodes_per_community=20,
        p_in=0.15, 
        p_out=0.09, 
        seed=44
    )
    save_network(G3, comm3, os.path.join(output_dir, "ambiguous_communities"))
