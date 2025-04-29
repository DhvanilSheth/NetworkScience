"""
Generate synthetic networks with planted communities.
"""

import networkx as nx
import pandas as pd
import os
import random
import sys

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

def main():
    # Determine project root directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.abspath(os.path.join(script_dir, ".."))
    
    # Create output directory
    synthetic_dir = os.path.join(project_dir, "data", "synthetic")
    os.makedirs(synthetic_dir, exist_ok=True)
    
    print(f"Creating synthetic networks in: {synthetic_dir}")
    
    # Generate networks with different community structures
    
    # 1. Clear community structure
    G1, comm1 = generate_network_with_communities(
        num_communities=5, 
        nodes_per_community=20,
        p_in=0.3, 
        p_out=0.01, 
        seed=42
    )
    
    # Save the network
    edges_file1 = os.path.join(synthetic_dir, "clear_communities.edgelist")
    comm_file1 = os.path.join(synthetic_dir, "clear_communities_ground_truth.csv")
    
    # Save as edgelist
    nx.write_edgelist(G1, edges_file1, data=False)
    
    # Save community data
    community_data = []
    for comm_id, nodes in comm1.items():
        for node in nodes:
            community_data.append([node, comm_id])
            
    pd.DataFrame(community_data, columns=['Node', 'Community']).to_csv(
        comm_file1, index=False
    )
    
    print(f"Created network with clear communities: {edges_file1}")
    print(f"  - {G1.number_of_nodes()} nodes, {G1.number_of_edges()} edges")
    print(f"  - Ground truth saved to {comm_file1}")
    
    # 2. Medium community structure
    G2, comm2 = generate_network_with_communities(
        num_communities=5, 
        nodes_per_community=20,
        p_in=0.2, 
        p_out=0.05, 
        seed=43
    )
    
    # Save the network
    edges_file2 = os.path.join(synthetic_dir, "medium_communities.edgelist")
    comm_file2 = os.path.join(synthetic_dir, "medium_communities_ground_truth.csv")
    
    # Save as edgelist
    nx.write_edgelist(G2, edges_file2, data=False)
    
    # Save community data
    community_data = []
    for comm_id, nodes in comm2.items():
        for node in nodes:
            community_data.append([node, comm_id])
            
    pd.DataFrame(community_data, columns=['Node', 'Community']).to_csv(
        comm_file2, index=False
    )
    
    print(f"Created network with medium communities: {edges_file2}")
    print(f"  - {G2.number_of_nodes()} nodes, {G2.number_of_edges()} edges")
    print(f"  - Ground truth saved to {comm_file2}")
    
    # 3. Ambiguous community structure
    G3, comm3 = generate_network_with_communities(
        num_communities=5, 
        nodes_per_community=20,
        p_in=0.15, 
        p_out=0.09, 
        seed=44
    )
    
    # Save the network
    edges_file3 = os.path.join(synthetic_dir, "ambiguous_communities.edgelist")
    comm_file3 = os.path.join(synthetic_dir, "ambiguous_communities_ground_truth.csv")
    
    # Save as edgelist
    nx.write_edgelist(G3, edges_file3, data=False)
    
    # Save community data
    community_data = []
    for comm_id, nodes in comm3.items():
        for node in nodes:
            community_data.append([node, comm_id])
            
    pd.DataFrame(community_data, columns=['Node', 'Community']).to_csv(
        comm_file3, index=False
    )
    
    print(f"Created network with ambiguous communities: {edges_file3}")
    print(f"  - {G3.number_of_nodes()} nodes, {G3.number_of_edges()} edges")
    print(f"  - Ground truth saved to {comm_file3}")

if __name__ == "__main__":
    main()
