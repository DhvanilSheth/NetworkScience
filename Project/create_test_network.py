# Generate a simple community-based network
import networkx as nx
import pandas as pd
import os
import random

# Create directory
os.makedirs("c:\\Users\\dhvan\\Downloads\\Sem 8\\NetworkScience\\Project\\data\\synthetic", exist_ok=True)

# Set random seed
random.seed(42)

# Create the community structure
num_communities = 4
nodes_per_community = 15

# Create graph
G = nx.Graph()

# Create communities
communities = {}
node_id = 0

for comm_id in range(num_communities):
    communities[comm_id] = []
    
    # Add nodes to this community
    for _ in range(nodes_per_community):
        G.add_node(node_id, community=comm_id)
        communities[comm_id].append(node_id)
        node_id += 1

# Add edges within communities (high probability)
p_in = 0.3
for comm_id, nodes in communities.items():
    for i in range(len(nodes)):
        for j in range(i+1, len(nodes)):
            if random.random() < p_in:
                G.add_edge(nodes[i], nodes[j])

# Add edges between communities (low probability)
p_out = 0.02
for comm1 in range(num_communities):
    for comm2 in range(comm1+1, num_communities):
        for node1 in communities[comm1]:
            for node2 in communities[comm2]:
                if random.random() < p_out:
                    G.add_edge(node1, node2)

# Save edgelist
nx.write_edgelist(G, "c:\\Users\\dhvan\\Downloads\\Sem 8\\NetworkScience\\Project\\data\\synthetic\\test_network.edgelist", data=False)

# Save ground truth communities
communities_list = []
for comm_id, nodes in communities.items():
    for node in nodes:
        communities_list.append([node, comm_id])

pd.DataFrame(communities_list, columns=['Node', 'Community']).to_csv(
    "c:\\Users\\dhvan\\Downloads\\Sem 8\\NetworkScience\\Project\\data\\synthetic\\test_communities.csv", 
    index=False
)

print(f"Created network with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")
print(f"Network has {num_communities} communities with {nodes_per_community} nodes each")
