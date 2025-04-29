"""
Create a simple example network from Zachary's Karate Club.

This script creates a copy of the Zachary's Karate Club network in our data directory
for demonstration purposes.
"""

import networkx as nx
import os
import pandas as pd

# Create output directory if it doesn't exist
os.makedirs("c:/Users/dhvan/Downloads/Sem 8/NetworkScience/Project/data/real", exist_ok=True)

# Load the karate club network
G = nx.karate_club_graph()

# Save as edgelist
nx.write_edgelist(G, "c:/Users/dhvan/Downloads/Sem 8/NetworkScience/Project/data/real/karate_club.edgelist", data=False)

# Save the ground truth communities (administrator vs. instructor factions)
ground_truth = {}
for node in G.nodes():
    club = G.nodes[node]['club']
    if club not in ground_truth:
        ground_truth[club] = []
    ground_truth[club].append(node)

# Convert to DataFrame
community_data = []
for comm_id, nodes in ground_truth.items():
    for node in nodes:
        community_data.append([node, comm_id])

pd.DataFrame(community_data, columns=['Node', 'Community']).to_csv(
    "c:/Users/dhvan/Downloads/Sem 8/NetworkScience/Project/data/real/karate_club_factions.csv",
    index=False
)

print(f"Karate Club network saved with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")
print("Ground truth factions saved to csv file")
