# Enhanced Community Detection Algorithm

This documentation describes the Enhanced Modularity-Based Community Detection algorithm implemented in this project.

## Overview

The Enhanced Community Detection algorithm improves upon the standard Greedy Modularity Optimization (GMO) by incorporating local structural information such as clustering coefficients and internal connectivity. This approach addresses common issues with GMO, such as misplaced nodes and suboptimal community assignments.

## Algorithm Steps

1. **Baseline Community Detection:**
   - Apply standard Greedy Modularity Optimization to detect initial communities
   - Create a mapping from nodes to communities

2. **Misfit Node Identification:**
   - For each node, calculate:
     - Local clustering coefficient
     - Internal connectivity ratio (edges within community / total edges)
   - Identify misfit nodes as those with:
     - Low clustering coefficient (below threshold)
     - Low internal connectivity ratio (below threshold)

3. **Node Reassignment:**
   - For each misfit node, compute a connectivity score for each neighboring community
   - Reassign the node to the community with the highest score
   - Update community structure accordingly

4. **Result Evaluation:**
   - Calculate modularity for both baseline and enhanced communities
   - Compare community structures and node assignments
   - Analyze improvements in community coherence

## Implementation Details

The algorithm is implemented in `enhanced_community_detection.py`:

- `EnhancedCommunityDetection` class provides the main functionality
- `detect_baseline_communities()` performs initial GMO detection
- `identify_misfit_nodes()` finds nodes with poor fit
- `enhance_communities()` improves the community structure
- `calculate_modularity()` measures the quality of communities

## Parameters

Key parameters that can be tuned include:

- `clustering_threshold`: Threshold for considering a node to have low clustering (default: 0.2)
- `internal_connectivity_threshold`: Threshold for considering a node to have low internal connectivity (default: 0.3)

## Usage

```python
from src.enhanced_community_detection import EnhancedCommunityDetection
import networkx as nx

# Create or load a network
G = nx.karate_club_graph()

# Initialize the detector
detector = EnhancedCommunityDetection(G)

# Run baseline community detection
baseline_communities = detector.detect_baseline_communities()

# Run enhanced community detection
enhanced_communities = detector.enhance_communities(
    clustering_threshold=0.2,
    internal_connectivity_threshold=0.3
)

# Calculate modularity for comparison
baseline_modularity = detector.calculate_modularity(baseline_communities)
enhanced_modularity = detector.calculate_modularity(enhanced_communities)

print(f"Baseline Modularity: {baseline_modularity:.4f}")
print(f"Enhanced Modularity: {enhanced_modularity:.4f}")
print(f"Improvement: {enhanced_modularity - baseline_modularity:.4f}")
```

## Visualization

The algorithm provides methods for visualizing the community structures:

```python
# Generate visualizations
fig, axes = detector.visualize_communities(method='both')
```

## Performance Considerations

- The algorithm has a time complexity of O(n²) in the worst case for the enhancement step
- For very large networks (>10,000 nodes), consider:
  - Sampling the network first
  - Increasing the thresholds to limit the number of misfit nodes
  - Using the command-line tool with appropriate parameters

## References

- Newman, M. E. J. (2004). "Fast algorithm for detecting community structure in networks."
- Fortunato, S. (2010). "Community detection in graphs."
