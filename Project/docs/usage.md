# User Guide: Enhanced Community Detection

This guide provides instructions for using the Enhanced Community Detection tools in this project.

## Installation

1. Clone the repository or download the source code
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Using the Streamlit App

The Streamlit app provides an interactive interface for community detection.

1. Start the app:
   ```bash
   streamlit run src/app.py
   ```

2. In the web interface that opens in your browser:
   - Upload a network file or use a demo network
   - Explore the network structure on the "Network Analysis" page
   - Run community detection algorithms on the "Community Detection" page
   - Compare results on the "Comparison" page
   - Explore advanced features on the "Advanced Features" page

## Command-line Usage

For batch processing or integration into other workflows, use the command-line tool:

```bash
python src/run_detection.py --input <network_file> --output <results_dir> [options]
```

Options:
- `--input`: Path to input network file (required)
- `--output`: Directory for output results (default: 'results')
- `--directed`: Flag indicating network is directed
- `--weighted`: Flag indicating network has edge weights
- `--clustering-threshold`: Threshold for clustering coefficient (default: 0.2)
- `--connectivity-threshold`: Threshold for internal connectivity (default: 0.3)
- `--visualize`: Flag to generate visualizations

Example:
```bash
python src/run_detection.py --input data/karate.edgelist --output results/karate --visualize
```

## Using as a Python Library

You can also use the algorithm in your own Python code:

```python
from src.enhanced_community_detection import EnhancedCommunityDetection
import networkx as nx

# Load or create a network
G = nx.karate_club_graph()

# Initialize detector
detector = EnhancedCommunityDetection(G)

# Run community detection
baseline = detector.detect_baseline_communities()
enhanced = detector.enhance_communities()

# Calculate metrics
metrics = detector.calculate_community_metrics()
print(metrics)

# Visualize results
fig, axes = detector.visualize_communities(method='both')
fig.savefig('community_comparison.png')
```

## Input File Formats

The tool supports various network file formats:

- Edgelists (.edgelist, .txt): Simple text files with one edge per line
- CSV files (.csv): With columns for source, target, and optionally weight
- GraphML (.graphml): XML-based format for graph structures
- GML (.gml): Text-based format for graphs
- GEXF (.gexf): XML-based format for complex networks

Example edgelist format:
```
0 1
0 2
1 2
2 3
...
```

Example CSV format:
```
source,target,weight
0,1,1.0
0,2,2.5
1,2,1.0
...
```

## Output Files

When using the command-line tool, the following outputs are generated:

- `community_metrics.csv`: Comparison of baseline and enhanced methods
- `baseline_communities.csv`: Node-to-community assignments from baseline method
- `enhanced_communities.csv`: Node-to-community assignments from enhanced method
- `reassigned_nodes.csv`: List of nodes that were reassigned
- `community_visualization.png`: Visual comparison of community structures

## Troubleshooting

- **Memory issues with large networks**: For very large networks (>100,000 nodes), use sampling methods from `data_utils.py` to reduce the size
- **Slow performance**: Increase the threshold values to reduce the number of misfit nodes
- **File format errors**: Ensure your network file matches one of the supported formats
- **Visualization issues**: For large networks, the visualization may become cluttered; use the `--no-visualization` flag
- **Module not found errors**: Ensure you've installed all dependencies from requirements.txt
