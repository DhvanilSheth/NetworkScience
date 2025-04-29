# Enhanced Modularity-Based Community Detection

This project implements an advanced community detection algorithm that enhances classic Greedy Modularity Optimization (GMO) by integrating local clustering coefficients and network structural metrics. The goal is to improve community coherence and better identify the true community structure in complex networks.

## Overview

Community detection algorithms often face challenges with misplaced nodes and suboptimal community assignments. This project addresses these limitations by:

1. Leveraging **local clustering coefficients** to identify nodes that may be incorrectly assigned
2. Analyzing **internal vs. external connectivity** to determine better community placements
3. Using **global network structure** to inform community refinement
4. Providing comprehensive **visualization tools** to interpret results

## Project Structure

```
Project/
├── data/               # Example networks and datasets
│   ├── synthetic/      # Synthetic networks with known community structure
│   └── real/           # Real-world network datasets
├── src/                # Source code for the enhanced algorithm
│   ├── enhanced_community_detection.py # Main algorithm implementation
│   ├── data_utils.py   # Utilities for loading and processing networks
│   ├── app.py          # Streamlit web application
│   └── run_detection.py # Command-line interface
├── notebooks/          # Jupyter notebooks for exploration and demonstration
├── docs/               # Documentation files
├── scripts/            # Analysis and utility scripts
├── results/            # Output results and metrics
└── visualizations/     # Generated visualizations
```

## Features

- **Macro-level network analysis**: degree distribution, assortativity, clustering
- **Enhanced community detection** leveraging local structure awareness
- **Multiple visualization methods** for community structure analysis
- **Interactive Streamlit interface** for exploration and demonstration
- **Performance comparison** with other community detection algorithms
- **Domain inference** capabilities to identify node functional groups

## Getting Started

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Run the Streamlit app:
```bash
streamlit run src/app.py
```

3. Or use the command-line interface:
```bash
python src/run_detection.py --input <network_file> --output <results_dir> --visualize
```

## Algorithm Methodology

The algorithm improves community detection through a multi-stage process:

1. **Initial Detection**: Uses Greedy Modularity Optimization to find baseline communities
2. **Misfit Identification**: Identifies nodes with low clustering coefficient and weak internal connectivity
3. **Node Reassignment**: Moves misfit nodes to more appropriate communities based on local structure
4. **Performance Evaluation**: Compares modularity, cohesion, and other metrics before and after refinement

For detailed information about the algorithm, see [docs/algorithm.md](docs/algorithm.md).

## Use Cases

This enhanced community detection approach is valuable for:

- **Social network analysis**: Identifying meaningful groups in social structures
- **Biological networks**: Finding functional modules in protein-protein interaction networks
- **Information networks**: Discovering topic clusters in citation or co-authorship networks
- **Infrastructure networks**: Detecting functional regions in transportation or utility networks

## References

- Newman, M. E. J. (2004). "Fast algorithm for detecting community structure in networks."
- Fortunato, S. (2010). "Community detection in graphs."
- Lancichinetti, A., & Fortunato, S. (2009). "Community detection algorithms: a comparative analysis."
