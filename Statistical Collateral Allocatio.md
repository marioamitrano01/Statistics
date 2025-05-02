# Collateral Allocation Optimizer

## Overview
This tool optimizes the allocation of collateral across multiple Credit Support Annexes (CSAs) using linear programming. It minimizes cost while satisfying margin requirements and respecting concentration limits.

## Features
- **Linear Programming Optimization**: Uses PuLP library to create and solve an efficient allocation model
- **Concentration Limits**: Enforces maximum percentage limits for each collateral type
- **Interactive Visualization**: Generates stacked bar charts to visualize allocations
- **Performance Optimized**: Pre-computes values and uses efficient data structures

The model is suitable for large-scale allocation problems across multiple CSAs with diverse collateral types.
