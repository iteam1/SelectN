"""
Visualization utilities for selectN.

This module provides functions for visualizing selected samples and patterns.
"""
import os
import json
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from typing import List, Dict, Any, Optional, Tuple


def plot_document_embeddings(vector_representations: np.ndarray,
                            selected_indices: Optional[List[int]] = None,
                            cluster_labels: Optional[np.ndarray] = None,
                            output_path: Optional[str] = None,
                            title: str = "Document Embeddings",
                            figsize: Tuple[int, int] = (12, 8)):
    """
    Plot document embeddings in 2D space using t-SNE.
    
    Args:
        vector_representations: Document vector representations.
        selected_indices: Indices of selected documents (highlighted in plot).
        cluster_labels: Optional cluster labels for coloring points.
        output_path: Path to save the plot (if None, plot is displayed).
        title: Plot title.
        figsize: Figure size.
    """
    # Reduce dimensions for visualization
    # Use a smaller perplexity for small datasets
    n_samples = vector_representations.shape[0]
    perplexity = min(30, max(3, n_samples // 4))  # Scale perplexity based on sample size
    tsne = TSNE(n_components=2, random_state=0, perplexity=perplexity)
    vectors_2d = tsne.fit_transform(vector_representations)
    
    # Create figure
    plt.figure(figsize=figsize)
    
    # Set up plot styles
    if cluster_labels is not None:
        # Color by cluster
        scatter = plt.scatter(vectors_2d[:, 0], vectors_2d[:, 1], 
                             c=cluster_labels, cmap='viridis', 
                             alpha=0.7, s=50)
        plt.colorbar(scatter, label='Cluster')
    else:
        # Default coloring
        plt.scatter(vectors_2d[:, 0], vectors_2d[:, 1], alpha=0.7, s=50)
    
    # Highlight selected points
    if selected_indices:
        selected_points = vectors_2d[selected_indices]
        plt.scatter(selected_points[:, 0], selected_points[:, 1], 
                   color='red', s=100, marker='*', 
                   label=f'Selected ({len(selected_indices)})')
        plt.legend()
    
    plt.title(title)
    plt.tight_layout()
    
    # Save or display
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        return output_path
    else:
        plt.show()
        return None


def plot_similarity_matrix(similarities: np.ndarray,
                          selected_indices: Optional[List[int]] = None,
                          output_path: Optional[str] = None,
                          title: str = "Document Similarity Matrix",
                          figsize: Tuple[int, int] = (10, 8)):
    """
    Plot document similarity matrix as a heatmap.
    
    Args:
        similarities: Document similarity matrix.
        selected_indices: Indices of selected documents (highlighted in plot).
        output_path: Path to save the plot (if None, plot is displayed).
        title: Plot title.
        figsize: Figure size.
    """
    plt.figure(figsize=figsize)
    
    # Create heatmap
    ax = sns.heatmap(similarities, cmap='viridis', vmin=0, vmax=1)
    
    # Highlight selected documents
    if selected_indices:
        for idx in selected_indices:
            # Add a rectangle around the row and column
            ax.add_patch(plt.Rectangle((0, idx), similarities.shape[1], 1, 
                                      fill=False, edgecolor='red', lw=2))
            ax.add_patch(plt.Rectangle((idx, 0), 1, similarities.shape[0], 
                                      fill=False, edgecolor='red', lw=2))
    
    plt.title(title)
    plt.tight_layout()
    
    # Save or display
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        return output_path
    else:
        plt.show()
        return None


def plot_diversity_scores(diversity_scores: np.ndarray,
                         selected_indices: Optional[List[int]] = None,
                         output_path: Optional[str] = None,
                         title: str = "Document Diversity Scores",
                         figsize: Tuple[int, int] = (12, 6)):
    """
    Plot document diversity scores as a bar chart.
    
    Args:
        diversity_scores: Document diversity scores.
        selected_indices: Indices of selected documents (highlighted in plot).
        output_path: Path to save the plot (if None, plot is displayed).
        title: Plot title.
        figsize: Figure size.
    """
    plt.figure(figsize=figsize)
    
    # Create bar colors
    colors = ['lightblue'] * len(diversity_scores)
    if selected_indices:
        for idx in selected_indices:
            colors[idx] = 'red'
    
    # Create bar chart
    plt.bar(range(len(diversity_scores)), diversity_scores, color=colors)
    
    # Add a legend if we have selected indices
    if selected_indices:
        import matplotlib.patches as mpatches
        red_patch = mpatches.Patch(color='red', label='Selected')
        blue_patch = mpatches.Patch(color='lightblue', label='Not Selected')
        plt.legend(handles=[red_patch, blue_patch])
    
    plt.title(title)
    plt.xlabel('Document Index')
    plt.ylabel('Diversity Score')
    plt.tight_layout()
    
    # Save or display
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        return output_path
    else:
        plt.show()
        return None


def create_interactive_visualization(viz_data: Dict[str, Any],
                                   output_path: str,
                                   title: str = "Document Visualization"):
    """
    Create an interactive HTML visualization of document embeddings.
    
    Args:
        viz_data: Visualization data dictionary.
        output_path: Path to save the HTML file.
        title: Page title.
    
    Returns:
        Path to the saved HTML file.
    """
    # Convert numpy types to native Python types for JSON serialization
    processed_data = {
        "points": [],
        "selected_indices": [int(idx) for idx in viz_data.get("selected_indices", [])]
    }
    
    for point in viz_data.get("points", []):
        processed_point = {}
        for k, v in point.items():
            if isinstance(v, (np.int64, np.int32, np.int8)):
                processed_point[k] = int(v)
            elif isinstance(v, (np.float64, np.float32)):
                processed_point[k] = float(v)
            elif isinstance(v, np.bool_):
                processed_point[k] = bool(v)
            elif isinstance(v, (np.ndarray, list, tuple)):
                processed_point[k] = [float(x) if isinstance(x, (np.float64, np.float32)) 
                                    else int(x) if isinstance(x, (np.int64, np.int32, np.int8))
                                    else x for x in v]
            else:
                processed_point[k] = v
        processed_data["points"].append(processed_point)
    
    # Create HTML content
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>{title}</title>
        <script src="https://cdn.jsdelivr.net/npm/d3@7"></script>
        <style>
            body {{
                font-family: Arial, sans-serif;
                margin: 0;
                padding: 20px;
            }}
            #visualization {{
                width: 800px;
                height: 600px;
                margin: 20px auto;
                border: 1px solid #ccc;
            }}
            .point {{
                cursor: pointer;
            }}
            .selected {{
                stroke: red;
                stroke-width: 2px;
            }}
            #details {{
                width: 800px;
                margin: 20px auto;
                padding: 10px;
                border: 1px solid #ccc;
                border-radius: 5px;
                background-color: #f9f9f9;
                white-space: pre-wrap;
                font-family: monospace;
                max-height: 300px;
                overflow-y: auto;
            }}
            .cluster-legend {{
                margin: 10px;
            }}
            .legend-item {{
                display: inline-block;
                margin-right: 15px;
            }}
            .legend-color {{
                display: inline-block;
                width: 12px;
                height: 12px;
                margin-right: 5px;
            }}
        </style>
    </head>
    <body>
        <h1 style="text-align: center;">{title}</h1>
        <div id="visualization"></div>
        <div class="cluster-legend" id="legend"></div>
        <div id="details">Click on a point to see document details.</div>
        
        <script>
            // Parse visualization data
            const vizData = {json.dumps(processed_data)};
            
            // Set up the SVG
            const width = 800;
            const height = 600;
            const margin = {{ top: 40, right: 40, bottom: 40, left: 40 }};
            const innerWidth = width - margin.left - margin.right;
            const innerHeight = height - margin.top - margin.bottom;
            
            const svg = d3.select("#visualization")
                .append("svg")
                .attr("width", width)
                .attr("height", height);
                
            const g = svg.append("g")
                .attr("transform", `translate(${{margin.left}}, ${{margin.top}})`);
            
            // Compute scales
            const xExtent = d3.extent(vizData.points, d => d.x);
            const yExtent = d3.extent(vizData.points, d => d.y);
            
            const xScale = d3.scaleLinear()
                .domain([xExtent[0] - 1, xExtent[1] + 1])
                .range([0, innerWidth]);
                
            const yScale = d3.scaleLinear()
                .domain([yExtent[0] - 1, yExtent[1] + 1])
                .range([innerHeight, 0]);
            
            // Set up color scale for clusters
            const clusters = [...new Set(vizData.points.map(d => d.cluster))];
            const colorScale = d3.scaleOrdinal(d3.schemeCategory10)
                .domain(clusters);
            
            // Create legend
            const legend = d3.select("#legend");
            clusters.forEach(cluster => {{
                const item = legend.append("div")
                    .attr("class", "legend-item");
                
                item.append("div")
                    .attr("class", "legend-color")
                    .style("background-color", colorScale(cluster));
                    
                item.append("span")
                    .text(`Cluster ${{cluster}}`);
            }});
            
            // Add axes
            const xAxis = d3.axisBottom(xScale);
            const yAxis = d3.axisLeft(yScale);
            
            g.append("g")
                .attr("transform", `translate(0, ${{innerHeight}})`)
                .call(xAxis);
                
            g.append("g")
                .call(yAxis);
            
            // Add points
            g.selectAll("circle")
                .data(vizData.points)
                .enter()
                .append("circle")
                .attr("class", d => `point ${{d.selected ? "selected" : ""}}`)
                .attr("cx", d => xScale(d.x))
                .attr("cy", d => yScale(d.y))
                .attr("r", d => d.selected ? 8 : 5)
                .attr("fill", d => colorScale(d.cluster))
                .attr("opacity", 0.7)
                .on("mouseover", function(event, d) {{
                    d3.select(this).attr("opacity", 1);
                }})
                .on("mouseout", function(event, d) {{
                    d3.select(this).attr("opacity", 0.7);
                }})
                .on("click", function(event, d) {{
                    // Display document details
                    const details = document.getElementById("details");
                    
                    let detailsText = "Document Details:\\n";
                    
                    if (d.metadata && d.metadata.file_path) {{
                        detailsText += `File: ${{d.metadata.file_path}}\\n`;
                    }}
                    
                    detailsText += `\\nCluster: ${{d.cluster}}\\n`;
                    detailsText += `Diversity Score: ${{d.diversity.toFixed(4)}}\\n`;
                    detailsText += `Selected: ${{d.selected ? "Yes" : "No"}}\\n`;
                    
                    details.textContent = detailsText;
                }});
            
            // Add a legend for selected points
            svg.append("circle")
                .attr("cx", width - 150)
                .attr("cy", 20)
                .attr("r", 8)
                .attr("fill", "none")
                .attr("stroke", "red")
                .attr("stroke-width", 2);
                
            svg.append("text")
                .attr("x", width - 135)
                .attr("y", 24)
                .text("Selected Documents");
        </script>
    </body>
    </html>
    """
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Write to file
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    return output_path


def generate_visualization_suite(viz_data: Dict[str, Any], 
                               output_dir: str,
                               prefix: str = "viz_") -> Dict[str, str]:
    """
    Generate a suite of visualizations based on visualization data.
    
    Args:
        viz_data: Visualization data dictionary.
        output_dir: Directory to save visualizations.
        prefix: File name prefix for saved files.
    
    Returns:
        Dictionary mapping visualization type to file path.
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Extract data
    points = np.array([[p['x'], p['y']] for p in viz_data['points']])
    selected_indices = viz_data['selected_indices']
    clusters = np.array([p['cluster'] for p in viz_data['points']])
    diversity_scores = np.array([p['diversity'] for p in viz_data['points']])
    
    # Generate visualizations
    viz_files = {}
    
    # 1. Document embeddings plot
    embedding_path = os.path.join(output_dir, f"{prefix}embeddings.png")
    plot_document_embeddings(points, selected_indices, clusters, 
                           embedding_path, "Document Embeddings")
    viz_files['embeddings'] = embedding_path
    
    # 2. Diversity scores plot
    diversity_path = os.path.join(output_dir, f"{prefix}diversity.png")
    plot_diversity_scores(diversity_scores, selected_indices,
                        diversity_path, "Document Diversity Scores")
    viz_files['diversity'] = diversity_path
    
    # 3. Interactive visualization
    interactive_path = os.path.join(output_dir, f"{prefix}interactive.html")
    create_interactive_visualization(viz_data, interactive_path,
                                  "Interactive Document Visualization")
    viz_files['interactive'] = interactive_path
    
    return viz_files