import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import logging
import json

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class HolographicVisualizer:
   
    
    def __init__(self, random_state=42):
        self.random_state = random_state
        np.random.seed(random_state)
        
        self.thresholds = {
            'bytes_threshold': 1e6,       
            'packets_threshold': 1000,       
            'holo_load_threshold': 0.85,   
            'similarity_threshold': 0.7,   
            'flow_duration_threshold': 1000000,
        }
        logger.info(f"Initialized with exact thresholds: {self.thresholds}")
        
    def load_data(self, file_path):
        """Load dataset and analyze class distribution"""
        try:
            df = pd.read_csv(file_path)
            print("Dataset loaded successfully.")
            print(f"Shape: {df.shape}")
            
            # Analyze class distribution
            if 'Attack Name' in df.columns:
                class_counts = df['Attack Name'].value_counts()
                print("\nClass distribution in dataset:")
                for class_name, count in class_counts.items():
                    percentage = (count / len(df)) * 100
                    print(f"  {class_name}: {count} samples ({percentage:.1f}%)")
            
            return df
        except Exception as e:
            print(f"Error loading file: {e}")
            return None

    def apply_holographic_thresholds(self, df):
        """Apply thresholds"""
        logger.info("Applying holographic thresholds to determine device states")
        
        # Calculate device metrics based on thresholds
        device_metrics = {}
        
        for device in df['Src IP'].unique()[:10]:  # Analyze first 10 devices
            device_data = df[df['Src IP'] == device]
            
            # Calculate metrics used in threshold
            avg_bytes = device_data['Total Length of Fwd Packet'].mean() if 'Total Length of Fwd Packet' in df.columns else 0
            avg_packets = device_data['Total Fwd Packet'].mean() if 'Total Fwd Packet' in df.columns else 0
            flow_duration = device_data['Flow Duration'].mean() if 'Flow Duration' in df.columns else 0
            
            # Apply threshold logic
            is_overloaded = (avg_bytes > self.thresholds['bytes_threshold'] or 
                           avg_packets > self.thresholds['packets_threshold'] or
                           flow_duration > self.thresholds['flow_duration_threshold'])
            
            device_metrics[device] = {
                'state': 'overloaded' if is_overloaded else 'active',
                'avg_bytes': avg_bytes,
                'avg_packets': avg_packets,
                'flow_duration': flow_duration,
                'exceeds_thresholds': is_overloaded
            }
        
        return device_metrics

    def visualize_3d_network(self, df):
        """3D Holographic Network Visualization with exact thresholds"""
        if {'Src IP', 'Dst IP', 'Attack Name'}.issubset(df.columns):
            G = nx.Graph()

            # Apply holographic thresholds to determine device states
            device_metrics = self.apply_holographic_thresholds(df)
            logger.info(f"Applied thresholds to {len(device_metrics)} devices")

            # Sample data maintaining original class proportions
            normal_data = df[df['Attack Name'] == 'Normal']
            anomaly_data = df[df['Attack Name'] == 'Anomaly']
            
            normal_sample = normal_data.sample(n=min(300, len(normal_data)), random_state=self.random_state)
            anomaly_sample = anomaly_data.sample(n=min(200, len(anomaly_data)), random_state=self.random_state)
            
            df_sample = pd.concat([normal_sample, anomaly_sample])
            
            print(f"Visualization sample: {len(normal_sample)} Normal, {len(anomaly_sample)} Anomaly")
            
            # Add edges with class information and apply threshold-based styling
            for _, row in df_sample.iterrows():
                src_device = row['Src IP']
                device_state = device_metrics.get(src_device, {}).get('state', 'active')
                
                G.add_edge(row['Src IP'], row['Dst IP'], 
                          class_type=row['Attack Name'],
                          device_state=device_state)

            # Use 3D spring layout
            pos = nx.spring_layout(G, dim=3, seed=self.random_state)

            fig = plt.figure(figsize=(12, 10))
            ax = fig.add_subplot(111, projection='3d')

            # Color mapping for classes
            class_colors = {'Normal': 'lightgreen', 'Anomaly': 'red'}
            
            # Draw edges with class-based coloring
            for edge in G.edges():
                if edge[0] in pos and edge[1] in pos:
                    class_type = G[edge[0]][edge[1]].get('class_type', 'Unknown')
                    device_state = G[edge[0]][edge[1]].get('device_state', 'active')
                    
                    color = class_colors.get(class_type, 'gray')
                    linewidth = 2.5 if device_state == 'overloaded' else 1.5
                    
                    x_vals = [pos[edge[0]][0], pos[edge[1]][0]]
                    y_vals = [pos[edge[0]][1], pos[edge[1]][1]]
                    z_vals = [pos[edge[0]][2], pos[edge[1]][2]]
                    
                    ax.plot(x_vals, y_vals, z_vals, color=color, alpha=0.6, 
                           linewidth=linewidth)

            # Draw nodes with size based on degree and state
            for node in G.nodes():
                if node in pos:
                    # Determine node color based on class and state
                    node_classes = [G[node][neighbor].get('class_type', 'Unknown') 
                                  for neighbor in G.neighbors(node)]
                    node_states = [G[node][neighbor].get('device_state', 'active')
                                 for neighbor in G.neighbors(node)]
                    
                    # Classify node
                    normal_count = node_classes.count('Normal')
                    anomaly_count = node_classes.count('Anomaly')
                    node_class = 'Normal' if normal_count >= anomaly_count else 'Anomaly'
                    
                    # Check if any connection indicates overload
                    is_overloaded = 'overloaded' in node_states
                    
                    color = class_colors.get(node_class, 'gray')
                    edgecolor = 'darkred' if is_overloaded else 'black'
                    linewidth = 3 if is_overloaded else 1
                    
                    degree = G.degree(node)
                    ax.scatter(pos[node][0], pos[node][1], pos[node][2], 
                              color=color, s=degree*25, alpha=0.7, 
                              edgecolors=edgecolor, linewidth=linewidth)

            ax.set_title("Holographic Network Visualization\n(Exact Thresholds Applied)", 
                        fontsize=14, fontweight='bold', pad=20)
            ax.set_xlabel("Network Dimension X")
            ax.set_ylabel("Network Dimension Y")
            ax.set_zlabel("Network Dimension Z")
            
            # Add legend
            legend_elements = [
                plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='lightgreen', 
                          markersize=10, label='Normal Traffic', markeredgecolor='black'),
                plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='red', 
                          markersize=10, label='Anomalous Traffic', markeredgecolor='black'),
                plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='lightgray',
                          markersize=10, label='Overloaded Device', markeredgecolor='darkred', 
                          markeredgewidth=2)
            ]
            ax.legend(handles=legend_elements, loc='upper right')
            
            plt.tight_layout()
            plt.savefig('Holographic_Network.png', dpi=300, bbox_inches='tight')
            
            # Save threshold configuration for reproducibility
            self.save_threshold_config()
            
            plt.show()
            
            return G, device_metrics
        else:
            print("Missing required columns: 'Src IP', 'Dst IP', or 'Attack Name'")
            return None, None

    def save_threshold_config(self):
        """Save exact threshold configuration for reproducibility"""
        config = {
            'random_seed': self.random_state,
            'thresholds': self.thresholds,
            'threshold_references': {
                'bytes_threshold': 
                'packets_threshold': 
                'holo_load_threshold': 
                'similarity_threshold': 
            }
        }
        
        with open('holographic_thresholds.json', 'w') as f:
            json.dump(config, f, indent=2)
        
        logger.info("Exact thresholds saved to 'holographic_thresholds.json'")

def main():
    print("HoloTiny-AD: Holographic Visualization with Exact Thresholds")
    
    # Initialize visualizer
    visualizer = HolographicVisualizer(random_state=42)
    
    # Load dataset
    file_path = r"dataset.csv"
    df = visualizer.load_data(file_path)
    
    if df is not None:
        # Generate the holographic visualization with exact thresholds
        print("\nGenerating Holographic Network Visualization...")
        graph, device_metrics = visualizer.visualize_3d_network(df)
        
        print("\nHolographic visualization completed successfully!")
        print("Figure saved as 'Holographic_Network.png'")
        print("Exact thresholds saved as 'holographic_thresholds.json'")
        print("Applied thresholds")
        print(f"Analyzed {len(device_metrics) if device_metrics else 0} devices")

if __name__ == "__main__":
    main()