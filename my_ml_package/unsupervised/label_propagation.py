import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import matplotlib.patches as mpatches

class LabelPropagationGraph:
    """
    Label Propagation Graph

    Attributes:
        edge_data (np array): numpy array of shape (n_edges, 2) with each row containing the vertices connected by an edge
        directed (Boolean): true or false depending on whether the graph is directed or not
    """
    def __init__(self, edge_data, directed):
        """
        Initializes the graph with given edge data and directed, 
        """
        self.edge_data = edge_data
        self.directed = directed
        self.adj_matrix = self.build_adj(edge_data)
        self.labels = None # Labels of each vertex
        self.iterations = 0 # Keeps track of number of iterations used
        

    def build_adj(self, data):
        """
        Builds the adjacency matrix from edge data for a graph.

        Given a data matrix that contains the edges, build the adjacency matrix for a graph.

        Args:
            data: (n_edges, 2) dimensional np array that contains the edge data
        """
        vertices = np.max(data)
        adj = np.zeros((vertices + 1, vertices + 1))
        for i in range(len(data)):
            a, b = int(data[i, 0]), int(data[i, 1])
            adj[a, b] = 1
            if self.directed is False:
                adj[b, a] = 1 # undirected graphs
        return adj

    def run_label_propagation(self, max_iter=100):
        """
        Runs the label propagation algorithm on the graph.

        Args:
             max_iter: the max number of iterations the algorithm runs for before converging 
        """
        n = len(self.adj_matrix)
        # Initialize each node with label
        self.labels = np.arange(n)
        self.iterations = 0

        while True:
            updated = False

            for i in range(n):
                neighbors = np.where(self.adj_matrix[i] == 1)[0]
                if len(neighbors) == 0:
                    continue
                # Finding most common label amongst neighbors
                neighbor_labels = self.labels[neighbors]

                # Count how often each label appears
                unique_labels, counts = np.unique(neighbor_labels, return_counts=True)
                max_labels = unique_labels[counts == counts.max()]

                # Choose between ties randomly
                chosen_label = np.random.choice(max_labels)

                if self.labels[i] != chosen_label:
                    self.labels[i] = chosen_label
                    updated = True

            self.iterations += 1
            if not updated or self.iterations > max_iter:
                break

        return self.labels, self.iterations

    def plot(self):
        """
        Plots the graph with vertex colors corresponding to their labels.
        """
        if self.labels is None:
            raise ValueError("Run label propagation before plotting.")

        graph = nx.Graph(self.adj_matrix)
        unique_clusters = np.unique(self.labels)
        cmap = plt.cm.Set3
        colors = cmap(np.linspace(0, 1, len(unique_clusters)))

        pos = nx.spring_layout(graph, seed=42)
        node_colors = [colors[np.where(unique_clusters == label)[0][0]] for label in self.labels]

        plt.figure(figsize=(7, 5))
        nx.draw(
            graph,
            pos,
            node_color=node_colors,
            with_labels=False,
            node_size=100,
            edge_color='gray'
        )

        plt.title(f'Graph Colored by Label Propagation After {self.iterations} Iterations')

        legend_handles = [
            mpatches.Patch(color=colors[i], label=f'Cluster {label}')
            for i, label in enumerate(unique_clusters)
        ]
        plt.legend(handles=legend_handles, loc='best', title='Clusters', fontsize='small')
        plt.show()

if __name__ == '__main__':
    # Simple test case
    edges = np.array([
    [0, 1],
    [1, 2],
    [2, 3],
    [3, 4],
    [4, 0],
    [0, 2],
    [5, 6]])

    # Running the algorithm
    g = LabelPropagationGraph(edges, directed = False)
    g.run_label_propagation()

    # Plotting the graph with clusters
    g.plot()


