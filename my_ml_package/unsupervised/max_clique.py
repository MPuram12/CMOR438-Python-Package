import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from scipy.optimize import linprog, milp, Bounds, LinearConstraint
import pandas as pd

class MaxCliqueFinder:
    def __init__(self, edge_data, directed):
        '''
        Initialize the class with edge data and build the adjacency matrix.
        '''
        self.edge_data = edge_data
        self.adj = self._build_adj(edge_data)
        self.directed = directed # whether graph is directed or not

    def _build_adj(self, data, directed):
        '''
        Given a data matrix that contains the edges, build the adjacency matrix for a graph.
        '''
        vertices = int(np.max(data))
        adj = np.zeros((vertices + 1, vertices + 1))
        for i in range(len(data)):
            u, v = int(data[i, 0]), int(data[i, 1])
            adj[u, v] = 1
            if self.directed = False:
              adj[v, u] = 1
        return adj

    def find_max_clique(self):
        '''
        Solves the maximum clique problem using MILP and returns the list of vertices in the max clique.
        '''
        l = len(self.adj)

        # Complementary adjacency matrix
        adjcomp = 1 - self.adj - np.eye(l)

        # Pairs of vertices not connected in the original graph
        v1, v2 = np.where(adjcomp == 1)

        # Constraints: no two non-connected vertices can both be in the clique
        A = np.zeros((len(v1), l))
        b = np.ones(len(v1))
        for i in range(len(v1)):
            A[i, v1[i]] = 1
            A[i, v2[i]] = 1

        # Objective: maximize number of nodes in the clique
        f = -np.ones(l)
        integrality = np.ones(l)
        bounds = Bounds(np.zeros(l), np.ones(l))
        constraints = LinearConstraint(A, -np.inf, b)

        # Solve MILP
        result = milp(c=f, integrality=integrality, bounds=bounds, constraints=constraints)
        clique_vertices = np.where(np.abs(result.x + f) < 1e-3)[0].tolist()

        return clique_vertices

    if __name__ == '__main__':
        edges = np.array([
        [0, 1],
        [1, 2],
        [2, 0],
        [0, 3]])

        finder = MaxCliqueFinder(edges)
        max_clique = finder.find_max_clique()
        print("Maximum Clique:", max_clique)
