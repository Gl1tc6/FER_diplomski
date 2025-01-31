from typing import List, Iterable, Dict, Any, Tuple, Union

NoneType = type(None)

class EdgePossibility:
    """
    Class representing a single possibility in the discrete edge weight distribution.

    Attributes
    ----------
    prob: float
        Probability of of the value.
    value: int
        The weight value.
    """

    def __init__(self, prob: float, value: int) -> None:
        self.prob = prob
        self.value = value

    def __str__(self) -> str:
        return f'{self.prob:2.f}: {self.value}'

    def __repr__(self) -> str:
        return f'EdgePossibility({self.prob}, {self.value})'

class EdgeWeightDistribution:
    """
    Distribution of weights for a single edge.

    No public attributes.
    """

    def __init__(self, *possibilities: EdgePossibility) -> None:
        self._possibilities = self.__normalize(possibilities)

    @staticmethod
    def __normalize(possibilities: Iterable[Union[EdgePossibility, tuple]]) -> Dict[int, float]:
        """
        Private static helper method used to normalize all of the probabilities so they sum to 1.

        Args:
            possibilities (Iterable[EdgePossibility): Collection of possible values which need to be normalized

        Returns:
            dict: A dictionary of values mapped to their respective probabilities.
        """
        possibilities = list(map(lambda p: p if isinstance(p, EdgePossibility) else EdgePossibility(*p), possibilities))
        total_prob = sum(map(lambda p: p.prob, possibilities))
        if total_prob == 0.00:
            total_prob = 1e-3
        return {pos.value: round(pos.prob / total_prob, 2) for pos in possibilities}

    def value_prob(self, value: int) -> float:
        """
        Fetch the probability of the weight taking the specified `value`.

        Args:
            value (int): The value whose probability we want to fetch.

        Returns:
            float: The probability of the value.
        """
        return self._possibilities.get(value, 0.0)

    def iter_possibilities(self):
        """
        Helper method used for iterating over the values and their probabilities in
        a for-each manner, i.e. `for value, prob in dist.iter_possibilities():...`

        Returns:
            Iterator over the possible values and their probabilities.
        """
        for val, prob in self._possibilities.items():
            yield val, prob

    def __repr__(self) -> str:
        possibilities_repr = ', '.join(map(lambda value_prob: f'EdgePossibility({value_prob[1]}, {value_prob[0]})', self.iter_possibilities()))
        return f'EdgeWeightDistribution({possibilities_repr})'
    
    def __eq__(self, other) -> bool:
        if isinstance(other, EdgeWeightDistribution):
            return self._possibilities == other._possibilities
        return False

ProbEdgeGraph = List[List[Union[EdgeWeightDistribution, NoneType]]]
"""
Graph represented as an adjacency matrix, but with distributions for edge weights instead
instead of actual values.
"""


def create_adj_matrix(n_nodes: int) -> ProbEdgeGraph:
    """
    Create a (weighted) adjacency matrix for a graph with `n_nodes` nodes.
    All of the weights are set to 0.

    Args:
        n_nodes (int): Number of nodes in the graph.

    Returns:
        ProbEdgeGraph: The adjacency matrix as a list of lists.
    """
    return [[None] * n_nodes for _ in range(n_nodes)]

class DisjointSetForest:
    """
    Class used to represent a forest of disjoint sets. Used
    for the union-find part of the Kruskal algorithm. Should be
    the compressed variant of the data structure.

    No public attributes.
    """

    def __init__(self):
        self._parents = {}

    @staticmethod
    def create_forest(nodes: Iterable[int]) -> 'DisjointSetForest':
        """
        Static helper method which creates a DisjointSetForest with
        with each node being in a set of its own, with itself as the representative of
        the set.

        Args:
            nodes (Iterable[int]): A iterable object which iterates over the node indices.
        Returns:
            DisjointSetForest: A new disjoint-set forest with a set for each of the nodes.
        """
        forest = DisjointSetForest()
        forest._parents.update({n: n for n in nodes})
        return forest

    def make_set(self, node: int) -> bool:
        """
        Create a new set in the disjoint-set forest with only the `node` in it and
        with the set having the node as the representative.

        Args:
            node (int): The node for which we are creating a new set in the forest.

        Returns:
            bool: True if it was able to create a new set (if the node is not already
            in the forest. False, otherwise.
        """
        if node in self._parents:
            return False
        self._parents[node] = node
        return True

    def find(self, node: int) -> int:
        """
        Find the representative of the set containing the `node`.
        Compress the set along the way.

        Args:
            node (int): The node for which to find the representative (of the set).

        Returns:
            int: The set representative.
        """
        # TODO: Find the node representative, compress the structure (if necessary) and return the
        # representative.
        pass

    def union(self, node_a: int, node_b: int) -> None:
        """
        Combine the sets containing `node_a` and `node_b` with the
        representative of `node_a`'s set being the new representative
        of the merged set.

        Args:
            node_a (int): A node from the first set.
            node_b (int): A node from the second set.
        """
        # TODO: Union the two sets.
    

def kruskal(G: ProbEdgeGraph) -> ProbEdgeGraph:
    """
    Implementation of the Kruskal algorithm on a ProbEdgeGraph.
    Returns the MST which gives us the best outcome over a range of traversals. The best average case MST.

    Args:
        G (ProbEdgeGraph): A graph whose edges have a probability distribution of weights.
    Returns:
        ProbEdgeGraph: The resulting MST.
    """
    n_nodes = len(G)
    forest = DisjointSetForest.create_forest(range(n_nodes))
    MST = create_adj_matrix(n_nodes)
    # TODO: Determine the MST
    return MST