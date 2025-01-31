from collections import deque
from typing import List, Dict, Tuple

Graph = Dict[str, List[str]]
Circuit = List[str]
EulerPath = List[str]

def augmented_hierholzer(G: Graph, start: str) -> Tuple[EulerPath, List[Circuit]]:
    """
    Args:
        G (Graph): A Graph as an adjacency matrix. Assumed to be Eulerian.
        start (str): Starting node for the Hierholzer algorithm.
    Returns:
        Tuple[EulerPath, List[Circuit]]: A tuple containing an Eulerian path in the Euler graph
        and a list of all the circuits found on the path.
    """
    stack = deque()
    stack.append(start)
    
    path = []
    circuits = []
    current_circuit = []
    
    while stack:
        u = stack[-1]
        if G[u]:  
            v = G[u][0]
            stack.append(v)
            G[u].remove(v)
            G[v].remove(u)
        else:
            node = stack.pop()
            path.append(node)
            if current_circuit:
                current_circuit.append(node)
                if node == current_circuit[0]:
                    circuits.append(current_circuit[:])
                    current_circuit = []
            else:
                current_circuit = [node]
    
    # Merge overlapping circuits
    merged_circuits = []
    for circuit in circuits:
        if not merged_circuits:
            merged_circuits.append(circuit)
        else:
            merged = False
            for i, merged_circuit in enumerate(merged_circuits):
                # Check if the circuit overlaps with the merged circuit
                for j in range(len(merged_circuit)):
                    if merged_circuit[j] == circuit[0]:
                        # Merge the circuit into the merged circuit
                        merged_circuits[i] = merged_circuit[:j] + circuit + merged_circuit[j+1:]
                        merged = True
                        break
                if merged:
                    break
            if not merged:
                merged_circuits.append(circuit)
    
    return path, merged_circuits

if __name__ == "__main__":
    import copy

    G = {'a': ['b', 'c', 'd', 'e'],
        'b': ['a', 'd', 'e'],
        'c': ['a', 'e'],
        'd': ['a', 'b', 'e'],
        'e': ['a', 'b', 'c', 'd']}
        
    G1 = copy.deepcopy(G)

    path, circles = augmented_hierholzer(G1, 'b')
    path.reverse()
    print("Assert 1")
    print(circles)
    print([['d', 'e', 'b', 'd'], ['e', 'b', 'd', 'a', 'e'], ['a', 'e', 'c', 'a'], ['b', 'd', 'a', 'e', 'c', 'a', 'b']])
    assert path == ['b', 'a', 'c', 'e', 'a', 'd', 'b', 'e', 'd']
    assert circles == [['d', 'e', 'b', 'd'], ['e', 'b', 'd', 'a', 'e'], ['a', 'e', 'c', 'a'], ['b', 'd', 'a', 'e', 'c', 'a', 'b']]

    G1 = copy.deepcopy(G)

    path, circles = augmented_hierholzer(G1, 'd')
    path.reverse()
    print("Assert 2")
    assert path == ['d', 'a', 'b', 'd', 'e', 'a', 'c', 'e', 'b']
    assert circles == [['e', 'c', 'a', 'e'], ['b', 'e', 'c', 'a', 'e', 'd', 'b'], ['a', 'e', 'd', 'b', 'a'], ['d', 'b', 'a', 'd']]