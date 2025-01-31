import random
from math import ceil, floor, log
from typing import List


class SkipNode:
    """Class for a node in skip list
    
    ...
    
    Attributes
    ----------
    
    predecessor: SkipNode
        Pointer to predecessor node on THE lowest level.
    pointers: List[SkipNode | None]
        pointers to successors (over full height of the node)
    value: int
        value stored in the node 
    """

    def __init__(self, value: int, pred_node: 'SkipNode' = None, succ_node: 'SkipNode' = None):
        """Creates skip node for skip list.
        
        Args:
            value (int): value to be stored
            pred_node (SkipNode, optional): predecessor node on the lowest level. Defaults to None.
            succ_node (SkipNode, optional): successor node on the lowest level. Defaults to None.
        """
        self.value = value
        self.pointers = [None]
        
        if pred_node is not None:
            pred_node.pointers[0] = self
        self.predecessor = pred_node
        
        if succ_node is not None:
            self.pointers[0] = succ_node
            succ_node.predecessor = self



class SkipList:
    """Class for skip list
    
    ...
    
    Attributes
    ----------
    
    n: int
        Expected capacity of the list.
    p: float
        Probability of jumping up by one level.
    max_level: int
        Maximum possible level, based on calculation from probability and capacity.
    head: SkipNode
        The first node in the list.
    histogram: List[int]
        The skip list histogram used for inserting values.
    """

    def __init__(self, n: int = 10000, p: float = 0.5):
        """Creates skip list and initializes all the necessary
        parameters

        Args:
            n (int, optional): expected capacity. Defaults to 10000.
            p (float, optional): probability of jumping up by one level. Defaults to 0.5.
        """
        self.p = p
        self.n = n
        self.max_level = ceil(log(n, 1/p))# TODO: fix calculation for max_level
        self.head = SkipNode(None)
        self.head.pointers = [None]*self.max_level
        self.histogram = self._createHistogram()

    def _createHistogram(self) -> List:
        """Creates histogram of degrees.

        Returns:
            List: histogram according to the third method from the lecture
        """
        H = [0]*(self.max_level+1)
        for i in range(1, self.max_level+1):
            H[i] = ceil(self.n * (1-(self.p ** i)))# TODO: fix value for the histogram
        return H
    
    def sampleLevel(self) -> int:
        """Randomly decides the height of the node,
        according to the histogram, using only one sample!

        Returns:
            int: returns valid randomly selected height
        """
        ctoss = random.randint(1, self.n)
        i = 1
        while i <= self.max_level and ctoss > self.histogram[i]: # TODO: fix the condition for sampling!
            i += 1
        return i

    def insert(self, value: int) -> None:
        """Inserts an element into the skip list if it has not already existed.
        Duplicates are not allowed.

        Args:
            value (int): key to insert into the skip list
        """
        level = self.sampleLevel()

        pred_node = self._search(value)
        if pred_node.value == value:
            return
        succ_node = pred_node.pointers[0]

        new_node = SkipNode(value, pred_node=pred_node)
        new_node.pointers = [None]*level
        new_node.pointers[0] = succ_node

        #TODO: update all the pointers on necessary levels. 
        # Hint: Also use pred_node for backsearch over the levels to update. Or, use alternative way!
        for i in range(1, level):
            while pred_node is not None and len(pred_node.pointers) <= i:
                pred_node = pred_node.predecessor
            if pred_node is not None:
                new_node.pointers[i] = pred_node.pointers[i]
                pred_node.pointers[i] = new_node


    def _search(self, value: int) -> SkipNode:
        """Searches for the element in a list

        Args:
            value (int): search key

        Returns:
            SkipNode: skiplist node containing the search key, if it exists.
            Otherwise it returns the last element of the skip list.
        """
        node = self.head
        curr_height = len(node.pointers)-1
        while curr_height >= 0:
            next_node = node.pointers[curr_height]
            if next_node is None or next_node.value > value: #TODO: add the missing condition here
                curr_height = curr_height-1
            else:
                node = next_node
        return node
    
    def search(self, value: int) -> SkipNode:
        """tidy wrapper around _search

        Args:
            value (int): search key to find

        Returns:
            SkipNode: Returns the found node or None.
        """
        node = self._search(value)
        if node.value == value:
            return node
        else:
            return None


    def __str__(self) -> str:
        """string representation of skip list - useful for debugging

        Returns:
            str: string representation
        """
        curr_node = self.head.pointers[0]
        str = ''
        while curr_node is not None:
            str = str + \
                f"(Node {curr_node.value},level={len(curr_node.pointers)}),"
            curr_node = curr_node.pointers[0]
        return str