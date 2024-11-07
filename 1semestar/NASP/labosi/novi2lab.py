from typing import Optional


class Node:
    """
    Class representing a single node of a binary tree containing integer values.

    ...

    Attributes
    ----------

    value: int
        Value stored in the node.
    parent: Node, optional
        Parent of the current node. Can be None.
    left: Node, optional
        Left child of the current node. Can be None.
    right: Node, optional
        Right child of the current node. Can be None.
    """

    def __init__(self, value: int) -> None:
        self.value = value
        self.parent = self.right = self.left = None

    def set_left_child(self, node: Optional["Node"]) -> None:
        """
        Set the the left child of self to the given node.
        Sets the node's parent to self (if it is not None).

        Args:
            node (Node, optional): the node to set as the child.
        """
        self.left = node
        if node is not None:
            node.parent = self

    def set_right_child(self, node: Optional["Node"]) -> None:
        """
        Set the the right child of self to the given node.
        Sets the node's parent to self (if it is not None).

        Args:
            node (Node, optional): the node to set as the child.
        """
        self.right = node
        if node is not None:
            node.parent = self

    def __repr__(self) -> str:
        """
        Get the string representation of the Node.

        Returns:
            str: A string representation which can create the Node object.
        """
        return f'Node({self.value})'



class BinaryTree:
    """
    Class repreesenting a binary tree, consisting of Nodes.

    ...

    Attributes
    ----------
    root : Node, optional
        the root node of the BinaryTree of type Node (or None)
    """

    def __init__(self, root: Optional[Node] = None) -> None:
        self.root = root

    def set_root(self, node: Optional[Node]) -> None:
        """
        Set the root of the tree to the provided node and set the node's parent to None (if the node is not None).

        Args:
            node (Node, optional): The Node object to set as the root (whose parent is set to None)
        """
        self.root = node
        if self.root is not None:
            self.root.parent = None

    def insert(self, value: int) -> bool:
        """
        Insert the given integer value into the tree at the right position.

        Args:
            value (int): The value to insert

        Returns:
            bool: True if the element was not already in the tree (insertion was successful), otherwise False.
        """
        node = self.root
        if node is None:
            self.set_root(Node(value))
            return True

        while node is not None:
            if value < node.value:
                if node.left is None:
                    node.set_left_child(Node(value))
                    break
                else:
                    node = node.left
            elif value > node.value:
                if node.right is None:
                    node.set_right_child(Node(value))
                    break
                else:
                    node = node.right
            else:
                return False
        return True


def get_successor(node: Optional[Node]) -> Optional[Node]:
    """
    Fetch the successor of the given node.
    If theere is no successor of the node return None.

    Args:
        node (Node, optional): The whose successor we want to fetch.

    Returns:
        Node, optional: The successor of the node or None if there isn't one.
    """
    if node is None or node.right is None:
        return None

    parent, current = node, node.right
    while current is not None:
        # TODO: Set the parent and current. When the successor is reached, current will be his child
        parrent = current
        current = current.left
    return parent


def _delete_single_child(tree: BinaryTree, node: Node) -> None:
    """
    Helper method used in the delete_by_copy function to delete nodes with only a single child or no children (leaves). 

    Args:
        tree (BinaryTree): The tree from which the node will be deleted
        node (Node): The node to delete from the tree
    """
    # TODO: Delete a node with only a single child
    if node.left is not None:
        child = node.left
    else:
        child = node.right
    
    if node.parent is None:
        tree.set_root(child)
    elif node == node.parent.left:
        node.parent.set_left_child(child)
    else:
        node.parent.set_right_child(child)
    
    if child is not None:
        child.parent = node.parent
    


def delete_by_copy(tree: BinaryTree, value: int) -> bool:
    """
    Deletes the value specifed as the argument from the tree if it exists.
    If the value was not deleted (i.e. was not in the tree) returns False, otherwise True.

    Args:
        value (int): The value to delete from the tree.

    Returns:
        bool: True if the value was in the tree and was successfuly deleted, otherwise False.
    """
    if tree is None or value is None or tree.root is None:
        return False

    node = tree.root
    
    while node is not None:
        # TODO: Get to the node which has node.value == value by moving through the tree
        if node.value == value:break
        if value < node.value:
            node = node.left
        elif value > node.value:
            node = node.right

    if node is None:
        return False

    node_to_delete = get_successor(node)
    if node_to_delete is None:
        node_to_delete = node

    node.value = node_to_delete.value
    _delete_single_child(tree, node_to_delete)
    return True