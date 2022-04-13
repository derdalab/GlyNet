#!/usr/bin/python3

#
# This program outputs a list of glycan structures
# Eric Carpenter - 2021-05-25
#
# usage: ./list-possible-glycans.py <size>
# where <size> is the number of monosaccharides in the outputs
#


import copy
import sys

glycan_size = int(sys.argv[1])   # monosaccharide number from the command line 

# dict of the monosaccharide names with list of which positions additional
# glycosidic bonds will be considered

monosaccharides = {
     'Gal': (1, 2, 3, 4, 6), 'Glc': (1, 2, 3, 4, 6), 'Man': (1, 2, 3, 4, 6),
     'GalNAc': (1, 3, 4, 6), 'GlcNAc': (1, 3, 4, 6), 'GlcA': (1, 2, 3, 4),
     'Fuc': (), 'Neu5Ac': (), 'Neu5Gc':(), 'Kdn':()}

# rewrite the above, doubling to list both alpha and beta forms of each unit 
can_follow_positions = {}
for k, v in monosaccharides.items():
    can_follow_positions['α'+k] = v
    can_follow_positions['β'+k] = v
units = list(can_follow_positions.keys())
can_follow_positions['Sp'] = (0,)


# data structure for tracking information about each node in the glycan tree
class node():
   def __init__(self, kind, links):
       self.kind = kind
       self.links = links  # list of (carbon number: node number) pairs

def print_subtree(link_from, node_list):
    # link_from: (carbon # on parent node, node number)
    root_node = node_list[link_from[1]]
    if len(root_node.links) == 0:
        out = ''
    else:
        # process any subtrees attached to the current one
        for i, link in enumerate(root_node.links.items()):
            if i == 0:
                out = print_subtree(link, node_list)
            else:  # after the first enclose subtree in []
                out += '[' + print_subtree(link, node_list) + ']'

    # append a string for the current node - special case for the root node
    if link_from[0] == 0:
        out = out + root_node.kind[1:] + '(' + root_node.kind[0] + '1-'
    else:
        out = out + '{}({}1-{})'.format(root_node.kind[1:],
                                        root_node.kind[0], link_from[0])
    return out

def print_glycan(node_list):
    #print(len(node_list) - 1, print_subtree((0, 1), node_list))
    print(print_subtree((0, 1), node_list))

def walk_node_tree(node_list, start = 0):
    # iterate over nodes in the tree adding new nodes
    for extend_i, extend_node in enumerate(node_list[start:], start = start):
        for from_carbon in can_follow_positions[extend_node.kind]:
            # proposed addition is on from_carbon of node# extend_i:
            # check that this carbon is available - max of 3 children
            # avoid generating both X(a1-6)[X(a1-3)] and X(a1-3)[X(a1-6)] by
            # restricting branches to carbons below occupied ones
            if from_carbon != 1 and len(extend_node.links) < 3 and (
                    len(extend_node.links) == 0 or from_carbon < min(extend_node.links)):
                for kind in units:
                    # add link from the extend_node to a new node
                    # and add the new node too
                    node_list[extend_i].links[from_carbon] = len(node_list)
                    node_list.append(node(kind, {}))
                    
                    # explore new trees derived from the current one
                    #print_glycan(node_list)
                    if len(node_list) <= glycan_size:
                         # to prevent A[B]C being generated from AC and BC
                         # restrict further search to sites past this one
                         # conveniently this includes the new node
                         walk_node_tree(node_list, start = extend_i)
                    else:
                        print_glycan(node_list)
                    
                    # remove the new node from the tree
                    del node_list[extend_i].links[from_carbon]
                    node_list.pop()

# run the code starting from a root node named "Sp"
walk_node_tree([node('Sp', {})])
