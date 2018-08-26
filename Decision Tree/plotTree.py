import pydot

def walkXmlDict(graph, xmlDict, parentNode = None):
    '''
    Recursive plotting function for the decision tree stored as a xmlDict
    '''
    for k, v in xmlDict.items():
        if parentNode is not None:
            fromName = parentNode.get_name().replace("\"", "") + '_' + str(k)
            fromLabel = str(k)

            nodeFrom = pydot.Node(fromName, label = fromLabel)
            graph.add_node(nodeFrom)
            graph.add_edge( pydot.Edge(parentNode, nodeFrom) )

            if isinstance(v, dict): # if interim node
                walkXmlDict(graph, v, nodeFrom)
            elif isinstance(v, list): # if interim node is list
                for item in v:
                    if isinstance(item, dict):
                        walkXmlDict(graph, item, nodeFrom)
                    else:
                        toName = str(k) + '_' + str(item) # unique name
                        toLabel = str(item)

                        nodeTo = pydot.Node(toName, label = toLabel, shape = 'box')
                        graph.add_node(nodeTo)
                        graph.add_edge(pydot.Edge(nodeFrom, nodeTo))
            else: # if leaf node
                toName = str(k) + '_' + str(v) # unique name
                toLabel = str(v)

                nodeTo = pydot.Node(toName, label = toLabel, shape = 'box')
                graph.add_node(nodeTo)
                graph.add_edge(pydot.Edge(nodeFrom, nodeTo))
        else:

            fromName =  str(k)
            fromLabel = str(k)

            nodeFrom = pydot.Node(fromName, label = fromLabel)
            walkXmlDict(graph, xmlDict[k], nodeFrom)


def plotTree(tree): # tree as dictionary
    # first you create a new graph, you do that with pydot.Dot()
    graph = pydot.Dot(graph_type='graph')

    walkXmlDict(graph, tree)

    return graph
