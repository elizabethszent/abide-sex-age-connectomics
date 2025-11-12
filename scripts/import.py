import networkx as nx
G = nx.read_edgelist("Users\eliza\CPSC_599_CONNECTOMICS\TERMProject\results", nodetype=int, data=(("weight", int),))
ws = [d["weight"] for _,_,d in G.edges(data=True)]
print("min/max/mean weights:", min(ws), max(ws), sum(ws)/len(ws))