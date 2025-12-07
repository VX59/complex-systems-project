from project import SocialNetowrk
import networkx as nx
S = SocialNetowrk(n=500,k=2)
print(nx.clustering(S.G))