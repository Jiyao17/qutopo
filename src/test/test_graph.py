
from ..task import GroundNetwork, GroundTopology
import networkx as nx



att = GroundTopology(GroundNetwork.ATT)
edge0 = att.net.edges(data=True)[0]
print(edge0)
