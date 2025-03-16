from pyvis.network import Network

def render_graph(graph, 
                 target, 
                 name='karateclub', 
                 color_dct={0:'red',1:'green'}):
    
    nodes = graph.nodes.keys()
    edges = graph.edges()
    
    nodes = [int(i) for i in nodes]
    colors = [color_dct[i] for i in target]
    labels = [str(i) for i in nodes]
    
    net = Network(notebook=False,
                  bgcolor='#222222',
                  font_color = "white",
                  height='1000px',
                  width='1000px')
    
    net.add_nodes(nodes, color=colors, label=labels)
    net.add_edges(edges)
    
    net.force_atlas_2based()
    net.show(f'{name}.html', notebook=False, local=False)