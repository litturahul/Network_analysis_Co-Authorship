import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from itertools import count

#read data
papers = pd.read_csv("C:/Users/Abhinay/Desktop/rahul/data.csv")
#select columns which are required
temp = papers[['Author', 'Article Title','Publication Date']]
#extracting the year of publication by cleaning
temp['Publication Date']=temp['Publication Date'].str.extract(r'(\d+)')
temp['Publication Date']=temp['Publication Date'].apply(lambda x: x if len(x)==4 else '19'+x)
#spliting the Authors which are sperated by semicolon we are doing this making the single column into multiple rows.
temp = temp.set_index(temp.columns.drop('Author', 1).tolist()).Author.str.split(';', expand=True).stack().reset_index().rename(columns={0: 'Author'}).loc[:, temp.columns]
#removing the extra spaces around the text
temp['Author'] = temp['Author'].apply(lambda x: pd.Series(x.strip()))
#removing the comma in between the firts name and last name
temp['Author'] = temp['Author'].str.replace(',', '')
temp[['First', 'Last']] = temp['Author'].str.split(' ', 1, expand=True)
#creating rank for each author worked for an article to create edges
temp['rank'] = temp.groupby('Article Title').cumcount()+1
#a = temp[temp['rank'] == 1]
#b = temp[temp['rank'] != 1]
edges=pd.DataFrame()
for name,group in temp.groupby('Article Title'):
    max_rank=max(group['rank'])
    if(max_rank>1):
        for i in range(1, max_rank+1):
            x = group[group['rank'] == i]
            y = group[group['rank'] > i]
            z = pd.merge(x, y, on='Article Title', how='outer')
            edges = edges.append(z, ignore_index=True)

edges = edges[edges['rank_y'].notna()]

df = temp.merge(edges, on="Article Title", how="outer", indicator=True)
df = df[df['_merge'] == 'left_only']

#creating edges
#edges = pd.merge(a, b, on='Article Title', how='outer')
#cleaning edges
df['Author_x'].fillna(df['Author'],inplace=True)
df['Author_y'].fillna(df['Author'],inplace=True)
df['First_x'].fillna(df['First'],inplace=True)
df['First_y'].fillna(df['First'],inplace=True)
df['Last_y'].fillna(df['Last'],inplace=True)
df['Last_x'].fillna(df['Last'],inplace=True)
df['Publication Date_x'].fillna(df['Publication Date'],inplace=True)
df['Publication Date_y'].fillna(df['Publication Date'],inplace=True)
df['rank_y'].fillna(df['rank'],inplace=True)
df['rank_x'].fillna(df['rank'],inplace=True)
del df['Author']
del df['First']
del df['Last']
del df['Publication Date']
del df['rank']
del df['_merge']

edges = edges.append(df,ignore_index=True)

#creating nodes
#nodes=temp.Author.unique()
a = temp.Author.value_counts().to_frame()
a.reset_index(level=0, inplace=True)
a = a.rename(columns={'index': 'Author', 'Author': 'count'})
nodes = pd.merge(pd.DataFrame(temp.Author.unique().tolist(), columns=["Author"]), a, on="Author", how="inner")

#Building Network Graph
G = nx.Graph()
#Adding Nodes with Node Attributes
for x in nodes.to_numpy():
    G.add_node(x[0])
    G.node[x[0]]['count']=x[1]

#adding metrics of nodes as node attribute
nx.set_node_attributes(G,nx.degree_centrality(G),'degree_centrality')
nx.set_node_attributes(G,nx.betweenness_centrality(G),'betweenness_centrality')
nx.set_node_attributes(G,nx.closeness_centrality(G),'closeness_centrality')
nx.set_node_attributes(G,nx.eigenvector_centrality(G),'eigen_vector_centrality')
nx.set_node_attributes(G,nx.clustering(G),'clustering')

#adding Edges with edge attributes
for x in edges.to_numpy():
    G.add_edge(x[1], x[2])
    G[x[1]][x[2]]["paper"] = x[0]
    G[x[1]][x[2]]["year"] = x[8]
    G[x[1]][x[2]]["family_names"] = x[3]+" & "+x[4]

#extracting the node size and edge labels
edge_labels = {e: G.get_edge_data(e[0], e[1])["family_names"] for e in G.edges()}
node_size=nx.get_node_attributes(G,'count')
#drawing Network Graph
pos=nx.spring_layout(G)
nx.draw(G,pos)
nx.draw_networkx_nodes(G,pos,node_color='b',node_size=[s*1000 for s in node_size.values()],alpha=0.8)
nx.draw_networkx_labels(G,pos,labels={k:k.split(" ")[0] for k,v in node_size.items()},font_size=8)
nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels,font_size=8)

