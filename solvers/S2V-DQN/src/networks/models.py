import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import aggr
from torch_geometric.nn.pool import global_add_pool,global_mean_pool

from torch_geometric.utils import degree



class S2VDQN_layer(nn.Module):

    def __init__(self,input_dim,hidden_dim):

        super(S2VDQN_layer, self).__init__()
        

        self.theta_1=nn.Linear(input_dim,hidden_dim,bias=False)
        self.theta_2=nn.Linear(hidden_dim,hidden_dim,bias=False)
        self.theta_3=nn.Linear(hidden_dim,hidden_dim,bias=False)
        self.theta_4=nn.Linear(1,hidden_dim,bias=False)
        self.aggr=aggr.SumAggregation()
        

    def forward(self,x,edge_index,edge_attr,node_embedding):

        row,col=edge_index

        node_feature_embedding=self.theta_1(x)

        node_embedding_aggr=self.aggr(node_embedding[col],
                                          row,
                                          dim=0,
                                          dim_size=len(x)
                                        )

        edge_feature_embedding=F.relu(self.theta_4(edge_attr))

        edge_embedding_aggr=self.aggr(edge_feature_embedding,
                                          row,
                                          dim=0,
                                          dim_size=len(x)
                                        )
        
        node_embedding=F.relu(node_feature_embedding+self.theta_2(node_embedding_aggr)
                                  +self.theta_3(edge_embedding_aggr))
        
        return node_embedding
    
class S2VDQN(nn.Module):

    def __init__(self, input_dim, hidden_dim, hop):
        super(S2VDQN, self).__init__()
        self.hidden_dim=hidden_dim

        self.device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.layers=nn.ModuleList([
            S2VDQN_layer(input_dim=input_dim,hidden_dim=hidden_dim) for _ in range(hop)
        ])
        self.theta_5=nn.Linear(2*hidden_dim,1)
        self.theta_6=nn.Linear(hidden_dim, hidden_dim,bias=False)
        self.theta_7=nn.Linear(hidden_dim, hidden_dim,bias=False)

        self.device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def forward(self,data):

        x,edge_index,edge_attr=data.x,data.edge_index,data.edge_attr

        node_embedding=torch.zeros(size=(len(x),self.hidden_dim),device=self.device)

        for layer in self.layers:
            node_embedding=layer(x,edge_index,edge_attr,node_embedding)

        sum_node_embedding = self.theta_6(global_add_pool(node_embedding,data.batch)) 

        return self.theta_5(F.relu(torch.cat([sum_node_embedding[data.batch],self.theta_7(node_embedding)],dim=-1)))
# class S2V(nn.Module):

#     def __init__(self, input_dim, hidden_dim, output_dim,hop):
#         super(S2V, self).__init__()

#         self.hidden_dim=hidden_dim

#         self.theta_1=nn.Linear(input_dim,hidden_dim)
#         self.theta_2=nn.Linear(hidden_dim, hidden_dim)
#         self.theta_3=nn.Linear(hidden_dim, hidden_dim)
#         self.theta_4=nn.Linear(1, hidden_dim)
#         self.theta_5=nn.Linear(2*hidden_dim,output_dim)
#         self.theta_6=nn.Linear(hidden_dim, hidden_dim)
#         self.theta_7=nn.Linear(hidden_dim, hidden_dim)
#         self.aggr=aggr.SumAggregation()
#         self.hop=hop
#         self.device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#     def forward(self,data):
#         x,edge_index,edge_attr=data.x,data.edge_index,data.edge_attr

        
#         row,col=edge_index

#         node_embedding=torch.zeros(size=(len(x),self.hidden_dim),device=self.device)

#         node_feature_embedding=self.theta_1(x)

#         for _ in range(self.hop):

#             node_embedding_aggr=self.aggr(node_embedding[col],
#                                           row,
#                                           dim=0,
#                                           dim_size=len(x)
#                                         )
            
#             edge_feature_embedding=F.relu(self.theta_4(edge_attr))

#             edge_embedding_aggr=self.aggr(edge_feature_embedding,
#                                           row,
#                                           dim=0,
#                                           dim_size=len(x)
#                                         )
            
#             node_embedding=F.relu(node_feature_embedding+self.theta_2(node_embedding_aggr)
#                                   +self.theta_3(edge_embedding_aggr))
            
#         sum_node_embedding = self.theta_6(global_add_pool(node_embedding,data.batch)) 
#         # print(sum_node_embedding.shape)

#         return self.theta_5(F.relu(torch.cat([sum_node_embedding[data.batch],self.theta_7(node_embedding)],dim=-1)))

#         # if data.batch is not None:
#         #     return self.theta_5(F.relu(torch.cat([sum_node_embedding[data.batch],self.theta_7(node_embedding)],dim=-1)))
#         # else:
#         #     return self.theta_5(F.relu(torch.cat([sum_node_embedding[torch.zeros(len(x),dtype=torch.long)],
#         #                                           self.theta_7(node_embedding)],dim=-1)))

        

        
        


        


class ECODQN_layer(nn.Module):

    def __init__(self, dim_embedding):
        super().__init__()

        # eq (6)
        self.message = nn.Sequential(
            nn.Linear(2 * dim_embedding, dim_embedding,bias=False),
            nn.ReLU()
        )

        # eq (7)
        self.update = nn.Sequential(
            nn.Linear(2 * dim_embedding, dim_embedding,bias=False),
            nn.ReLU()
        )

        self.mean_aggr=aggr.MeanAggregation()

    # def forward(self, x, edge_index, edge_attr, x_agg_emb, *args, **kwargs):
    def forward(self, x, edge_index, edge_attr, x_agg_emb):
        
        row,col = edge_index # row src, col target

        x_agg = self.mean_aggr(
            edge_attr * x[col],
            row,
            dim=0,
            dim_size=len(x)
        )
        m = self.message(
            torch.cat([x_agg, x_agg_emb], dim=-1)
        )
        x = self.update(torch.cat([x, m], dim=-1))
        return x


class MPNN(nn.Module):

    def __init__(
            self,
            dim_in=7,
            dim_embedding=64,
            num_layers=3,
    ):
        super().__init__()

        # eq (4)
        self.embed_node = nn.Sequential(
            nn.Linear(dim_in, dim_embedding,bias=False),
            nn.ReLU()
        )

        # eq (5) inner
        self.embed_node_and_edge = nn.Sequential(
            nn.Linear(dim_in + 1, dim_embedding - 1,bias=False),
            nn.ReLU()
        )

        # eq (5) outer
        self.embed_agg_nodes_and_degree = nn.Sequential(
            nn.Linear(dim_embedding, dim_embedding,bias=False),
            nn.ReLU()
        )

        # eq (6-7)
        self.layers = nn.ModuleList([
            ECODQN_layer(dim_embedding) for _ in range(num_layers)
        ])

        # eq (8) inner
        self.agg_nodes = nn.Sequential(
            nn.Linear(dim_embedding, dim_embedding,bias=False),
            nn.ReLU()
        )

        # eq (8) outer
        self.readout = nn.Sequential(
            # nn.ReLU(),
            nn.Linear(2*dim_embedding, 1),
        )

        self.mean_aggr=aggr.MeanAggregation()

    def forward(self, data):

        x, edge_index, edge_attr=data.x,data.edge_index,data.edge_attr 
        row, col = edge_index


        x_emb = self.embed_node(x)

        node_edge=torch.cat([edge_attr,x[col]],dim=-1)
        node_edge = self.embed_node_and_edge(node_edge)
        



        node_edge = self.mean_aggr(
            node_edge,
            row,
            dim=0,
            dim_size = len(x)
        )
        
        degree_norm = degree(col,num_nodes=len(x),dtype=torch.float).unsqueeze(-1)

        # degree_norm = scatter(degree, batch, reduce='max', dim=0)[batch]
        # degree_norm = self._expand_as_over_tradj(degree_norm, node_edge)
        x_agg_emb = self.embed_agg_nodes_and_degree(
            torch.cat([node_edge, degree_norm], dim=-1)
        )

        for layer in self.layers:
            x_emb = layer(x_emb, edge_index, edge_attr, x_agg_emb)


        g_agg = global_mean_pool(
            x_emb, data.batch
        )
        g_agg = self.agg_nodes(g_agg)
        inp = torch.cat([g_agg[data.batch], x_emb], dim=-1)

        q_vals = self.readout(inp)

        return q_vals


class LSDQN_layer(nn.Module):
    
    def __init__(self,input_dim,hidden_dim):
        super(LSDQN_layer,self).__init__()

        self.linear_0=nn.Linear(input_dim, hidden_dim)
        self.linear_1=nn.Linear(hidden_dim,hidden_dim)
        self.linear_2=nn.Linear(input_dim,hidden_dim)
        self.linear_3=nn.Linear(1,input_dim)
        self.mean_aggr=aggr.MeanAggregation()

    def forward(self,x,edge_index,edge_attr,u):
        row,col=edge_index
        first_term=self.linear_0(x)

        u_agg=self.mean_aggr(
              edge_attr * u[col],
                row,
                dim=0,
                dim_size=len(x)

            )
        
        second_term=self.linear_1(u_agg)

        edge_embed=F.relu(self.linear_3(edge_attr))

        edge_embed=self.mean_aggr(
            edge_embed,
            row,
            dim=0,
            dim_size=len(x)
        )
        third_term=self.linear_2(edge_embed)
        u=F.relu(first_term+second_term+third_term)

        return u


class LSDQN(nn.Module):
    def __init__(self,input_dim,hidden_dim,hop):

        super().__init__()
        

        self.hidden_dim=hidden_dim

        self.device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.layers=nn.ModuleList([
            LSDQN_layer(input_dim=input_dim,hidden_dim=hidden_dim) for _ in range(hop)
        ])

        self.attention=nn.Linear(2*hidden_dim,hidden_dim)

        self.space_embed=nn.Linear(hidden_dim,hidden_dim)
        self.action_embed=nn.Linear(2*hidden_dim,hidden_dim)

        self.last_layer=nn.Linear(2*hidden_dim,1)

        self.hop=hop
        self.mean_aggr=aggr.MeanAggregation()
        self.device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def forward(self,data):


        x,edge_index,edge_attr=data.x,data.edge_index,data.edge_attr
        edge_attr=edge_attr.view(-1,1)

        u=torch.zeros(size=(x.shape[0],self.hidden_dim),device=self.device)

        for layer in self.layers:
            u=layer(x,edge_index,edge_attr,u)


       
        h_c_0=global_mean_pool(
                u*(1-x),
                data.batch,
                )/global_mean_pool(
                1-x,
                data.batch)
                
        
        h_c_1=global_mean_pool(
                u*x,
                data.batch,
                )/global_mean_pool(
                x,
                data.batch)

        target_cluster_embed=x*h_c_0[data.batch]+(1-x)*h_c_1[data.batch]
        h_a=torch.cat([u,target_cluster_embed],dim=-1)

        _attention=self.attention(h_a)
        w_0=torch.sum(_attention*h_c_0[data.batch],axis=1,keepdim=True)
        w_1=torch.sum(_attention*h_c_1[data.batch],axis=1,keepdim=True)
        w=F.softmax(torch.cat([w_0,w_1],dim=-1),dim=-1)
        
        h_s=w[:,0][:, None]*h_c_0[data.batch]+w[:,1][:, None]*h_c_1[data.batch]
        q=self.last_layer(torch.cat([self.space_embed(h_s),self.action_embed(h_a)],dim=-1))

        return q



class LinearRegression(nn.Module):
    def __init__(self,input_dim=2):
        super().__init__()
        self.linear=nn.Linear(input_dim,1)

    def forward(self,data):

        return self.linear(data.x[:,1:3])