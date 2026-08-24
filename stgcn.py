import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class GCNLayer(nn.Module):
    def __init__(self, in_features, out_features):
        super(GCNLayer,self).__init__()
        self.W=nn.Parameter(torch.FloatTensor(in_features,out_features))
        self.reset_parameters()

    def reset_parameters(self):
        stdv=1./math.sqrt(self.W.size(1))
        self.W.data.uniform_(-stdv,stdv)

    def forward(self,X,A_hat):
        support=torch.matmul(X,self.W)
        out=torch.matmul(A_hat,support)
        return F.relu(out)
    
class SelfAttention(nn.Module):
    def __init__(self,input_dim, output_dim, hidden_dim):
        super(SelfAttention,self).__init__()
        self.Q=nn.Linear(input_dim, hidden_dim)
        self.K=nn.Linear(input_dim, hidden_dim)
        self.V=nn.Linear(input_dim, hidden_dim)

        self.out_proj=nn.Linear(hidden_dim,output_dim)

    def forward(self,X):
        Q=self.Q(X)
        K=self.K(X)
        V=self.V(X)

        scores=torch.matmul(Q,K.transpose(-2,-1))/math.sqrt(K.size(-1))
        attn=F.softmax(scores,dim=-1)
        out=torch.matmul(attn,V)

        return self.out_proj(out)

# gcn_dim=16
# gru_hidden_dim=32
# gru_output_dim=16
# attention_hidden_dim=64
# attention_output_dim=16
# fc_hidden_dim=16

gcn_dim=32
gru_hidden_dim=64
gru_output_dim=32
attention_hidden_dim=64
attention_output_dim=32
fc_hidden_dim=64

class GRU_GCN_Attention(nn.Module):
    def __init__(self,num_nodes,in_features,timesteps_input,timesteps_output):
        super(GRU_GCN_Attention,self).__init__()

        self.num_nodes=num_nodes
        self.timesteps_input = timesteps_input
        self.timesteps_output = timesteps_output
        self.dropout=nn.Dropout(p=0.2)

        self.gru=nn.GRU(input_size= in_features,
                        hidden_size= gru_hidden_dim,
                        num_layers=2,
                        dropout=0.2,
                        batch_first=True)
        
        self.gru_proj=nn.Linear(gru_hidden_dim,gru_output_dim)

        self.gcn1=GCNLayer(in_features,gcn_dim)
        self.gcn2=GCNLayer(gcn_dim,gcn_dim)

        fusion_dim=gcn_dim+gru_output_dim

        self.attention=SelfAttention(input_dim=fusion_dim,
                                    output_dim=attention_output_dim,
                                    hidden_dim=attention_hidden_dim)

        self.pred_gcn=GCNLayer(attention_output_dim, attention_output_dim)

        self.fc1=nn.Linear(attention_output_dim*timesteps_input,fc_hidden_dim)
        self.fc2=nn.Linear(fc_hidden_dim,timesteps_output)

    def forward(self,A_hat,X):
        B,N,T,F=X.shape

        X_time=X.reshape(B*N,T,F)
        gru_out,_=self.gru(X_time)
        gru_out=self.gru_proj(gru_out)
        gru_out=gru_out.reshape(B,N,T,-1)

        gcn_outputs=[]
        for t in range(T):
            Xt=X[:,:,t,:]
            gcn_t=self.gcn1(Xt,A_hat)
            gcn_t=self.dropout(gcn_t)
            gcn_t=self.gcn2(gcn_t,A_hat)
            gcn_outputs.append(gcn_t.unsqueeze(2))

        gcn_out=torch.cat(gcn_outputs,dim=2)

        fusion=torch.cat([gru_out,gcn_out],dim=-1)
        
        #暂时试一下时空联合注意力
        fusion=fusion.permute(0,2,1,3)    #(B,T,N,F)
        fusion=fusion.reshape(B,T*N,fusion.shape[-1])

        attn_out=self.attention(fusion)
        attn_out=self.dropout(attn_out)

        attn_out=attn_out.reshape(B,T,N,attention_output_dim)
        attn_out=attn_out.permute(0,2,1,3)

        #######
        #时间注意力
        # fusion=fusion.reshape(B*N,T,-1)
        #######

        ##########
        #暂时试一下空间注意力
        # (B,N,T,F)
        # fusion=fusion.permute(0,2,1,3)
        # # (B*T,N,F)
        # fusion=fusion.reshape(B*T,N,fusion.shape[-1])
        ######

        #######
        #时间注意力
        # attn_out = attn_out.reshape(B, N,T, attention_output_dim)
        ########

        ##########
        #暂时试一下空间注意力
        # attn_out = attn_out.reshape(B, T,N, attention_output_dim)
        # attn_out=attn_out.permute(0,2,1,3)
        ########

        pred_gcn_outputs=[]
        for t in range(T):
            Xt=attn_out[:,:,t,:]
            gcn_t=self.pred_gcn(Xt,A_hat)
            pred_gcn_outputs.append(gcn_t.unsqueeze(2))

        attn_out_gcn = torch.cat(pred_gcn_outputs,dim=2)

        out_flat = attn_out_gcn.reshape(B,N,-1)

        out = torch.nn.functional.relu(self.fc1(out_flat))
        out=self.dropout(out)
        out=self.fc2(out)
        return out