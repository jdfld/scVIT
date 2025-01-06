import sklearn as sk
from sklearn.metrics import accuracy_score
from pytorch_forecasting.metrics.distributions import NegativeBinomialDistributionLoss as NBDL
import numpy as np
from torch import nn
import loss as l
import copy
import torch
import torchvision as tv
from anndata.experimental import AnnLoader
from typing import Callable, List, Optional, Sequence, Tuple, Union


class BasicNeuralNetwork(nn.Module): # basic neural network architecture with dropout and ReLu as activation
    def __init__(self,emb_genes,cell_types, encoder,layer_dims, lr,double=False,copy=False,norm='layer'):
        super().__init__()
        self.emb_count = len(emb_genes)
        self.emb_genes = emb_genes
        self.cell_count = len(cell_types)
        self.cell_types = emb_genes
        self.lr = lr
        self.encoder = encoder
        if copy:
            self.device = None
            self.layer_dims = None
            self.model = None
            return
        self.layer_dims = [self.emb_count]+list(filter(lambda x:x!=0,layer_dims))+[self.cell_count]
        model_layers = []
        step = 1+double
        for i in range(1,len(self.layer_dims)-1,step):
            if i > 1: # Do not want to drop from first layer due to sparse input
                model_layers.append(nn.Dropout(p=0.4))
            model_layers.append(nn.Linear(self.layer_dims[i-1],self.layer_dims[i]))
            if double == True:
                model_layers.append(nn.Linear(self.layer_dims[i],self.layer_dims[i+1],bias=False))
            model_layers.append(nn.LeakyReLU())
            if norm == 'layer':
                model_layers.append(nn.LayerNorm(self.layer_dims[i+double]))
            if norm == 'batch':
                model_layers.append(nn.BatchNorm1d(self.layer_dims[i+double]))
        model_layers.append(nn.Linear(self.layer_dims[-2],self.layer_dims[-1]))
        model_layers.append(nn.LogSoftmax(dim=1))
        self.model = nn.Sequential(*model_layers)

        def init_kaiming(module):
            if type(module) == nn.Linear:
                nn.init.kaiming_uniform_(module.weight,nonlinearity='leaky_relu')
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

        self.model.apply(init_kaiming)

        device = 'cpu'
        if torch.cuda.is_available():
            device = "cuda:0"
        elif torch.backends.mps.is_available():
            if not torch.backends.mps.is_built():
                print("MPS not available because the current PyTorch install was not "
                    "built with MPS enabled.")
            device = 'mps'
        #self.to(device)
        self.device = device

    def freeze(self):
        for p in self.model.parameters():
            p.requires_grad = False
        
    def reheat(self):
        for p in self.model.parameters():
            p.requires_grad = True
    

    def copy_model(self,new_cells,new_encoder): # creates a shallow copy of the object itself
        new_model = BasicNeuralNetwork(self.emb_genes,new_cells,new_encoder,None,self.lr,copy=True)
        new_model.layer_dims = self.layer_dims[:-1]
        new_model.layer_dims.append(new_model.cell_count)
        new_model.model = self.model[:-2] # remove previous output layer
        new_model.model.append(nn.Linear(self.layer_dims[-2],new_model.cell_count)).append(nn.LogSoftmax(dim=1))
        new_model.to(self.device)
        new_model.device = self.device
        return new_model

    def forward(self,x):
        return self.model(x)

    def predict_decode(self,x):
        y = self.predict(x)
        return self.encoder.inverse_transform(y)

    def predict(self,x):
        y = self.forward(x)
        y = y.cpu()
        return y.detach().numpy().argmax(axis=1)
    

    def predict_acc(self,X,y):
        y = y.cpu()
        return accuracy_score(y_true=y,y_pred=self.predict(X))

class TransformerAutoencoder(nn.Module):
    def __init__(self,input_size, soih, enc_lin, dec_lin,lr,conv_layers=0, layers = "patchwise", normalization="batch" ,  loss = "MSE"):
        super().__init__()
        self.input_size = input_size
        self.loss = loss
        self.soih = soih
        self.lr = lr
        if layers == "patchwise":
            lin = PatchwiseLinear
        else:
            lin = nn.Linear
        # Patchify!
        prev_c = 1
      
        patchify = []
        patchify.append(torch.nn.Unflatten(0,(-1,1)))
        patchify.append(Conv1dNormActivation(in_channels=prev_c,out_channels=soih[0][0],stride=soih[0][0], kernel_size=soih[0][0]))
        prev_c = soih[0][0]
        for i in range(conv_layers):
            patchify.append(Conv1dNormActivation(in_channels=prev_c,out_channels=soih[0][0], kernel_size=5))
            prev_c = soih[0][0]
        patchify.append(nn.Conv1d(prev_c,soih[0][0],kernel_size=1))
        patchify.append(nn.Flatten(start_dim=1))
        self.patchify = nn.Sequential(*patchify)
        # Encoder
        encoder = []
        for i in range(len(soih)):
            encoder.append(nn.Unflatten(1,(soih[i][0],soih[i][2])))
            encoder.append(nn.BatchNorm1d(soih[i][0], eps=1e-05))
            encoder.append(PatchwiseLinear(soih[i][0], soih[i][1],soih[i][2]))
            #encoder.append(nn.Linear(soih[i][2], soih[i][1]))
            encoder.append(nn.Dropout(p=0.5))
            encoder.append(nn.BatchNorm1d(soih[i][0], eps=1e-05))
            encoder.append(nn.MultiheadAttention(soih[i][1],soih[i][3],dropout=0.5, batch_first = True))
            encoder.append(nn.Dropout(p=0.5))
            encoder.append(nn.SELU())
            encoder.append(nn.Flatten(start_dim=1))
        last = soih[-1][0]*soih[-1][1]
        for i in range(len(enc_lin)):
            encoder.append(nn.Linear(last,enc_lin[i]))
            encoder.append(nn.Dropout(p=0.5))
            encoder.append(nn.SELU())
            encoder.append(nn.LayerNorm((enc_lin[i]), eps=1e-05, elementwise_affine=True))
            last = enc_lin[i]
        
        decoder = []
        # Decoder
        dec_lin.append(soih[-1][0]*soih[-1][1])
        for i in range(len(dec_lin)):
            decoder.append(nn.Linear(last, dec_lin[i]))
            decoder.append(nn.Dropout(p=0.5))
            decoder.append(nn.SELU())
            decoder.append(nn.BatchNorm1d(dec_lin[i],eps=1e-05))
            last = dec_lin[i]
            
        for i in range(len(soih)-1,-1,-1):
            decoder.append(nn.Unflatten(dim=1, unflattened_size = (soih[i][0], soih[i][1])))
            decoder.append(nn.MultiheadAttention(soih[i][1],soih[i][3],dropout=0.5, batch_first = True))
            decoder.append(nn.Dropout(p=0.5))
            if(i != 0):
                decoder.append(PatchwiseLinear(soih[i][0], soih[i][2],soih[i][1]))
                #decoder.append(nn.Linear(soih[i][1],soih[i][2]))
                decoder.append(nn.SELU())
                decoder.append(nn.BatchNorm1d(soih[i][0], eps=1e-05))
                decoder.append(nn.Flatten(start_dim=1))
        if self.loss == "NB":
            decoder.append(PatchwiseLinear(soih[0][0], 2*soih[0][2],soih[0][1]))
            decoder.append(nn.Flatten(start_dim=1))
            self.mean_act = lambda x: torch.clamp(torch.exp(x),1e-5, 1e6)
            self.disp_act = lambda x: torch.clamp(torch.nn.functional.softplus(x),1e-4, 1e4)
        elif self.loss == "ZINB":
            decoder.append(PatchwiseLinear(soih[0][0], 3*soih[0][2],soih[0][1]))
            decoder.append(nn.Flatten(start_dim=1))
            self.mean_act = lambda x: torch.clamp(torch.nn.functional.relu(x),1e-6, 1e6)
            self.disp_act = lambda x: torch.clamp(torch.nn.functional.softplus(x),1e-6, 1e6)
            self.dropout_act = lambda x: torch.nn.functional.logsigmoid(x)
        elif self.loss == "MSE":
            decoder.append(PatchwiseLinear(soih[0][0], soih[0][2],soih[0][1]))
            decoder.append(nn.Flatten())
        else:
            raise Exception("invalid loss type")
            
        self.encoder = nn.Sequential(*encoder)
        self.decoder = nn.Sequential(*decoder)
        
        def init_kaiming(model):
            for module in model:
                if type(module) == nn.Linear:
                    nn.init.kaiming_uniform_(module.weight,nonlinearity='selu')
                    nn.init.zeros_(module.bias)
                elif type(module) == PatchwiseLinear:
                    nn.init.kaiming_uniform_(module.weight, nonlinearity='selu')
                elif type(module) == Conv1dNormActivation:
                    nn.init.kaiming_uniform_(module.layers[0].weight, nonlinearity='selu')
        

        init_kaiming(self.encoder)
        init_kaiming(self.decoder)

    def encode(self,x):
        for l in self.encoder:
            if type(l) == nn.MultiheadAttention:
                x = l(x,x,x)[0] + x
            else:
                x = l(x)
        return x

    def decode(self,x):    
        for l in self.decoder:
            if type(l) == nn.MultiheadAttention:
                x = l(x,x,x)[0] + x
            else:
                x = l(x)        
        if self.loss == "ZINB":
            limits = x.shape[1]//3
            mean = self.mean_act(x[:,0:limits])
            disp = self.disp_act(x[:,limits:limits*2])
            pi = self.dropout_act(x[:,limits*2:])
            return torch.concat((mean.unsqueeze(2),disp.unsqueeze(2), pi.unsqueeze(2)),dim=2)        
        if self.loss == "NB":
            limits = x.shape[1]//2
            mean = self.mean_act(x[:,0:limits])
            disp = self.disp_act(x[:,limits:])
            return torch.concat((mean.unsqueeze(2),disp.unsqueeze(2)),dim=2)     
        return x

    def forward(self,x):
        x = self.patchify(x)
        x = self.encode(x)
        y = self.decode(x)
        return y
    
    def predict_decode(self, x):
        return self.decode(x)[:,:,0] # we just get the mean.

class PatchwiseLinear(nn.Module):
    def __init__(self,seq_len,out_dim,in_dim):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(seq_len,out_dim,in_dim))

    def forward(self,x):
        return torch.einsum('nom,bnm->bno',self.weight,x)
        
class Conv1dNormActivation(tv.ops.misc.ConvNormActivation):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple[int, int]] = 5,
        stride: Union[int, Tuple[int, int]] = 1,
        padding: Optional[Union[int, Tuple[int, int], str]] = None,
        groups: int = 1,
        norm_layer: Optional[Callable[..., torch.nn.Module]] = torch.nn.BatchNorm1d,
        activation_layer: Optional[Callable[..., torch.nn.Module]] = torch.nn.SELU,
        dilation: Union[int, Tuple[int, int]] = 1,
        inplace: Optional[bool] = True,
        bias: Optional[bool] = None,
    ) -> None:

        super().__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            groups,
            norm_layer,
            activation_layer,
            dilation,
            inplace,
            bias,
            torch.nn.Conv1d,
        )

