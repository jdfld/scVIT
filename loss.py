from pytorch_forecasting.metrics.distributions import DistributionLoss
from pyro.distributions import ZeroInflatedNegativeBinomial
import torch

class ZINB():
    def map_x_to_distribution(self,x):
        mean = x[:,:,0]
        shape = x[:,:,1]
        pi = x[:,:,2]
        r = 1.0 / shape
        p = torch.log(mean) - torch.log(mean+r)
        return ZeroInflatedNegativeBinomial(total_count=r, logits=p, gate_logits=pi)


    def loss(self,y_pred, y_true):
        dist = self.map_x_to_distribution(y_pred)
        loss = -dist.log_prob(y_true)
        return loss
    