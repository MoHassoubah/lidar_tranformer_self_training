import torch
from torch import nn
# import logging

eps = 1e-7

class NCECriterion(nn.Module):

    def __init__(self, nLem):
        super(NCECriterion, self).__init__()
        self.nLem =  nLem#number of data samples

    # def forward(self, x, targets):
        # batchSize = x.size(0)
        # K = x.size(1)-1 #number of -ve samples
        
        # nominator = x.select(1,0)
        # loss_partial = -torch.log(nominator / torch.sum(x, dim=1))
        # loss = torch.sum(loss_partial) / batchSize
        
        # return loss
        
    def forward(self, x, targets):
        batchSize = x.size(0)
        K = x.size(1)-1 #number of -ve samples
        Pnt = 1 / float(self.nLem) # 1/n->Pn(i) noise distribution as a uniform distribution
        Pns = 1 / float(self.nLem)
        
        # eq 5.1 : P(origin=model) = Pmt / (Pmt + k*Pnt) 
        # x size is (batchSize , K+1)
        Pmt = x.select(1,0) #P(i|v) of all images in the batch->+ve sample
        
        # logging.info('P(i|v)->#mean in batch : %f' % (Pmt.mean().item()))
        Pmt_div = Pmt.add(K * Pnt + eps) #P(i|v) + mPn(i) of all images in the batch
        lnPmt = torch.div(Pmt, Pmt_div)
        
        # eq 5.2 : P(origin=noise) = k*Pns / (Pms + k*Pns)
        Pon_div = x.narrow(1,1,K).add(K * Pns + eps)#x.narrow(1,1,K) eq x[:,1:1+K] -> to choose P(i|v) that belongs to the product of fi and v(-ve samples)
        Pon = Pon_div.clone().fill_(K * Pns)# this outputs vectos size like Pon_div has values K * Pns (m/n in the paper)
        lnPon = torch.div(Pon, Pon_div) #torch.div Divides each element of the input input by the corresponding element of other
     
        # equation 6 in ref. A
        lnPmt.log_()
        lnPon.log_()
        
        lnPmtsum = lnPmt.sum(0)
        lnPonsum = lnPon.view(-1, 1).sum(0) #(-1) generate additional dimension
        
        loss = - (lnPmtsum + lnPonsum) / batchSize#removed the m multiplication as in the paper as there was a sum of K dim vector before
        
        return loss

