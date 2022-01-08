# swin-unet

## Topo loss:
from /metrics/topo_loss.py  
used in line 237, 239 in /trainer.py:
    
    if epoch_num > int(max_epoch * 0.9):
        loss_topo = getTopoLoss(outputs, label_batch)  
        loss = 0.4 * loss_ce + 0.6 * loss_dice if epoch_num <= int(max_epoch * 0.9) else 0.4 * loss_ce + 0.6 * loss_dice + 0.0005 * loss_topo  
        
## superpixel:
defined in line 18-99, in /networks/swin_transformer_unet_skip_expand_decoder_sys.py  

## CCT:
from /networks/cct.py  
defined in line 790 in /networks/swin_transformer_unet_skip_expand_decoder_sys.py  
## MCCT:
defined in line 789 in /networks/swin_transformer_unet_skip_expand_decoder_sys.py  

## External attention:
defined in line 797-799 in /networks/swin_transformer_unet_skip_expand_decoder_sys.py  

# optimization (from line 882 to line 909 in /networks/swin_transformer_unet_skip_expand_decoder_sys.py)
the original code was:  

    def forward(self, x):  
        x, x_downsample = self.forward_features(x)  
        x = self.forward_up_features(x,x_downsample)  
        x = self.up_x4(x)  
        
here are what I did:  

    def forward(self, x):  
        CCT_modual = False  
        MCCT_modual = False  
        EA_channel = False  
        superpixel = False  
        d0=self.cnnt1(x)  
        if superpixel == True:  
            x_supp = self.supp(x)  

        x, [d1, d2, d3, d4] = self.forward_features(x)  

        if CCT_modual == True:  
            d1, d2, d3, d4, w = self.cct(self.skipshape(d1), self.skipshape(d2), self.skipshape(d3), self.skipshape(d4))  
            d1, d2, d3, d4 = self.rev_skipshape(d1),self.rev_skipshape(d2),self.rev_skipshape(d3),self.rev_skipshape(d4)  
        if MCCT_modual == True:  
            d0, d1, d2, d3, w = self.mcct(d0, self.skipshape(d1), self.skipshape(d2), self.skipshape(d3))  
            d1, d2, d3 = self.rev_skipshape(d1),self.rev_skipshape(d2),self.rev_skipshape(d3)  
        if EA_channel == True:  
            d1 = self.EA_channeld1(d1)  
            d2 = self.EA_channeld2(d2)  
            d3 = self.EA_channeld3(d3)  

        x = self.forward_up_features(x, [d1, d2, d3, d4])  
        x = self.up_x4(x)  

        if superpixel == True:  
            x = (x + x_supp)/2  

        return x  
