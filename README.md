# swin-unet

## Topo loss:
from /metrics/topo_loss.py  
used in line 237, 239 in /trainer.py:
    
    if epoch_num > int(max_epoch * 0.9):
        loss_topo = getTopoLoss(outputs, label_batch)  
        loss = 0.4 * loss_ce + 0.6 * loss_dice if epoch_num <= int(max_epoch * 0.9) else 0.4 * loss_ce + 0.6 * loss_dice + 0.0005 * loss_topo  
        
## superpixel:
SLIC

## CCT:
from /networks/cct.py  
defined in line 709 in /networks/swin_transformer_unet_skip_expand_decoder_sys.py  
融合d1,d2,d3,d4

## MCCT:
defined in line 708 in /networks/swin_transformer_unet_skip_expand_decoder_sys.py  
融合只经过cnn的输入和d1,d2,d3

## External attention:
defined in line 718-720 in /networks/swin_transformer_unet_skip_expand_decoder_sys.py  

# optimization 
(from line 800 to line 842 in /networks/swin_transformer_unet_skip_expand_decoder_sys.py)
the original code was:  

    def forward(self, x):  
        x, x_downsample = self.forward_features(x)  
        x = self.forward_up_features(x,x_downsample)  
        x = self.up_x4(x)  
        
here are what I did:  

    def forward(self, x):
        CCT_module = False
        MCCT_module = False
        EA_channel = False
        superpixel = False

        if MCCT_module == True:
            d0=self.cnn4CCT(x)
            
        if superpixel == True:
            nimage = [mark_boundaries(i.permute(1,2,0),slic(i.permute(1,2,0), n_segments=100, sigma=10)) for i in x]
            x_supp = torch.tensor(nimage, dtype=torch.float).permute(0,3,1,2)
            
        x, [d1, d2, d3, d4] = self.forward_features(x)

        if CCT_module == True:
            d1, d2, d3, d4, w = self.cct(self.skipshape(d1), self.skipshape(d2), self.skipshape(d3), self.skipshape(d4))
            d1, d2, d3, d4 = self.rev_skipshape(d1),self.rev_skipshape(d2),self.rev_skipshape(d3),self.rev_skipshape(d4)
        if MCCT_module == True:
            d0, d1, d2, d3, w = self.mcct(d0, self.skipshape(d1), self.skipshape(d2), self.skipshape(d3))
            d1, d2, d3 = self.rev_skipshape(d1),self.rev_skipshape(d2),self.rev_skipshape(d3)
        if EA_channel == True:
            d1 = self.EA_channeld1(d1)
            d2 = self.EA_channeld2(d2)
            d3 = self.EA_channeld3(d3)

        x = self.forward_up_features(x, [d1, d2, d3, d4])
        x = self.up_x4(x)

        if superpixel == True:
            x = torch.cat([x, x_supp],1)
            x = self.cnn4supp(x)

        return x
我把swin transformer v2 也加到swin unet里了，但是效果好像很差啊。 不好意思大佬，上面英语部分是用研究室的工作站写的，没有中文输入法
