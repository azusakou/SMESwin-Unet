# Train
## before training
1 cd superpixel/../cython   
2 setup.py: python setup.py build_ext --inplace  
3 install packages
## training 
command:  
1 python train.py --dataset GlaS --cfg configs/swin_tiny_patch4_window7_224_lite.yaml --root_path ./data/GlaS --max_epochs 1 --output_dir ./output  --img_size 224 --base_lr 0.05 --batch_size 18  
2 tensorboard --logdir ../output (there is something wrong in linux 20+) 
3 nvidia-smi -l 1  
4 kill -9 
## Ablation study 
python test.py --dataset GlaS --cfg configs/swin_tiny_patch4_window7_224_lite.yaml --is_saveni --volume_path ./data/GlaS --output_dir ./output --max_epoch 150 --base_lr 0.05 --img_size 224 --batch_size 24  

### reminder
x input shape: torch.Size([1, 3, 224, 224])  
d1,d2,d3,d4 shape: torch.Size([1, 3136, 96]) torch.Size([1, 784, 192]) torch.Size([1, 196, 384]) torch.Size([1, 49, 768])  
x upsampling shape: torch.Size([1, 3136, 96])  
x output shape: torch.Size([1, 2, 224, 224])  

[ref](https://github.com/azusakou/External-Attention-pytorch#mlp-series)

### Baseline: swin unet V1 
mean_dice 0.901184 mean_hd95 49.901207 mean_IoU 0.827942  

### ExternalAttention (d_model = channeld1, S=8)  
mean_dice 0.908098 mean_hd95 43.893057 mean_IoU 0.838083 (d1)  
mean_dice 0.904035 mean_hd95 47.023975 mean_IoU 0.831332 (d2)  
mean_dice 0.897228 mean_hd95 49.850001 mean_IoU 0.820484 (d3)  
mean_dice 0.903174 mean_hd95 46.106423 mean_IoU 0.829137 (d1+d3)  
mean_dice 0.867021 mean_hd95 58.527846 mean_IoU 0.776660 (d1+d2)  
mean_dice 0.902326 mean_hd95 48.685495 mean_IoU 0.828485 (d2+d3)  
mean_dice 0.910144 mean_hd95 47.026681 mean_IoU 0.842595 (d1+d2+d3, 1000 epoch)

### CCT
mean_dice 0.913837 mean_hd95 46.878118 mean_IoU 0.848151 mean_dice 0.913837  

### MCCT  
mean_dice 0.910646 mean_hd95 46.688599 mean_IoU 0.844525 (input: d0,d1,d2,d3; output: d1,d2,d3)  

### EAMCCT  
mean_dice 0.914312 mean_hd95 45.035457 mean_IoU 0.848958 (input: d0,d1,d2,d3; ExternalAttion(d1); output: d1,d2,d3)  
mean_dice 0.916212 mean_hd95 41.295990 mean_IoU 0.851688 (input: d0,d1,d2,d3; ExternalAttion(d1, d2, d3); output: d1,d2,d3)