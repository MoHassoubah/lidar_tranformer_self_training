# lidar_tranformer_self_training
![alt text](https://github.com/MoHassoubah/lidar_tranformer_self_training/blob/main/arch.png)
The code behind this paper "Study LiDAR segmentation and  model's uncertainty using Transformer for different self-trainings"
## Study applied on architectures:
* TransUnet architecture.
* TransUnet  architecture without the Transformerblock i.e.U-Net architecture where the output of the CNNencoder is passed directly to the decoder.
* TransUnet architecture but with replacing the CNNbased  encoder  and  decoder  blocks  with  those  used  inSalsaNext architecture .
* SalsaNext network implementation
## For Pre-training the following commands are used,
### Pre-Training TransUnet architecture using the reconstruction + contrastive loss,
  
```$ python train.py --vit_name R50-ViT-B_16 --batch_size <#images in batch> --pretrain --contrastive --use_transunet_enc_dec```

### Pre-Training TransUnet architecture using the reconstruction loss only,
  
```$ python train.py --vit_name R50-ViT-B_16 --batch_size <#images in batch> --pretrain --use_transunet_enc_dec```

### Pre-Training TransUnet architecture using the reconstruction loss only, but with replacing the CNNbased  encoder  and  decoder  blocks  with  those  used  inSalsaNext architecture (this architecture should not be trained using contrastive loss),
  
```$ python train.py --vit_name R50-ViT-B_16 --batch_size <#images in batch> --pretrain```


## For Segmentation training the following commands are used,
### TransUnet architecture but with replacing the CNNbased  encoder  and  decoder  blocks  with  those  used  inSalsaNext architecture,
  
```$ python train.py --vit_name R50-ViT-B_16 --batch_size <#images in batch> --restore_from_dir <directory of saved the pre-trained weights> --restore_from <pre-trained file_name>```

   ### Use batch normalisation layer learnt from pre-training,
  
```$ python train.py --vit_name R50-ViT-B_16 --batch_size <#images in batch> --bn_pretrain --restore_from_dir <directory of saved the pre-trained weights> --restore_from <pre-trained file_name>```
  
  ### SalsaNext network implementation,
  
```$ python train.py --vit_name R50-ViT-B_16 --batch_size <#images in batch> --use_salsa --train_fr_scratch --restore_from <pre-trained file_name>```

  ### TransUnet architecture,
  
```$ python eval.py --vit_name R50-ViT-B_16 --batch_size <#images in batch> --use_transunet_enc_dec --restore_from_dir <directory of saved the pre-trained weights> --restore_from <pre-trained file_name>```

  ### TransUnet  architecture without the Transformerblock i.e.U-Net architecture where the output of the CNNencoder is passed directly to the decoder,
  
```$ python train.py --vit_name R50-ViT-B_16 --batch_size <#images in batch> --use_transunet_enc_dec --remove_Transformer --restore_from_dir <directory of saved the pre-trained weights> --restore_from <pre-trained file_name>```

## For evaluation the following commands are used,

  ### TransUnet architecture but with replacing the CNNbased  encoder  and  decoder  blocks  with  those  used  inSalsaNext architecture,
  
```$ python eval.py --vit_name R50-ViT-B_16 --batch_size <#images in batch> --evaluate_model --restore_from_dir <directory of saved the trained weights> --restore_from <file_name>```

  
  ### SalsaNext network implementation,
  
```$ python eval.py --vit_name R50-ViT-B_16 --batch_size <#images in batch> --use_salsa --evaluate_model --restore_from <file_name>```

   ### Use batch normalisation layer learnt from pre-training,
  
```$ python eval.py --vit_name R50-ViT-B_16 --batch_size <#images in batch> --bn_pretrain --evaluate_model --restore_from_dir <directory of saved the trained weights> --restore_from <file_name>```
  
  ### TransUnet architecture,
  
```$ python eval.py --vit_name R50-ViT-B_16 --batch_size <#images in batch> --use_transunet_enc_dec --evaluate_model --restore_from_dir <directory of saved the trained weights> --restore_from <file_name>```

  ### TransUnet  architecture without the Transformerblock i.e.U-Net architecture where the output of the CNNencoder is passed directly to the decoder,
  
```$ python eval.py --vit_name R50-ViT-B_16 --batch_size <#images in batch> --use_transunet_enc_dec --remove_Transformer --evaluate_model --restore_from_dir <directory of saved the trained weights> --restore_from <file_name>```
  
Remove ```$ --evaluate_model``` for epistemic uncertainty evaluation.
  
### Disclamer

We based our code on [TranUnet](https://github.com/Beckschen/TransUNet), [RangeNet++](https://github.com/PRBonn/lidar-bonnetal), [SalsaNext](https://github.com/Halmstad-University/SalsaNext) and  [deep_uncertainty_estimation](https://github.com/uzh-rpg/deep_uncertainty_estimation) please go show some support!
