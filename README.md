# lidar_tranformer_self_training
![alt text](https://github.com/MoHassoubah/lidar_tranformer_self_training/blob/main/arch.png)
The code behind this paper "Study LiDAR segmentation and  model's uncertainty using Transformer for different self-trainings"
# Qualitative results
[embed]https://github.com/MoHassoubah/lidar_tranformer_self_training/blob/main/qualitative_results.pdf[/embed]
Showing the 2D segmentation output for differentpre-training configurations, architectures and segmentation loss functions.

![alt text](https://github.com/MoHassoubah/lidar_tranformer_self_training/blob/main/with_without transformer.pdf)
The red circles shows where in the image the TransUnet outperformed the U-Net architecture.For example the U-Net miss classifies between the terrain class and sidewalk classes many times.  TansUnetclassifies the Person class and the trunck class comparably better than the U-Net.

![alt text](https://github.com/MoHassoubah/lidar_tranformer_self_training/blob/main/salsa_vs_our_best.pdf)
The red circles shows where in the image the proposed architecture (TransUnet replacing ENC &DEC with those in [2]plusbeing self-supervised pre-trained using Xavier-I & Rec-ST & BN-R-Iplususingsegmentation loss Crossentropy + Lovasz-Softmax) outperformed the SalsaNext architecture.  For examplethe SalsaNext many times miss classifies unlabeled points, this is because authors in their implementationtrained the network while ignoring the unlabeled class (this may helped in augmenting the unlabeled class butat the same time has increased the epistemic uncertainty of the model’s output).  The proposed architecturebetter classifies the sidewalk and the trunk classes.
## Study applied on architectures:
* TransUnet architecture.
* TransUnet  architecture without the Transformerblock i.e.U-Net architecture where the output of the CNNencoder is passed directly to the decoder.
* TransUnet architecture but with replacing the CNNbased  encoder  and  decoder  blocks  with  those  used  inSalsaNext architecture .
* SalsaNext network implementation
## For Pre-training the following commands are used,
### Pre-Training TransUnet architecture using the reconstruction + contrastive loss,
  
```$ python train.py --vit_name R50-ViT-B_16 --batch_size <#images in batch> --root_path <data path> --pretrain --contrastive --use_transunet_enc_dec```

### Pre-Training TransUnet architecture using the reconstruction loss only,
  
```$ python train.py --vit_name R50-ViT-B_16 --batch_size <#images in batch> --root_path <data path> --pretrain --use_transunet_enc_dec```

### Pre-Training TransUnet architecture using the reconstruction loss only, but with replacing the CNNbased  encoder  and  decoder  blocks  with  those  used  inSalsaNext architecture (this architecture should not be trained using contrastive loss),
  
```$ python train.py --vit_name R50-ViT-B_16 --batch_size <#images in batch> --root_path <data path> --pretrain```


## For Segmentation training the following commands are used,
### TransUnet architecture but with replacing the CNNbased  encoder  and  decoder  blocks  with  those  used  inSalsaNext architecture,
  
```$ python train.py --vit_name R50-ViT-B_16 --batch_size <#images in batch> --root_path <data path> --restore_from_dir <directory of saved the pre-trained weights> --restore_from <pre-trained file_name>```

   ### Use batch normalisation layer learnt from pre-training,
  
```$ python train.py --vit_name R50-ViT-B_16 --batch_size <#images in batch> --bn_pretrain --root_path <data path> --restore_from_dir <directory of saved the pre-trained weights> --restore_from <pre-trained file_name>```
  
  ### SalsaNext network implementation,
  
```$ python train.py --vit_name R50-ViT-B_16 --batch_size <#images in batch> --use_salsa --train_fr_scratch --root_path <data path> --restore_from_dir <directory of saved the trained weights> --restore_from <file_name>```

  ### TransUnet architecture,
  
```$ python eval.py --vit_name R50-ViT-B_16 --batch_size <#images in batch> --use_transunet_enc_dec --root_path <data path> --restore_from_dir <directory of saved the pre-trained weights> --restore_from <pre-trained file_name>```

  ### TransUnet  architecture without the Transformerblock i.e.U-Net architecture where the output of the CNNencoder is passed directly to the decoder,
  
```$ python train.py --vit_name R50-ViT-B_16 --batch_size <#images in batch> --use_transunet_enc_dec --remove_Transformer --root_path <data path> --restore_from_dir <directory of saved the pre-trained weights> --restore_from <pre-trained file_name>```

## For evaluation the following commands are used,

  ### TransUnet architecture but with replacing the CNNbased  encoder  and  decoder  blocks  with  those  used  inSalsaNext architecture, ([pre-trained segmentation models](https://drive.google.com/drive/folders/1BPplPzaWfqqoqv0iYHMsFM4-_ozUFvJ7?usp=sharing))
  
```$ python eval.py --vit_name R50-ViT-B_16 --batch_size <#images in batch> --evaluate_model --root_path <data path> --restore_from_dir <directory of saved the trained weights> --restore_from <file_name>```

  
  ### SalsaNext network implementation,([pre-trained segmentation models](https://drive.google.com/drive/folders/18RKSfkXwWsSnQUCfUuTCsuda0gaQc8Zd?usp=sharing))
  
```$ python eval.py --vit_name R50-ViT-B_16 --batch_size <#images in batch> --use_salsa --evaluate_model --root_path <data path> --restore_from <file_name>```

   ### Use batch normalisation layer learnt from pre-training, ([pre-trained segmentation models](https://drive.google.com/drive/folders/1zDy-_rB4z0eCYykZhoT_6SyWThgArYO_?usp=sharing))
  
```$ python eval.py --vit_name R50-ViT-B_16 --batch_size <#images in batch> --bn_pretrain --evaluate_model --root_path <data path> --restore_from_dir <directory of saved the trained weights> --restore_from <file_name>```
  
  ### TransUnet architecture, ([pre-trained segmentation models](https://drive.google.com/drive/folders/1pXGka1-E6m9XsOqjIGQwYrS9Hm7feaHS?usp=sharing))
  
```$ python eval.py --vit_name R50-ViT-B_16 --batch_size <#images in batch> --use_transunet_enc_dec --evaluate_model --root_path <data path> --restore_from_dir <directory of saved the trained weights> --restore_from <file_name>```

  ### TransUnet  architecture without the Transformerblock i.e.U-Net architecture where the output of the CNNencoder is passed directly to the decoder, ([pre-trained segmentation models](https://drive.google.com/drive/folders/1bpxcYur43-gJjRe-Jas8QtSGoZ1dRPlJ?usp=sharing))
  
```$ python eval.py --vit_name R50-ViT-B_16 --batch_size <#images in batch> --use_transunet_enc_dec --root_path <data path> --remove_Transformer --evaluate_model --restore_from_dir <directory of saved the trained weights> --restore_from <file_name>```
  
Remove ```$ --evaluate_model``` for epistemic uncertainty evaluation.
  
### Disclamer

We based our code on [TranUnet](https://github.com/Beckschen/TransUNet), [RangeNet++](https://github.com/PRBonn/lidar-bonnetal), [SalsaNext](https://github.com/Halmstad-University/SalsaNext) and  [deep_uncertainty_estimation](https://github.com/uzh-rpg/deep_uncertainty_estimation) please go show some support!
