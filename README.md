# lidar_tranformer_self_training
The code behind this paper "Study LiDAR segmentation and  model's uncertainty using Transformer for different self-trainings"
Choose from:
TransUnet architecture.
TransUnet  architecture without the Transformerblock i.e.U-Net architecture where the output of the CNNencoder is passed directly to the decoder.
TransUnet architecture but with replacing the CNNbased  encoder  and  decoder  blocks  with  those  used  inSalsaNext architecture .
SalsaNext network implementation

For evaluation the following commands are used,
  TransUnet architecture but with replacing the CNNbased  encoder  and  decoder  blocks  with  those  used  inSalsaNext architecture,
python eval.py --vit_name R50-ViT-B_16 --batch_size <#images in batch> --evaluate_model --restore_from_dir <directory of saved the pre-trained weights> --restore_from <file_name>

  
  SalsaNext network implementation,
python eval.py --vit_name R50-ViT-B_16 --batch_size <#images in batch> --use_salsa --evaluate_model --restore_from <file_name>

   use batch normalisation layer learnt from pre-training,
python eval.py --vit_name R50-ViT-B_16 --batch_size <#images in batch> --bn_pretrain --evaluate_model --restore_from_dir <directory of saved the pre-trained weights> --restore_from <file_name>
  
  TransUnet architecture,
python eval.py --vit_name R50-ViT-B_16 --batch_size <#images in batch> --use_transunet_enc_dec --evaluate_model --restore_from_dir <directory of saved the pre-trained weights> --restore_from <file_name>

  TransUnet  architecture without the Transformerblock i.e.U-Net architecture where the output of the CNNencoder is passed directly to the decoder,
python eval.py --vit_name R50-ViT-B_16 --batch_size <#images in batch> --use_transunet_enc_dec --remove_Transformer --evaluate_model --restore_from_dir <directory of saved the pre-trained weights> --restore_from <file_name>
  
Remove --evaluate_model for epistemic uncertainty evaluation.
