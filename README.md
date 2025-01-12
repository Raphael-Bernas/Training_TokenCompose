# Paper
To see our paper we made for this project, refer to FPR_D_Bernas_Corlay.pdf

# Training_TokenCompose
A notebook to test on colab the code from "TOKENCOMPOSE: Text-to-Image Diffusion with Token-level Supervision" by Zirui Wang , Zhizhou Sha , Zheng Ding Yilin Wang and Zhuowen Tu (https://github.com/mlpc-ucsd/TokenCompose/)

# Training SD1.4
## With Token and Pixel losses :
Trainin_TokencCompose.ipynb
## With Pixel Loss :
Training_TokenCompose_without_token_loss.ipynb
## With Token Loss :
Training_TokenCompose_without_pixel_loss.ipynb
## With Token, Pixel and Spatial losses :
Training_TokenCompose_with_spatial_loss.ipynb

# Spatial Loss
In loss_utils_spatial.py and train_token_compose_spatial.py, the original code from TokenCompose has been partially modified to add a spatial loss.

After adding them in the original TokenCompose/train/src folder, you can compute from the TokenCompose/train/ folder train_spatial.sh. [All this is already done in the Training_TokenCompose_with_spatial_loss.ipynb notebook]
