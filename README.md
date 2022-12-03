# AutoInfo GAN: Towards a better image synthesis framework of GAN on high-fidelity few-shot datasets via NAS and contrastive learning - pytorch

## 0. Data
The datasets used in the paper can be found at [link](https://drive.google.com/file/d/1aAJCZbXNHyraJ6Mi13dSbe7pTyfPXha0/view?usp=sharing). 

After testing on over 20 datasets with each has less than 100 images, this GAN converges on 80% of them.
I still cannot summarize an obvious pattern of the "good properties" for a dataset which this GAN can converge on, please feel free to try with your own datasets.

please put these datasets to 'data' directory created by yourself.

## 1. Description
The code is structured as follows:
* models_search: the definition of shared GAN and controller used in searh.

* operation.py: the helper functions and data loading methods during training.

* search_mixed_2stage.py: hybrid 2-stage method.

* train_search.py: training derived GAN from the scratch.

* benchmarking: the functions we used to compute FID are located here, it automatically downloads the pytorch official inception model. 

* lpips: this folder contains the code to compute the LPIPS score, the inception model is also automatically download from official location.

* scripts: this folder contains many scripts you can use to play around the trained model. Including: 
    1. style_mix.py: style-mixing as introduced in the paper;
    2. generate_video.py: generating a continuous video from the interpolation of generated images;
    3. find_nearest_neighbor.py: given a generated image, find the closest real-image from the training set;
    4. train_backtracking_one.py: given a real-image, find the latent vector of this image from a trained Generator.

## 2. How to run
seach GAN, please call
```
sh ./exps/autogan_search_2stage.sh
```
train derived GAN, please call:
```
sh ./exps/derive.sh
```

Project will automatically generate a file directiory 'log', you can find models and logs in it.

## 3. Acknowledgement
Our project thanks to the FastGAN [link] (https://github.com/odegeasslbc/FastGAN-pytorch) and AutoGAN [link] (https://github.com/TAMU-VITA/AutoGAN)
