# Language-driven Semantic Segmentation (LSeg)
This branch contains Habana PyTorch implementation of paper [Language-driven Semantic Segmentation](https://arxiv.org/abs/2201.03546).

## Overview
LSeg uses a text encoder to compute embeddings of descriptive input labels (e.g., ''grass'' or 'building'') together with a transformer-based image encoder that computes dense per-pixel embeddings of the input image. The image encoder is trained with a contrastive objective to align pixel embeddings to the text embedding of the corresponding semantic class. The text embeddings provide a flexible label representation in which semantically similar labels map to similar regions in the embedding space (e.g., ''cat'' and ''furry''). This allows LSeg to generalize to previously unseen categories at test time, without retraining or even requiring a single additional training sample. We demonstrate that our approach achieves highly competitive zero-shot performance compared to existing zero- and few-shot semantic segmentation methods, and even matches the accuracy of traditional segmentation algorithms when a fixed label set is provided.

In the branch [habana_dev](https://github.com/intel-sandbox/lang-seg-Migration/tree/habana_dev), you can implement an interactable web demo which uses [Habana Gaudi](https://habana.ai/training/gaudi/) as the processor to demonstrate the results of LSeg.


## Usage
### Installation
#### Set Up AWS EC2 DL1 Instance
You can follow the [AWS DL1 Quick Start Guide](https://docs.habana.ai/en/latest/AWS_EC2_DL1_and_PyTorch_Quick_Start/AWS_EC2_DL1_and_PyTorch_Quick_Start.html) to launch an EC2 with Habana Deep Learning Base AMI.

#### Use Docker to Build Environment
We provide a ready-to-use docker image [lseg_habana.tar.gz](https://intel-my.sharepoint.com/:u:/p/yifan_wang/EUBP08GicbNNtXdagdpx4z8B4NehBFKzHT0i1H8sLZ3oRQ?e=EaG8Bc) for you to load. After downloading the image, run the following command to load the image and run a container:
 ```
 docker load < lseg_habana.tar.gz

 docker run -it \
        --runtime=habana \
        -e HABANA_VISIBLE_DEVICES=all \
        -e OMPI_MCA_btl_vader_single_copy_mechanism=none \
        --cap-add=sys_nice --net=host --ipc=host \
        -p 8501:8501 \
        --name lseg_habana \
        torch_hpu

 docker exec -it lseg_habana bash

 cd Projects/lseg
 ``` 



### Data Preparation
By default, for training, testing and demo, we use [ADE20k](https://groups.csail.mit.edu/vision/datasets/ADE20K/).

```
python prepare_ade20k.py
unzip ../datasets/ADEChallengeData2016.zip
```

Note: for demo, if you want to use random inputs, you can ignore data loading and comment the code at [link](https://github.com/isl-org/lang-seg/blob/main/modules/lseg_module.py#L55). 

### Try Web Demo
<table>
  <thead>
    <tr style="text-align: right;">
      <th>name</th>
      <th>backbone</th>
      <th>text encoder</th>
      <th>url</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>Model for demo</td>
      <td>ViT-L/16</td>
      <td>CLIP ViT-B/32</td>
      <td><a href="https://intel-my.sharepoint.com/:u:/p/yifan_wang/EYdy99kH55RBrBaRHXzB7boBGtBexoF2M7nOvnmA1FHVmA?e=gdjV6o">download</a></td>
    </tr>
  </tbody>
</table>

Download the model for demo and put it under folder `checkpoints` as `checkpoints/demo_e200_fp32.ckpt`.

Then split the terminal into two, one runs ```python lseg_web_backend.py```, the other runs ```streamlit run lseg_web_frontend.py``` (Tmux suggested). Then you are able to use the network url to try out the web demo!

### Configuration
You can adjust some common configurations in ```.app/lseg_habana_app.config.json```. Here are some examples:
```json
{
  "#hpu_mode": "lazy mode=1, eager mode=2",
  "#default_lazy_mode_label_number": "default=20",
  "#static_image_size_params": "[hw_ratios], [short_sizes]",
  ...
}
```


## Acknowledgement
Thanks to the code base from [DPT](https://github.com/isl-org/DPT), [Pytorch_lightning](https://github.com/PyTorchLightning/pytorch-lightning), [CLIP](https://github.com/openai/CLIP), [Pytorch Encoding](https://github.com/zhanghang1989/PyTorch-Encoding), [Streamlit](https://streamlit.io/), [Wandb](https://wandb.ai/site)
