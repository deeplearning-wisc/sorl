## A Graph-Theoretic Framework for Understanding Open-World Representation Learning
Yiyou Sun, Zhenmei Shi, Yixuan (Sharon) Li


This repo contains the reference source code in PyTorch of the SORL framework. 
For more details please check our paper [A Graph-Theoretic Framework for Understanding Open-World Representation Learning](https://openreview.net/pdf?id=ZITOHWeAy7) (NeurIPS 23, **spotlight**). 

### Dependencies

The code is built with following libraries:

- [PyTorch==1.7.1](https://pytorch.org/)

### Usage

##### Get Started
- Download models pre-trained by unsupervised spectral contrastive loss [here](https://drive.google.com/drive/folders/1Xhk42VThcMOMfsSYMCzoUSVY9LRZ3TYf?usp=sharing) and put under the `pretrained` folder.

- To train and evaluate on CIFAR-100/10, run

```bash
./run.sh
```

### Citing

If you find our code useful, please consider citing:

```
@inproceedings{
    sun2023sorl,
    title={A Graph-Theoretic Framework for Understanding Open-World Representation Learning},
    author={Yiyou Sun and Zhenmei Shi and Yixuan Li},
    booktitle={NeurIPS},
    year={2023},
    url={https://openreview.net/forum?id=ZITOHWeAy7}
}
```