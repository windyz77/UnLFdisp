# UnLFdisp

Wenhui Zhou, Lili Lin, Yongjie Hong, Qiujian Li, Xingfa Shen and Ercan Engin Kuruoglu, **Beyond Photometric Consistency: Geometry-based Occlusion-aware Unsupervised Light Field Disparity Estimation**. IEEE Transactions on Neural Networks and Learning Systems, 2023. [MP4](https://faculty.hdu.edu.cn/_upload/article/videos/13/d3/7edf42a345ddab5a383b47a4703f/ca0e6827-5f04-4bee-87c8-153b2957dbbc.mp4)，https://ieeexplore.ieee.org/document/10172242 

## Requirements

```
pip install -r requirements.txt
```



## dataset

We used the HCI 4D LF benchmark for training and evaluation.Please refer to the benchmark website for details.

## train

In monodepth_main_zhoubo_mask_v1.py, set train_or_test=0

```
python monodepth_main_zhoubo_mask_v1
```

## test

In monodepth_main_zhoubo_mask_v1.py, set train_or_test=1

```
python monodepth_main_zhoubo_mask_v1
```

