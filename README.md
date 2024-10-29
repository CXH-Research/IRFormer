### 📋 [Implicit Multi-Spectral Transformer: An Lightweight and Effective Visible to Infrared Image Translation Model](https://arxiv.org/abs/2404.07072)

<div>
<span class="author-block">
  Yijia Chen
</span>,
  <span class="author-block">
    Pinghua Chen<sup>📮</sup>
  </span>,
  <span class="author-block">
    Xiangxin Zhou
  </span>,
  <span class="author-block">
    Yingtie Lei
  </span>,
  <span class="author-block">
    Ziyang Zhou
  </span>,
  <span class="author-block">
  Mingxian Li
</span>
  (📮 Corresponding Author)
  </div>

<b>Guangdong University of Technology, University of Macau, Huizhou University</b>

In <b>_International Joint Conference on Neural Networks 2024 (IJCNN 2024)_<b>

## ⚙️ Usage
### Installation
```
git clone https://github.com/CXH-Research/IRFormer.git
cd IRFormer
pip install -r requirements.txt
```

### Training
Please first specify TRAIN_DIR, VAL_DIR and SAVE_DIR in section TRAINING in config.yml

For single GPU training:
```
python train.py
```
For multiple GPUs training:
```
accelerate config
accelerate launch train.py
```
If you have difficulties on the usage of accelerate, please refer to <a href="https://github.com/huggingface/accelerate">Accelerate</a>.

### Inference
Please first specify TRAIN_DIR, VAL_DIR and SAVE_DIR in section TESTING in traning.yml
```
python test.py
```

# 💗 Acknowledgements
This work was supported in part by the Guangdong Provincial Key R&D Programme under Grant No.2023B1111050010 and No.2020B0101100001, in part by the Huizhou Daya Bay Science and Technology Planning Project under Grant No.2020020003.

### 🛎 Citation
If you find our work helpful for your research, please cite:
```bib
@inproceedings{DBLP:conf/ijcnn/ChenCZLZL24,
  author       = {Yijia Chen and
                  Pinghua Chen and
                  Xiangxin Zhou and
                  Yingtie Lei and
                  Ziyang Zhou and
                  Mingxian Li},
  title        = {Implicit Multi-Spectral Transformer: An Lightweight and Effective
                  Visible to Infrared Image Translation Model},
  booktitle    = {International Joint Conference on Neural Networks},
  pages        = {1--8},
  year         = {2024}
}
```
