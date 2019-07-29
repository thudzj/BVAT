# BVAT (ICML2019 workshop paper)
Code for ["Batch Virtual Adversarial Training for Graph Convolutional Networks"](https://graphreason.github.io/papers/3.pdf) which is based on the original implementation of [GCN](https://github.com/tkipf/gcn).

# Requirements
- tensorflow (>0.12)
- networkx

# Run the codes
We provide two adversarial training algorithms (SBVAT and OBVAT). Please refer to our paper for the details. Typically, you can run the algorithms by:
```
cd obvat
python train.py
```

# Cite
Please cite our paper if you use this code in your own work:
```
@article{deng2019batch,
  title={Batch Virtual Adversarial Training for Graph Convolutional Networks},
  author={Deng, Zhijie and Dong, Yinpeng and Zhu, Jun},
  journal={arXiv preprint arXiv:1902.09192},
  year={2019}
}
```
