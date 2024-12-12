<div align="center">

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/trainable-activations-for-image/image-classification-on-mnist)](https://paperswithcode.com/sota/image-classification-on-mnist?p=trainable-activations-for-image)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/trainable-activations-for-image/image-classification-on-cifar-10)](https://paperswithcode.com/sota/image-classification-on-cifar-10?p=trainable-activations-for-image)

</div>


<h1 align="center">Trainable Activations for Image Classification</h1>

We propose a set of the trainable activation functions — Cosinu-Sigmoidal Linear Unit (CosLU),  DELU, Linear Combination (LinComb), Normalized Linear Combination (NormLinComb), Rectified Linear Unit N (ReLUN), Scaled Soft Sign (ScaledSoftSign), Shifted Rectified Linear Unit (ShiLU).

[Pretrained weights.](https://www.kaggle.com/datasets/pe4eniks/trainable-activations-checkpoints)

<h2 align="center">CosLU</h2>

$$CosLU(x) = (x + \alpha \cos(\beta x))\sigma(x)$$

$$\sigma(x) = \frac{1}{1 + e^{-x}}$$

<p align="center">
<img src="plot/functions/coslu.png" alt="CosLU" width="600"/>
</p>

<h2 align="center">DELU</h2>

$$
DELU(x) = \begin{cases} 
SiLU(x), x \leqslant 0 \\ 
(n + 0.5)x + |e^{-x} - 1|, x > 0 
\end{cases}
$$

$$SiLU(x) = x\sigma(x)$$

<p align="center">
<img src="plot/functions/delu.png" alt="DELU" width="600"/>
</p>

<h2 align="center">LinComb</h2>

$$LinComb(x) = \sum\limits_{i=0}^{n} w_i \mathcal{F}_i(x)$$

<p align="center">
<img src="plot/functions/lincomb.png" alt="LinComb" width="600"/>
</p>

<h2 align="center">NormLinComb</h2>

$$NormLinComb(x) = \frac{\sum\limits_{i=0}^{n} w_i \mathcal{F}_i(x)}{\mid \mid W \mid \mid}$$

<p align="center">
<img src="plot/functions/normlincomb.png" alt="NormLinComb" width="600"/>
</p>

<h2 align="center">ReLUN</h2>

$$ReLUN(x) = min(max(0, x), n)$$

<p align="center">
<img src="plot/functions/relun.png" alt="ReLUN" width="600"/>
</p>

<h2 align="center">ScaledSoftSign</h2>

$$ScaledSoftSign(x) = \frac{\alpha x}{\beta + |x|}$$

<p align="center">
<img src="plot/functions/scaledsoftsign.png" alt="ScaledSoftSign" width="600"/>
</p>

<h2 align="center">ShiLU</h2>

$$ShiLU(x) = \alpha ReLU(x) + \beta$$

$$ReLU(x) = max(0, x)$$

<p align="center">
<img src="plot/functions/shilu.png" alt="ShiLU" width="600"/>
</p>

<h2 align="center">INSTALLATION</h2>

Create venv.
```bash
python3 -m venv venv
```

Activate venv.
```bash
source venv/bin/activate
```

Install dependencies.
```bash
pip install -r requirements.txt
```

<h2 align="center">PROJECT STRUCTURE</h2>

There are 3 main files - [train.py](train.py), [test.py](test.py), [plot.py](plot.py). You should run train.py first, then test.py, then plot.py.

Use whatever configuration you want to test. Configurations can be found in the [configs](configs/) folder, train.py and test.py use the same config. There are several plot configurations in the [configs/plot](configs/plot) folder.

There are many predefined run scripts in the [scripts](scripts/) folder, just run one of them as .sh, [scripts/train.sh](scripts/train.sh) and [scripts/test.sh](scripts/test.sh) are scripts to train and test all possible configurations, [scripts/plot.sh](scripts/plot.sh) to plot results after training and testing.

All the results of the train / test phases are in the [logs](logs/) folder.

All proposed trainable activations are in [activation.py](activation.py).

<h2 align="center">HOW TO RUN</h2>

Let's say I want to train and test the ResNet-8 model with CosLU trainable activation on the CIFAR-10 dataset.
```python
python train.py --config configs/coslu/cifar10/resnet8.yaml
python test.py --config configs/coslu/cifar10/resnet8.yaml
```

If you want to train and test all proposed trainable activations with a specific model and dataset, you can use the script from the [scripts](scripts/) folder. For example, train and test the DNN2 model on the MNIST dataset.
```bash
sh scripts/dnn2_mnist.sh
```

Train and test all possible configurations.
```bash
sh scripts/train.sh
sh scripts/test.sh
```

Plot graphics for all configurations, it will work even if some configurations haven't been trained.
```bash
sh scripts/plot.sh
```

<h2 align="center">CITATION</h2>

<p align="center">
Project <a href="CITATION.cff" title="CITATION">CITATION</a>.
</p>

<h2 align="center">LICENSE</h2>

<p align="center">
Project is distributed under <a href="LICENSE" title="LICENSE">MIT License</a>.
</p>
