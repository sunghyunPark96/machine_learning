# Resnet Trainable Activation Function
ex) CosLU, ShiLU ...
논문 명 : Trainable Activations for Image Classification  
1) Resnet-20 & CosLU  
Max Accuracy : 0.899  
Papaer Accuracy : 0.904  
![TEST](https://github.com/user-attachments/assets/5b40b0cd-052d-4c3f-a6ff-ef46002fa2d0)

<div style="position: relative;">
  <pre>
    <code id="code-block">
python test.py --config configs/coslu/cifar10/resnet20.yaml
    </code>
  </pre>
  <button onclick="copyToClipboard('code-block')" style="position: absolute; top: 0; right: 0;">Copy</button>
</div>

python train.py --model ResNet-20 --activation CosLU
3) Resnet-8 & CosLU & Patch-embedding & Convmixer-Layer 모델 개선 코드는 improve에 구현하였습니다.  
<div style="position: relative;">
  <pre>
    <code id="code-block">
python test.py --config configs/coslu/cifar10/resnet8_convmixer.yaml
    </code>
  </pre>
  <button onclick="copyToClipboard('code-block')" style="position: absolute; top: 0; right: 0;">Copy</button>
</div>


참조 링크 : https://github.com/epishchik/TrainableActivation  
