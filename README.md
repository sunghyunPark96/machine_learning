# Resnet Trainable Activation Function
ex) CosLU, ShiLU ...
논문 명 : Trainable Activations for Image Classification  
## 1) Resnet-20 & CosLU  
Max Accuracy : 0.899  
Papaer Accuracy : 0.904  
![TEST](https://github.com/user-attachments/assets/5b40b0cd-052d-4c3f-a6ff-ef46002fa2d0)  

<div style="position: relative; background: #f9f9f9; border: 1px solid #ddd; border-radius: 4px; padding: 10px; font-family: monospace;">
<code id="code1">python test.py --config configs/coslu/cifar10/resnet20.yaml</code>
<button onclick="copyToClipboard('code1')" style="position: absolute; top: 5px; right: 10px; background: #007bff; color: white; border: none; border-radius: 4px; padding: 5px; cursor: pointer;">
</div>  

## 2) Resnet-8 & CosLU & Patch-embedding & Convmixer-Layer  
모델 개선 코드는 improve에 구현하였습니다.  
![convmixer](https://github.com/user-attachments/assets/91dcb285-94c3-4ed7-ab74-ca1919ee0a75)
![convmixer_Layer](https://github.com/user-attachments/assets/ea18f21b-dab7-47b1-bbc1-3054ff9aaceb)  
![patch_embedding](https://github.com/user-attachments/assets/973dbaf3-d1ac-4172-81d2-cedd4a8601cd)  
![resnet_class](https://github.com/user-attachments/assets/f1094aca-f9a1-4510-9b92-ba27cb1cb5f6)  
  
<div style="position: relative; background: #f9f9f9; border: 1px solid #ddd; border-radius: 4px; padding: 10px; font-family: monospace;">
<code id="code2">python test.py --config configs/coslu/cifar10/resnet8_convmixer.yaml</code>
<button onclick="copyToClipboard('code2')" style="position: absolute; top: 5px; right: 10px; background: #007bff; color: white; border: none; border-radius: 4px; padding: 5px; cursor: pointer;">
</div>  


참조 링크 : https://github.com/epishchik/TrainableActivation 
