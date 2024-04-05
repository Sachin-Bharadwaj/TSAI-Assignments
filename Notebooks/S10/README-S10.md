# TSAI-S9

This assignment is about training a Conv architecture with 4 conceptual blocks with architecture having pyramidal structure. Each block having different number of layers such that last layer of each block except the last Block has stride2. We have used 4 layers with num_layer per block = [4,3,3,2]. The 2nd, 3rd and 4th Block has depthwise separable convolutions to reduce the number of parameters in the architecture. The 2 Block uses dilated convolution to increase receptive field w/o adding additional parameters. The goal is to acheive > 85% test accuracy with less than 200K parameters in as many epoch as required. <br>

Our Basic architecture looks like
<img width="198" alt="image" src="https://github.com/Sachin-Bharadwaj/TSAI-S9/assets/26499326/18515210-5424-4191-9783-6b78d0583992"> <br>
It has less than 200K parameters. <br>

We perform following experiments. <br>
**Experiment1**: <img width="806" alt="image" src="https://github.com/Sachin-Bharadwaj/TSAI-S9/assets/26499326/9e3441b8-0d7e-46a9-a404-66c3c2006f24"> <br>
- Observation: The network does not get trained. So, we decide to add BatchNorm to smoothen the loss landscape so that network gets trained. <br>

**Experiment2**: <img width="144" alt="image" src="https://github.com/Sachin-Bharadwaj/TSAI-S9/assets/26499326/f7a087f4-4ed3-4087-ac83-8a94bbde4a3b"> <br>
- Observation: Batch Norm helps in network getting trained but we see heavy overfitting. So we decide to add Dropout to reduce overfitting. <br>

**Experiment3**: <img width="194" alt="image" src="https://github.com/Sachin-Bharadwaj/TSAI-S9/assets/26499326/5dde5642-50b8-45d6-9166-f54e4e1acc23"> <br>
- Observation: We achieve test acc > 82% but we see some overfitting in 70-100 epochs. So lets add some data augmentation to increase the test accuracy and reduce overfitting. <br>

**Experiment4**: <img width="142" alt="image" src="https://github.com/Sachin-Bharadwaj/TSAI-S9/assets/26499326/16d548d1-1771-432a-a9e1-765d813a5fec"> <br>
- Observation: We use HorizontalFlip, ShiftScaleRotate, CoarseDropout as data augmentation. In about 250 epcohs we get test acc > 85% with no overfitting.

 

  




