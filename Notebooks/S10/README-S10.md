# TSAI-S10

This assignment is about training a custom resnet architecture for CIFAR10 dataset as shown below: <br>
<img width="535" alt="image" src="https://github.com/Sachin-Bharadwaj/TSAI-Assignments/assets/26499326/f13e8a15-b3a2-44bf-83cf-34c465bb38a8"> <br>

Goal: We have to achieve test acc > 90% in maximum of 24 epochs and have to use one cycle scheduler without annealing. The configeration of one cycle scheduler is shown below: <br>
<img width="620" alt="image" src="https://github.com/Sachin-Bharadwaj/TSAI-Assignments/assets/26499326/514df0d1-8a1d-43c9-a552-75983cf393a1"> <br>.
We ensured that max learning rate in one cycle policy is achieved @ 5th epoch. Maximum learning rate is figured out using learning rate finder.

**Summary**: <br>
We are able to achieve test acc > 90% consistently beyond 17 epochs and overfitting is within our allocated budget of 5%. <br>
<img width="740" alt="image" src="https://github.com/Sachin-Bharadwaj/TSAI-Assignments/assets/26499326/90584daf-170c-45f3-95a4-92a1678cca2f">




 

  




