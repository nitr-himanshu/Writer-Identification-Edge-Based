# Writer-Identification-IAM-2

 **Dataset** : IAM 
 **Feature extraction** :  Edge based directional feature.

## Results

- For 5 writers where each writer has written 1 form the accuracy is 86.15% (reults are in excel sheet) using neural network. But for whole dataset the accuracy is very poor.

### Test 1

```py
m.compress(64)
m.detectEdge(200,200)
m.featureExtraction(4)
```
13.21% using SVM.
