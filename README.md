#light_ml4j
A framework implemented only in native java.

## Introduction
Due to the limited power consumption and memory on mobile devices, 
other third-party libraries such as tensorflow Lite often cannot 
be successfully embedded into mobile applications due to their 
large dependent package size. Therefore, the package is dedicated 
to providing a recommendation framework that can be trained and predicted 
on mobile phones, also known as federated learning recommendation, 
but it is not suitable for task processing with dense features such as 
image classification and speech recognition.



On the end, local data cannot be uploaded due to privacy protection and other reasons. 
Therefore, recommendation tasks on mobile phones often need to be trained on the 
end rather than just predicted. At the same time, because the package only relies 
on native Java, it cannot effectively use hardware for computing acceleration. 
Therefore, it is not suitable for dense feature models and complex models. 
For simple recommendation models, the features are generally sparse, so it does not
 require high computing power, The framework can be well adapted to this task.

## Features
- MLP
- softmax,logistic,square loss

##TODO:
- support spare feature, embeddings
- support rnn, lstm