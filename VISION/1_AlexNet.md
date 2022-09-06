# <div align="center"> ImageNet Classification with Deep Convolutional Neural Networks </div>

<div align="center">
    Alex Krizhevsky, Ilya Sutskever, Geoffrey E. Hinton
    </br>
    </br>
    Advances in neural information processing systems 25(NIPS)-2012
</div>

</br>

## 0. Abstract

> We trained a large, deep convolutional neural network to classify the 1.2 million high-resolution images in the ImageNet LSVRC-2010 contest into the 1000 different classes.

$\to$ ImageNet LSVRC-2010 이미지 분류 대회에서 1000개의 class를 가진 1200만개의 이미지를 분류하기 위해 깊은 convolution 신경망을 학습시켰다.

---

> On the test data, we achieved top-1 and top-5 error rates of 37.5% and 17.0% which is considerably better than the previous state-of-the-art.

$\to$ 우리 모델은 top-1, top-5 error rate 에서 37.5%, 17.0% 를 기록하며 SOTA 달성하였다.(top error rate : 모델이 예측한 상위 n개 데이터 중 정답이 없는 오류율)

---

> The neural network, which has 60 million parameters and 650,000 neurons, consists of five convolutional layers, some of which are followed by max-pooling layers, and three fully-connected layers with a final 1000-way softmax.

$\to$ 신경망은 6천만개의 파라미터와 65만개의 뉴런, max-pooling을 사용하기도 한 5개의 컨볼루션 층, 마지막 1000 가지를 분류하기 위한 3개의 FCN으로 구성되어있다.

---

> To reduce overfitting in the fully-connected layers we employed a recently-developed regularization method called “dropout” that proved to be very effective.

$\to$ 오버피팅을 줄이기 위해 FCN에 regularization 기법으로 효과적인 dropout을 적용하였다.

---

**SMRY : ImageNet LSVRC-2010 대회에서 SOTA를 달성. convolution layer를 사용하고 max-pooling, dropout 등을 사용함. GPU로 학습시킴.**

</br>

## 1. Introduction

> Until recently, datasets of labeled images were relatively small.

$\to$ 최근까지도, 라벨링 된 데이터 셋은 상대적으로 적었다.

---

> Simple recognition tasks can be solved quite well with datasets of this size, ~. But objects in realistic settings exhibit considerable variability, so to learn to recognize them it is necessary to use much larger training sets.

$\to$ 간단한 구분 작업은 적은 데이터 셋으로도 쉽게 가능하지만, 현실 세계의 사물들은 상당히 다양하므로 그들을 구분하는 것은 훨씬 더 많은 training datasets을 필요로 한다.

---

> The new larger datasets include LabelMe, which consists of hundreds of thousands of fully-segmented images, and ImageNet, which consists of over 15 million labeled high-resolution images in over 22,000 categories.

$\to$ 수십만 개의 fully-segmented 이미지로 이루어진 LabelMe, 1500 만개 이상, 22000 개가 넘는 카테고리의 높은 해상도의 이미지의 ImageNet 과 같은 데이터셋이 최근에야 만들어짐.

---

> To learn about thousands of objects from millions of images, we need a model with a large learning capacity.

$\to$ 수백만 개의 이미지로 부터 학습하려면, 모델이 매우 커야 한다.

---

>  However, the immense complexity of the object recognition task means that this problem cannot be specified even by a dataset as large as ImageNet, so our model should also have lots of prior knowledge to compensate for all the data we don’t have. 

$\to$ 하지만 매우 복잡한 객체 인식 작업은 단순히 방대한 데이터만 있어서는 안되며, 모델이 우리가 가지고 있지 않은 데이터에 대해서도 사전 지식이 필요하다.(데이터만으로는 문제를 극복할 수 없고, 모델에서도 극복해야할 문제가 있다는 의미정도로 이해)

---

> Convolutional neural networks(CNNs) constitute one such class of models.

$\to$ CNN 이 위의 문제를 어느정도 해결해주는 모델 구조 중 하나이다.

---

> Their capacity can be controlled by varying their depth and breadth, and they also make strong and mostly correct assumptions about the nature of imagess (namely, stationarity of statistics and locality of pixel dependencies).

$\to$ CNN 은 크기를 다양하게 조절할 수 있고, 자연의 이미지에 대해서 잘 작동한다.

---

> Thus, compared to standard feedforward neural networks with similarly-sized layers, CNNs have much fewer connections and parameters and so they are easier to train, while their theoretically-best performance is likely to be only slightly worse.

$\to$ 일반적인 FFN과 비교했을 때 비슷한 크기의 층을 가지고 있더라도 CNN은 훨씬 더 적은 파라미터를 가지므로 학습하기 더 쉽다. 최고 성능은 아주 약간 떨어지더라도.

---

> Despite the attractive qualities of CNNs, ~, they have still been prohibitively expensive to apply in large scale to high-resolution images.

$\to$ 이러한 CNN의 매력적인 성질에도 불구하고, 고화질의 이미지에 대해서 계산량이 비싸기 때문에 그동안 잘 쓰지 못했다.

---

> Luckily, current GPUs, ~, are powerful enough to facilitate the training of interestingly-large CNNs, ~.
