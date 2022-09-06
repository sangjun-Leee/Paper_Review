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

> 
