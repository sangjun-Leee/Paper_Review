## 0. Abstract

> With the capability of modeling bidirectional contexts, denoising autoencoding based pretraining like **BERT** achieves better performance than pretraining approaches based on autoregressive language modeling.

```
양방향 컨텍스트를 모델링하게 되면서, BERT와 같은 노이즈를 제거하는 오토인코딩에 기반한 모델이
autoregressive 한 언어 모델보다 더 좋은 성능을 보인다.
```

---

> However, relying on corrupting the input with masks, **BERT** neglects dependency between the masked positions and suffers from a pretrain-finetune discrepancy.

```
하지만, mask 된 input에 의존하는 BERT는 mask된 위치간의 의존성을 무시하고 pretrain 과 finetune 의 불일치 문제가 있다.
(실제 모델이 마주하게 되는 input에는 [mask] 토큰이 없기 때문.)
```

---

> we propose **XLNet**, a generalized autoregressive pretraining method that (1) enables learning bidirectional contexts by maximizing the expected likelihood over all permutations of the factorization order and (2) overcomes the limitations of **BERT** thanks to its autoregressive formulation.

```
우리는 autoregressive pretraining 방법을 사용해서 (1) context의 모든 순열의 likelihood 가 최대가 되게 하여 bidirectional contexts 를 학습 가능하게 하고,
(2) autoregressive 한 구조 때문에 BERT 의 한계를 극복한 XLNet을 제안한다.
```

---

- **SMRY : Autoencoding 방식의 BERT와 같은 pretraining 방식의 모델이 autoregressive 방식의 모델보다 좋은 성능을 보였다. 하지만 BERT 는 input에 \[mask] 와 같이 finetune 단계에서는 전혀 볼 수 없는 형태의 문자가 들어가기 때문에 pretraining 단계과 finetune 단계에서 discrepancy 가 발생한다.
또한 mask 된 문자 사이의 dependency 는 무시한다는 문제점이 존재한다. 따라서 문제점을 해결하고자 autoregressive 방식의 모델을 사용했고, bidirectional context 를 학습할 수 있도록 바꾸었다.** 

</br>

## 1. Introduction

> 
