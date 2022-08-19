## 0. Abstract
 
> Transformers have a potential of learning longer-term dependency, but are limited by a fixed-length context in the setting of language modeling.   

```
트렌스포머는 long-term dependency 문제를 많이 극복했지만, 모델의 구조가 고정된 길이의 문맥에 제한됐다.   
```

Q : limited by a fixed-length context 가 정확히 무엇을 뜻하는지? transformer 에서 이 부분을 캐치하지 못한 것 같다.

---

> We propose a novel neural architecture Transformer-XL that enables learning dependency beyond a fixed length without disrupting temporal coherence.

```
그래서 Transformer-XL 을 제안하였고, 얘는 dependency 를 학습하면서 시간적 일관성을 방해하지 않는다.   
```

Q : disrupting temporal coherence가 정확히 무엇을 뜻하는지? 시간적 일관성이 무슨말인가

---

> It consists of a segment-level recurrence mechanism and a novel positional encoding scheme.

```
Transformer-XL 은 segment-level 의 반복 메커니즘과 새로운 positional encoding 을 사용한다.
```

---

> Our method not only enables capturing longer-term dependency, but also resolves the context fragmentation problem.

```
longer-term dependency 를 잡을 수 있었고, fragmentation problem 을 해결할 수 있었다.
```

---

> As a result, Transformer-XL learns dependency that is 80% longer than RNNs and 450% longer than vanilla Transformers, achieves better performance on both short and long sequences, and is up to 1800+ times faster than vanilla Transformers during evaluation

```
RNN 보다 80%, 기존 transformer 보다 450% dependency 학습을 향상 시켰다. 짧은 길이, 긴 길이 모두 transformer-xl 이 더 좋은 성능을 보였다.
```

Q : RNN 보다 transformer 에서 향상률이 더 좋은데, 왜 이런 것?

---

</br>

- **SMRY : 기존 Transformer 는 long-term dependency 문제를 개선했지만, fixed-length context 때문에 context fragmentation problem 이 발생하였고 Transformer-XL 은 segment-level 의 반복 매커니즘과 새로운 positional encoding 방법으로 문제를 해결하였다.**

</br>

## 1. Introduction

> RNNs are difficult to optimize due to gradient vanishing and explosion, and the introduction of gating in LSTMs and the gradient clipping technique imght not be sufficient to fully address this issue.

```
RNN 은 gradient vanishing, explosion 문제 때문에 최적화 하기가 힘들다.
이 문제를 극복하기 위해 제안된 gating 방법의 LSTM 이나 gradient clipping 기법도 이 문제를 완전히 다루기에는 충분하지 않다.
```

---

> Despite the success, the LM training in Al-Rfou et al. (2018) is performed on separated fixed-length segments of a few hundred characters, without any information flow across segments.

```
attention 기반의 transformer 는 직접적으로 거리가 먼 단어 쌍들을 연결함으로써 long-term dependency 를 보다 쉽게 최적화 할 수 있지만
언어 모델 학습이 segment 간의 정보 교류 없이 수백개의 문자가 분리된 고정된 길이의 segment 에서 이루어진다.
```

---

> As a consequence of the fixed context length, the model cannot capture any longer-term dependency beyond the predefined context length.

```
고정된 길이로 학습한 결과, 모델은 그 길이 이상을 넘어가는 어떠한 long-term dependency 도 잡아내질 못한다.
```

---

>  In addition, the fixed-length segments are created by selecting a consecutive chunk of symbols without respecting the sentence or any other semantic boundary.

```
게다가, 고정된 길이의 segments 는 문장이나 의미 경계를 고려하지 않고 그저 연속적인 기호 덩어리를 선택하여 생성된다.
```

---

> Hence, the model lacks necessary contextual information needed to well predict the first few symbols, leading to inefficient optimization and inferior performance. We refer to this problem as context fragmentation.

```
그러므로, 모델은 처음 몇 개의 기호를 잘 예측하는 데 필요한 필수적인 문맥 정보가 부족하게 되고 이는 비효율적인 최적화와 성능 저하를 초래한다.
이러한 문제를 여기서는 'context fragmentation' 이라고 정의한다.
```

---

> Instead of computing the hidden states from scratch for each new segment, we reuse the hidden states obtained in previous segments.

```
각각의 새로운 segment 에서 hidden states 를 계산하는 대신에, 우리는 전 단계의 segments 에서 얻은 hidden states 를 재사용 한다.
```

---

> The reused hidden states serve as memory for the current segment, which builds up a recurrent connection between the segments.

```
재사용 된 hiddens states 는 현재 segment의  메모리 역할을 하며, segment 간의 재귀적 연결을 만든다.
```

---

> As a result, modeling very long-term dependency becomes possible because information can be propagated through the recurrent connections.

```
그 결과, 정보들이 재귀적 연결을 통해서 전달되므로 long-term dependency 모델링이 가능해졌다.
```

---

> Meanwhile, passing information from the previous segment can also resolve the problem of context fragmentation.

```
한편, 이전 segment 로부터 전달된 정보는 context fragmentation problem 도 해결할 수 있다.
```

---

> More importantly, we show the necessity of using relative positional encodings rather than absolute ones, in order to enable state reuse without causing temporal confusion.

```
우리는 temporal confusion 을 발생시키지 않고 state 를 재사용하기 위해 상대적인 positional encoding 사용이 필요함을 보였다.
```

---

> Our main technical contributions include introducing the notion of recurrence in a purely self-attentive model and deriving a novel positional encoding scheme.

```
우리는 self-attentive 모델에 재귀적 개념을 사용한 것과 새로운 positional encoding 방법을 제시하였다.
```

---

</br>

- **SMRY : Transformer-XL의 저자들은 long-term dependency 와 fixed-length context 때문에 발생하는 context fragmentation 문제를 해결하기 위해 attention 에 재귀적 개념을 접목시켰고, 새로운 상대적 positional encoding 기법을 사용하였다.**

</br>

## 2. Related Work

</br>

## 3. Model

> In this work, we stick to the standard neural approach to modeling the conditional probability.

```
이 연구에서 우리는 조건부 확률을 모델링하기 위해 일반적인 뉴럴 접근방식을 고수했다.
```

</br>

### 3-1. Vanilla Transformer Language Models

> the central problem is how to train a Transformer to effectively encode an arbitrarily long context into a fixed size representation.

```
주된 문제는 긴 context 를 어떻게 효과적으로 고정된 크기의 representation 으로 인코딩하여 Transformer 를 train 할 것인가 이다.
```

---

> Given infinite memory and computation, a simple solution would be to process the entire context sequence using an unconditional Transformer decoder, similar to a feed-forward neural network. However, this is usually infeasible with the limited resource in practice.

```
제한 없는 컴퓨팅 자원이 주어진다면, FFNN 처럼 제한없이 Transformer decoder 에 전 문맥을 사용하면 된다.
하지만 실제로는 그것은 불가능하다.
```

---

> One feasible but crude approximation is to split the entire corpus into shorter segments of manageable sizes, and only train the model within each segment, ignoring all contextual information from previous segments.

```
한가지 가능하지만 썩 좋지않은 접근방법은 전체 단어집합을 다룰 수 있는 크기의 짧은 segment 들로 나누는 것이다.
그리고 모델은 이전 segment 에서 온 문맥 정보를 무시하고 각각의 segment 에 대해서만 train 하게 된다.
```

---

> Under this training paradigm, information never flows across segments in either the forward or backward pass.

```
이러한 훈련 패러다임에서는 segments 간 정보가 forward 또는 backward 방향으로 교류되지 못한다.
```

---

> There are two critical limitations of using a fixed-length context.

```
fixed-length context 를 사용하는 데에는 두가지 치명적인 한계가 존재한다.
```

---

> First, the largest possible dependency length is upper bounded by the segment length, which is a few hundred on character-level language modeling.

```
첫 번째는, segment 길이에 의해 가능한 최대 dependency 길이가 상한된다는 것이다.
```

---

> Second, though it is possible to use padding to respect the sentence or other semantic boundaries, in practice it has been standard practice to simply chunk long text into fixed-length segments due to improved efficiency.

```
두 번째로 문장이나 의미적 경계를 고려하여 패딩을 사용할 수 있지만,
실제로는 효율성의 향상을 위해 fixed-length segments 를 사용하는 것이 일반적 이었다.
```

---

> However, simply chunking a sequence into fixed-length segments will lead to the context fragmentation problem.

```
하지만, 단순히 fixed-length segments 로 chunking 하는 것은 context fragmentation 문제를 야기할 수 있다.
```

---

> During evaluation, at each step, the vanilla model also consumes a segment of the same length as in training, but only makes one prediction at the last position.

```
evaluation 동안, 매 단계에서 바닐라 모델은 training 때와 같은 길이의 segment를 보지만, 마지막 위치에서 하나의 예측을 한다.
```

Q : 이 부분 잘 이해가 안됨

---

> Then, at the next step, the segment is shifted to the right by only one position, and the new segment has to be processed all from scratch.

```
그리고 다음 단계에서 segment 는 하나의 position만 움직여져 다시 예측을 수행한다.
```

![img]()

---

> this procedure ensures that each prediction utilizes the longest possible context exposed during training, and also relieves context fragmentation issue encountered in training.

```
이러한 과정은 각 예측 단계가 training 시 학습했던 최대한의 긴 문맥을 활용한다.
그리고 training 시 발생했던 context fragmentation 문제를 어느정도 감소시킨다.
```

---

> However, this evaluation procedure is extremely expensive.

```
하지만 이러한 evaluation 과정은 매우 비용이 크다.
```

</br>

### 3-2. Segment-Level Recurrence with State Reuse

> To address the limitations of using a fixed-length context, we propose to introduce a recurrence mechanism to the Transformer architecture.

```
fixed-length context 사용에 따른 한계를 다루기 위해 우리는 재귀적 방법을 Transformer architecture 에 사용했다.
```

---

> During training, the hidden state sequence computed for the previous segment is *fixed* and *cached* to be reused as an extended context when the model processes the next new segment.

```
trainig 동안 모델이 다음 segment 를 처리할 때, 이전 segment 에서 계산된 hidden state 는
extended context 로 다시 사용되기 위해 고정되고 저장된다.
```

---

> this additional input allows the network to exploit information in the history, leading to an ability of modeling longer-term dependency and avoiding context fragmentation.

```
이 추가적인 입력은 네트워크가 과거 정보를 활용할 수 있게 해주고 이는 더 긴 dependency 를 모델링 할 수 있게 됩니다. 그리고 context fragmentation 도 피할 수 있습니다.
```

---
 
 </br>
 
 길이 $L$,  두 segment $\mathbf{s}\_{\tau} = \[x_{\tau, 1}, \cdots, x_{\tau, L}] \ , \mathbf{s}\_{\tau + 1} = \[x_{\tau + 1, 1}, \cdots, x_{\tau + 1, L}]$, n-th layer hidden state $\mathbf{h}^{n}\_{\tau}.
 
 $$\mathbf{\tilde{h}}^{n-1}\_{\tau + 1} = \[SG(\mathbf{h}^{n-1}\_{\tau}) \circ \mathbf{h}^{n-1}\_{\tau + 1}]$$
 
 
 Q
 1. 새로운 h
