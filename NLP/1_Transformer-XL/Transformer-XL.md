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

## 3. Model

