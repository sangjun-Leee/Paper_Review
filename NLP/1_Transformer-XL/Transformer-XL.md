## 0. Abstract
 
> Transformers have a potential of learning longer-term dependency, but are limited by a fixed-length context in the setting of language modeling.   

```
트렌스포머는 long-term dependency 문제를 많이 극복했지만, 모델의 구조가 고정된 길이의 문맥에 제한됐다.   
```

Q : limited by a fixed-length context가 정확히 무엇을 뜻하는지? transformer에서 이 부분을 캐치하지 못한 것 같다.

---

> We propose a novel neural architecture Transformer-XL that enables learning dependency beyond a fixed length without disrupting temporal coherence.

```
그래서 Transformer-XL을 제안하였고, 얘는 dependency를 학습하면서 시간적 일관성을 방해하지 않는다.   
```

Q : disrupting temporal coherence가 정확히 무엇을 뜻하는지? 시간적 일관성이 무슨말인가

---

> It consists of a segment-level recurrence mechanism and a novel positional encoding scheme.

```
Transformer-XL은 segment-level의 반복 메커니즘과 새로운 positional encoding을 사용한다.
```

---

> Our method not only enables capturing longer-term dependency, but also resolves the context fragmentation problem.

```
longer-term dependency를 잡을 수 있었고, fragmentation problem을 해결할 수 있었다.
```

---

> As a result, Transformer-XL learns dependency that is 80% longer than RNNs and 450% longer than vanilla Transformers, achieves better performance on both short and long sequences, and is up to 1800+ times faster than vanilla Transformers during evaluation

```
RNN보다 80%, 기존 transformer보다 450% dependency 학습을 향상 시켰다. 짧은 길이, 긴 길이 모두 transformer-xl이 더 좋은 성능을 보였다.
```

Q : RNN보다 transformer에서 향상률이 더 좋은데, 왜 이런 것?

---

</br>

- **SMRY : 기존 Transformer는 long-term dependency문제를 개선했지만, fixed-length context때문에 context fragmentation problem이 발생하였고 Transformer-XL은 segment-level의 반복 매커니즘과 새로운 positional encoding방법으로 문제를 해결하였다.**

</br>

## 1. Introduction

> 
