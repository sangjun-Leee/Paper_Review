# VQA: Visual Question Answering

<div align="center">
  Aishwarya Agrawal, Jiasen Lu, Stanislaw Antol, Margaret Mitchell, C. Lawrence Zitnick, Dhruv Batra, Devi Parikh
  </br>
  </br>
  International Conference on Computer Vision(ICCV)-2015
</div>

</br>

## 0. Abstract

> We propose the task of *free-form* and *open-ended* Visual Question Answering (VQA).

$\to$ free-form 이고 open-ended 한 Visual Question Answering(VQA) task 를 제안한다.

---

> Given an image and a natural language question about the image, the task is to provide an accurate natural language answer.

$\to$ 이미지와 그 이미지에 관한 질문이 주어지고, 자연어로 질문에 대한 정답을 제공해야 하는 task.

---

> Visual questions selectively target different areas of an image, including background details and underlying context.

$\to$ 질문들은 배경 디테일과 기본 context 를 포함해서 이미지의 다양한 영역을 대상으로 삼는다.

---

> As a result, a system that succeeds at VQA typically needs a more detailed understanding of the image and complex reasoning than a system producing generic image captions.

$\to$ VQA 를 잘 하려면 일반적인 이미지 캡셔닝을 하는 것 보다 이미지에 대해 더 자세히 이해해야 하고, 복잡한 추론을 할 수 있어야 한다.

---

> We provide a dataset containing ~0.25M images, ~0.76M questions, and ~10M answers, and discuss the information it provides.

$\to$ 25만개의 이미지와 76만개의 질문, 천 만개의 정답이 있는 dataset을 제공하고 그것이 주는 정보를 논의합니다.

---

**SMRY : Visual Question Answering(VQA) dataset은 주어진 이미지와 그에 대한 질문의 답을 맞춰야 하는 작업으로, free-form & open-ended 하고 이미지에 대한 자세한 이해와 깊은 추론을 해야 풀 수 있는 task이다.**

</br>

## 1. Introduction

> an ideal task should (i) require multi-modal knowledge beyond a single sub-domain (such as CV) and (ii) have a well-defined quantitative evaluation metric to track progress.

$\to$ 이상적인 task 는 하나의 domain 이 아닌 multi-modal 지식을 필요로 해야하고, 과정을 확인하기 위해 잘 정의된 quantitative evaluation metric 이 있어야 한다.

---

> For some tasks, such as image captioning, automatic evaluation is still a difficult and open research problem.

$\to$ 이미지 캡셔닝과 같은 몇몇 task 에서는, automatic evaluation 은 아직도 어렵고 공개된 연구과제이다.

---

> A VQA system takes as input an image and a free-form, open-ended, natural language question about the image and produces a natural language answer as the output.

$\to$ VQA 시스템은 입력으로 이미지와 해당 이미지에 대한 자유로운 형식과 주제의 자연어 질문을 받아 자연어 정답을 결과로 출력하는 것이다.

---

> Open-ended questions require a potentially vast set of AI capabilities to answer - fine-grained recognition, object detection, activity recognition, knowledge base reasoning, and commonsense reasoning.

$\to$ Open-ended question 은 인공지능이 정답을 구하기 위해서 많은 능력을 가져야 하도록 한다. - 세심한 인지, 물체 탐지, 행동 인식, 지식 기반 추론, 일반 상식과 같은.

---

> While the answer to many questions is simply “yes” or “no”, the process for determining a correct answer is typically far from trivial.

$\to$ 많은 질문에 대한 정답은 단순히 "예", "아니오" 이지만, 정답을 결정하는데 필요한 과정은 결코 단순하지 않다.

---

> Moreover, since questions about images often tend to seek specific information, simple oneto-three word answers are sufficient for many questions.

$\to$ 게다가, 이미지에 관한 질문들이 세부적인 정보를 필요로 하기 때문에(지엽적), 단순히 하나 내지 세 단어 정도 만으로도 많은 질문에 대한 충분한 정답이 될 수 있다. (지엽적인 문제의 정답은 대부분 확실하기 때문에 라고 생각한다.)

---

> In such scenarios, we can easily evaluate ~

$\to$ 이러한 이유 때문에 쉽게 평가 가능하다.

---

> In this paper, we present both an open-ended answering task and a multiple choice task.

$\to$ 본 논문에서는, open-ended answering task 와 multiple-choice task 모두 제공한다.

---

> Unlike the open-ended task ~, the multiple-choice task only requires an algorithm to pick from a predefined list of possible answers.

$\to$ open-ended task 와는 다르게, multiple-choice task 는 미리 정의된 가능한 정답 리스트로부터 고르기만 하면 된다.

---

> Three questions were collected for each image or scene.

$\to$ 각 이미지나 풍경에 대해서 세개의 질문들을 모았다.

---

> Each question was answered by ten subjects along with their confidence.

$\to$ 각 질문에 대해서 10명의 사람들이 정답을 제시했다.

---

> The dataset contains over 760K questions with around 10M answers.

$\to$ dataset 은 76만개의 질문들과 천만개의 정답으로 구성되어있다.

---

> We analyze the types of questions asked and the types of answers provided.

$\to$ 우리는 질문들과 그 답변들을 분석했다.

---

**SMRY : VQA 는 다양한 AI 능력을 요구하는 open-ended, free-form 형태의 dataset 이다. 평가 또한 쉽고 multiple-choice 문제도 포함한다. 76만개의 질문들과 천만개의 정답들로 이루어져 있다. 각 질문에 대한 정답을 분석하여 평가하였다.**

</br>

## 2. Related Work




