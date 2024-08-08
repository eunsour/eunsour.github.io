---
title: "[리뷰] Mixture-of-Agents Enhances Large Language Model Capabilities"
date: 2024-08-08 12:00:00 +09:00
modified: 2024-08-08 12:00:00 +09:00
tags: 
    - NLP
    - LLM
usemathjax: true
---

>Wang, Junlin, Jue Wang, Ben Athiwaratkun, Ce Zhang, and James Zou. "Mixture-of-Agents Enhances Large Language Model Capabilities." arXiv preprint arXiv:2406.04692, 2024. [[paper]](https://arxiv.org/abs/2406.04692) [[github]](https://github.com/togethercomputer/moa)

<br>

# Introduce
대규모 언어 모델(LLM)은 방대한 데이터로 사전 학습된 후 인간의 선호도에 맞춰 조정되어 놀라운 성과를 보여주고 있다. 하지만 모델 규모와 학습 데이터에 관한 본질적 한계가 여전히 존재한다. LLM을 더욱 발전시키려면 막대한 비용과 광범위한 재학습이 필요하다.

동시에 각기 다른 LLM은 고유한 강점을 가지고 있으며, 다양한 작업 측면에 특화되어 있다. 예를 들어, WizardLM과 같은 일부 모델은 복잡한 명령어 추종에 탁월한 반면, Code Llama나 Deepseek-coder와 같은 모델은 코드 생성에 더 적합하다. 이러한 특화된 모델들은 각자의 영역에서 뛰어난 성능을 보여주며, 사용자의 특정 요구사항에 맞춰 선택할 수 있는 다양한 옵션을 제공한다.

LLM들이 각자 다른 강점을 가진 이런 다양성은 흥미로운 질문을 불러일으킨다. 여러 LLM의 전문성을 한데 모아 더 유능하고 강력한 모델을 만들 수 있을까? 저자들은 이 질문에 '**그렇다**'고 답한다.

저자들은 '**LLM의 협업성(collaborativeness of LLMs)**'이라는 독특한 현상을 발견했다. 이는 한 LLM이 다른 모델의 결과를 입력값으로 받을 때, 그 입력 모델의 성능이 상대적으로 낮더라도 더 나은 응답을 생성하는 경향을 말한다. 이 현상은 여러 LLM의 강점을 결합해 전체적인 성능을 끌어올릴 수 있는 가능성을 시사한다.

다음 그림은 6개의 인기 있는 LLM에 대한 AlpacaEval 2.0 벤치마크에서의 LC(length-controlled) 승률을 보여준다.



<center><img src="/assets/img/moa/1.png" style="zoom: 50%;" /></center>

<br>


모델들에게 다른 모델이 독립적으로 생성한 답변을 제공하면, LC 승률이 크게 향상된다. 이는 LLM 간의 협업 현상이 광범위하게 존재한다는 것을 보여준다. 특히 주목할 만한 점은, 이러한 성능 개선은 보조 답변의 품질이 해당 LLM이 혼자 생성할 수 있는 답변보다 낮은 경우에도 발생한다는 것이다.


본 논문에서는 이런 결과를 토대로 여러 LLM을 활용해 생성 품질을 단계적으로 향상시키는 **'Mixture-of-Agents (MoA)'** 방법론을 소개한다.

이 방법은 다음과 같이 동작한다:

1. 첫 번째 단계에서 \\(A_{1,1}, ... A_{1,n}\\)으로 표시되는 LLM들이 주어진 프롬프트에 대해 각자 독립적으로 응답을 생성한다.
2. 이렇게 생성된 응답들은 다음 단계의 에이전트 \\(A_{2, 1}, ...A_{2, n}\\)에게 전달된다. 이때 첫 번째 단계의 모델들을 재사용할 수도 있다.
3. 다음 단계의 에이전트들은 이 응답들을 바탕으로 더 정교한 답변을 만들어낸다.
4. 이 과정은 원하는 수준의 강력하고 포괄적인 응답을 얻을 때까지 여러 번 반복된다.

<br>

MoA의 각 레이어에 들어갈 LLM을 선택할 때는 효과적인 협업과 전반적인 응답 품질 향상을 위해 신중을 기해야 한다. 이 선택 과정에는 두 가지 주요 기준이 적용된다:

1. **성능**
    - 레이어 \\(i\\)에 있는 모델의 평균 승률이 레이어 \\(i + 1\\)에 포함될지를 결정하는 데 중요한 역할을 한다.
    - 입증된 성능 지표를 기준으로 모델을 선택하면 더 높은 품질의 결과를 얻을 수 있다.
2. **다양성**
    - 모델 출력의 다양성도 중요하다.
    - 서로 다른 특성을 가진 모델들이 생성한 응답은, 같은 모델이 여러 번 생성한 응답보다 훨씬 더 큰 기여를 한다.

이 두 기준을 활용해 MoA는 개별 모델의 약점을 보완하고, 여러 모델의 장점을 결합해 전반적인 응답 품질을 높이는 것을 목표로 한다.

저자들은 다양한 측면에서 응답 품질을 평가하기 위해 AlpacaEval 2.0, MT-Bench, FLASK 벤치마크를 사용해 종합적인 평가를 수행했다. 그 결과, AlpacaEval 2.0에서 65.8%의 새로운 최고 승률(SOTA)을 달성했는데, 이는 이전 최고 기록인 GPT-4o의 57.5%를 크게 앞선 결과이다.

<br>

# Mixture-of-Agents Methodology

### collaborativeness of LLMs

LLM의 협업성, 특히 다른 모델의 결과를 참조해 더 높은 품질의 응답을 생성하는 능력을 먼저 살펴본다. 많은 현대 LLM이 이런 협업 기능을 갖추고 있다.

LLM의 협업 과정에서 두 가지 주요 역할이 있다:

1. **Proposer**
    - 다른 모델이 참조할 수 있는 유용한 응답을 생성한다.
    - 높은 점수를 받는 답변보다는 다양한 관점과 맥락을 제공하는 데 중점을 둔다.
2. **Aggregator**
    - 다른 모델의 응답을 고품질 출력으로 합성한다.
    - 품질이 낮은 입력도 통합해 전체적인 출력 품질을 향상시킨다.

<br>

이 역할들을 실증적으로 검증한 결과, 대부분의 LLM이 두 역할을 모두 수행할 수 있지만, 일부 모델은 특정 역할에 더 특화되어 있음을 발견했다. 예를 들어:

- GPT-4o, Qwen1.5, LLaMA-3: 두 역할 모두에 효과적인 다목적 모델
- WizardLM: Proposer로서 탁월하지만 Aggregator로서는 덜 효과적

이러한 협업 잠재력을 더욱 강화하기 위해, 여러 Aggregator를 순차적으로 사용하는 방법을 제안한다. 이 방식은 여러 모델의 강점을 활용해 응답을 반복적으로 개선하며, '**Mixture-of-Agents**' 설계의 기초가 된다.

<br>


### Mixture of Agents

<center><img src="/assets/img/moa/2.png" style="zoom: 50%;" /></center>

<br>

**MoA의 구조 (Figure 2 참조):**
- \\(l\\)개의 레이어로 구성
- 각 layer-\\(i\\)는 \\(A_{i,1}, A_{i,2},...,A_{i,n}\\)으로 구성
- LLM은 레이어 내 또는 레이어 간 재사용 가능

<br>

**주요 특징:**
1. 같은 LLM을 여러 번 사용할 경우, temperature 샘플링으로 다양한 출력 생성 (single-proposer 구조)
2. 각 LLM \\(A_{i,j}\\)는 입력 텍스트를 처리하고 연속 생성
3. 파인 튜닝 불필요, 프롬프트와 LLM 생성 인터페이스만 사용

<br>

**수학적 표현: 입력 프롬프트 \\(x_1\\)에 대한 \\(i\\)번째 MoA 레이어 \\(y_i\\)의 출력:**
\begin{aligned}
\\(y_i = \bigoplus^n_{j=1}[A_{i,j}(x_i)] + x_1, x_{i+1} = y_i\\)
\end{aligned}

여기서 '+’ 는 텍스트 연결, '⊕' 는 집계 및 합성 프롬프트 적용(Table 1 참조)을 의미한다.

**실제 적용:**
- 마지막 레이어에서는 하나의 LLM만 사용
- 최종 출력: \\(l\\)번째 계층의 LLM 출력 (\\(A_{l,1}(x_l)\\))
- 이 최종 출력을 기반으로 메트릭 평가

<br>

<center><img src="/assets/img/moa/3.png"/></center>

<br>

### Analogy to Mixture-of-Experts

**Mixture-of-Experts (MoE) 개요:**
- 여러 전문가 네트워크가 각각 다른 기술에 특화된 잘 정립된 기법
- 복잡한 문제 해결에 다양한 모델 기능을 활용해 성공적
- MoA 방법의 영감이 됨

**MoE 구조:**
- MoE 레이어의 스택으로 구성
- 각 레이어: \\(n\\)개의 전문가 네트워크 + gating network + residual connection

**수학적 표현: layer \\(i\\)의 출력:**
\begin{aligned}
\\(y_i=∑^n_{j=1}G{i,j}(x_i)E_{i,j}(x_i)+x_i\\)
\end{aligned}


**여기서:**
- \\(G_{i,j}\\): 전문가 \\(j\\)에 대한 게이팅 네트워크 출력
- \\(E_{i,j}\\): 전문가 네트워크 \\(j\\)의 함수

**장점:**
- 다양한 기술 학습 가능
- 작업의 여러 측면에 집중 가능

<br>

**MoA 프레임워크의 주요 특징:**
1. MoE 개념의 확장:
    - 활성화 수준이 아닌 모델 수준에서 작동
    - 여러 개의 완전한 LLM을 여러 계층에 걸쳐 활용
2. 작동 방식:
    - 전적으로 프롬프트 인터페이스를 통해 작동
    - 내부 활성화나 가중치 수정 불필요
3. LLM의 역할:
    - 게이팅 네트워크와 전문가 네트워크의 기능을 통합
    - 프롬프트 해석 및 일관된 출력 생성 능력으로 입력을 효과적으로 정규화
4. 장점:
    - 미세 조정 관련 계산 오버헤드 없음
    - 높은 유연성과 확장성
    - 규모나 아키텍처에 관계없이 최신 LLM에 적용 가능

<br>

## Evaluation

**Models**

- 오픈 소스 모델만 사용
- 목록: Qwen1.5-110B-Chat, Qwen1.5-72B-Chat, WizardLM-8x22B, LLaMA-3-70B-Instruct, Mixtral-8x22B-v0.1, dbrx-instruct



**MoA 구성**

1. MoA
    - 3개의 MoA Layer, 각 layer에 동일한 모델 세트
    - 최종 Aggregator: Qwen1.5-110B-Chat
2. MoA w/ GPT-4o
    - 고품질 출력 우선
    - 최종 Aggregator: GPT-4o
3. MoA-Lite
    - 2개의 MoA Layer
    - 비용 효율적, AlpacaEval 2.0에서 1.8% 품질 향상
    - 최종 Aggregator: Qwen1.5-72B-Chat

모든 모델의 라이선스 조건을 준수하며, 오픈 소스 모델 추론은 [Together Inference Endpoint](https://api.together.ai/playground/chat)를 통해 실행하였다.



아래에서는 세 가지 주요 벤치마크, AlpacaEval 2.0, MT-Bench, FLASK에 대한 평가 결과를 제시한다. 

<center><img src="/assets/img/moa/4.png" style="zoom: 50%;" /></center>

<br>

<center><img src="/assets/img/moa/5.png" style="zoom: 50%;" /></center>

<br>

### What Makes Mixture-of-Agents Work Well?

**1. Mixture-of-Agents(MoA)와 LLM 랭커의 성능 비교**

<center><img src="/assets/img/moa/6.png" style="zoom: 50%;" /></center>

- 비교 대상:
  - MoA: aggregator 모델이 새로운 결과 생성
  - LLM 기반 랭커: 제안자가 생성한 답변 중 하나를 선택
- 결과:
  - MoA가 LLM 기반 랭킹 방식보다 월등히 우수한 성능 보임
  - 시사점: aggregator는 단순 답변 선택이 아닌 정교한 집계 수행 가능

<br>

**2. MoA의 제안된 답변 활용 경향**

<center><img src="/assets/img/moa/7.png" style="zoom: 50%;" /></center>

- 분석 방법:
  - 유사성 점수(ex: BLEU)를 사용해 aggregator와 proposer의 응답 비교
  - 각 샘플에서 유사도 점수와 GPT-4 기반 평가자의 선호도 점수 간 스피어만 상관 계수 계산
- 결과:
  - 승률과 BLEU 점수 간 양의 상관관계 확인
  - 다른 유사도 측정법(레벤슈타인, TF-IDF)도 선호도 점수와 양의 상관관계 보임 (부록 A 참조)

<br>

**3. 모델 다양성과 proposer 수의 영향**

<center><img src="/assets/img/moa/8.png" style="zoom: 50%;" /></center>

- 분석 방법:
  - 각 계층의 proposer 수(\\(n\\)) 변화에 따른 결과 품질 분석
  - 'single-proposer' vs 'multiple-proposer' 설정 비교
- 결과:
  - \\(n\\) 증가에 따라 점수 단조롭게 증가
  - 다양한 LLM 사용 시 일관되게 더 나은 결과 도출
- 시사점:
  - 보조 정보가 많을수록 성능 향상
  - 각 MoA 레이어에 다양한 LLM 에이전트 사용 시 성능 개선 가능
  - MoA의 폭 확장이 향후 연구 방향이 될 수 있음

<br>

**4. Mixture-of-Agent 생태계 내 모델 특화**

<center><img src="/assets/img/moa/9.png" style="zoom: 50%;" /></center>

- GPT-4o, Qwen, LLaMA-3: 다목적 모델로 효과적
- WizardLM: proposer로 탁월하나 aggregating 능력은 제한적

<br>

### Budget and Token Analysis

<figure align="center">
<center><img src="/assets/img/moa/10.png" style="zoom: 50%;" /></center>
<figcaption>AlpacaEval 2.0 벤치마크의 평균 추론 비용 대비 LC 승률 표시</figcaption>
</figure>

<br>

<figure align="center">
<center><img src="/assets/img/moa/11.png" style="zoom: 50%;" /></center>
<figcaption>LC 승률과 Tflops수 관계 표시</figcaption>
</figure>


주요 발견:
1. 품질 우선: MoA가 최적
2. 품질-비용 균형: MoA-Lite가 GPT-4o와 비슷한 비용으로 더 높은 품질 제공
3. MoA-Lite: GPT-4 터보보다 4% 우수한 성능, 2배 이상 비용 효율적
4. MoA와 MoA-Lite가 비용 효율성과 성능 면에서 우수한 선택임을 보여줌

<br>

## Conclusion

**MoA(Mixture-of-Agents)** 접근법은 여러 LLM의 기능을 효과적으로 통합하여 AI 성능을 크게 향상시켰다. 이 방법은 반복적 협업을 통한 단계적 성능 개선을 실현하며, 특히 파인 튜닝 없이 프롬프트와 LLM 생성 인터페이스만으로 구현 가능하다는 점에서 주목할 만하다. MoA의 핵심은 다양한 LLM의 강점을 활용한 협업 시스템에 있으며, proposer와 aggregator 역할의 효과적 분배를 통해 성능을 최적화한다. 연구 결과, 모델 다양성과 proposer 수 증가에 따라 성능이 향상되는 것으로 나타났다.

그러나 MoA의 한계점으로는 <u>첫 번째 토큰 생성 시간(TTFT)</u>이 길어질 수 있다는 점이 주요 문제로 지적된다. 이는 마지막 MoA 계층까지 처리한 후에야 첫 토큰을 결정할 수 있기 때문이다. 이러한 한계를 극복하기 위해, 향후 연구에서는 MoA 레이어 수 최적화, 응답의 청크 단위 집계 방법 모색, 다양한 최적화 기법을 통한 성능 및 효율성 개선 등이 필요할 것으로 보인다. 

<br>
