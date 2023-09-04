---
title: LoRA 정리
date: 2023-09-03 11:58:47 +07:00
modified: 2023-09-03 11:58:47 +07:00
tags: [NLP]
---

## Introduce

`PEFT(Parameter Efficient Fine-Tuning)`는 모델의 모든 파라미터를 튜닝하는 것이 아닌 일부 파라미터만을 튜닝함으로써 모델의 성능을 적은 자원으로도 높게 유지하는 방법론이다. 

GPT-3와 같은 매우 큰 언어 모델이 등장하면서 다양한 문제들을 쉽게 해결할 수 있게 되었지만, 일반 사용자의 하드웨어로는 완전한 파인 튜닝이 불가능해졌다. 또한 각 다운스트림 태스크에 대해 파인 튜닝된 모델을 저장하고 배포하는 것은 많은 비용이 든다. 왜냐하면 파인 튜닝된 모델은 원래 사전 학습 모델과 크기가 동일하기 때문이다. PEFT는 두 가지 문제를 모두 해결하기 위한 방법을 제시한다. 

PEFT 접근 방식은 사전 학습된 LLM의 대부분의 파라미터를 동결하면서 소수의 모델 파라미터만 파인 튜닝하므로 계산 및 저장 비용이 크게 절감되며 이는 LLM의 파인 튜닝 중에 발생하는 <u>catastrophic forgetting</u>를 극복하기도 한다. (미리 학습된 Pretrained weights는 고정된 상태로 유지되기 때문에)
또한 각 다운스트림 데이터 세트에 대해 발생하는 체크포인트는 몇 MB에 불과하면서도 full 파인 튜닝에 필적하는 성능을 달성할 수 있다. 학습된 작은 가중치는 사전 학습된 LLM에 추가되기 때문에 전체 모델을 교체할 필요 없이 작은 가중치를 추가하여 동일한 LLM을 여러 작업에 사용할 수 있다.

<u>즉, 적은 양의 파라미터(ex. 0.01%)로 full fine-tuning에 필적하는 성능을 얻을 수 있다.</u> 
현재 PEFT를 위한 다양한 방법론들이 연구되고 있으며, 그 중 가장 유명한 것 중 하나가 **LoRA**이다. 



## LoRA



















현재 HuggingFace에서 지원하는 PEFT 기법들은 다음과 같다. 

```python
class PeftType(str, enum.Enum):
    PROMPT_TUNING = "PROMPT_TUNING"
    P_TUNING = "P_TUNING"
    PREFIX_TUNING = "PREFIX_TUNING"
    LORA = "LORA"
    ADALORA = "ADALORA"
    ADAPTION_PROMPT = "ADAPTION_PROMPT"
    IA3 = "IA3"

class TaskType(str, enum.Enum):
    SEQ_CLS = "SEQ_CLS"
    SEQ_2_SEQ_LM = "SEQ_2_SEQ_LM"
    CAUSAL_LM = "CAUSAL_LM"
    TOKEN_CLS = "TOKEN_CLS"
    QUESTION_ANS = "QUESTION_ANS"
    FEATURE_EXTRACTION = "FEATURE_EXTRACTION"
```

