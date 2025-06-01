---
title: "[리뷰] A Survey of Efficient LLM Inference Serving"
date: 2025-06-01 12:00:00 +09:00
modified: 2025-08-01 12:00:00 +09:00
tags: 
    - NLP
    - LLM
usemathjax: true
---

>Zhen, Ranran, et al. "Taming the Titans: A Survey of Efficient LLM Inference Serving." arXiv preprint arXiv:2504.19720, 2024. [[paper]](https://arxiv.org/abs/2504.19720)


최근 오픈소스 LLM들이 빠르게 발전하며 모델 아키텍처와 기능이 지속적으로 업데이트되고 있으며, 이에 대한 수요 또한 급증하고 있다. 하지만 LLM의 방대한 매개변수와 어텐션 메커니즘은 메모리 및 컴퓨팅 자원에 큰 부담을 주어, 추론 서비스 시 낮은 지연 시간과 높은 처리량 달성을 어렵게 만든다. 이러한 어려움은 서비스 수준 목표(SLO; Service Level Objectives)를 충족하기 위한 추론 서비스 최적화 연구를 촉진시켰다. 

본 논문은 LLM 추론 서비스의 최적화 방법들을 인스턴스 수준, 클러스터 규모 전략, 새로운 시나리오 및 기타 중요 영역으로 계층화하여 체계적으로 조사하며, 기존 연구들의 한계를 보완하여 최첨단 방법론에 대한 세분화된 분류를 제공하고 미래 연구 방향을 제시하고자 한다.


# 1. Introduction
이 논문은 그림 1과 같이 LLM 추론 서비스 최적화 방법들을 다음과 같이 분류하여 설명한다.
<center><img src="/assets/img/taming_the_titans/1.png" style="zoom: 70%;" /></center>

<br>

- 인스턴스 수준 최적화 (Instance-Level optimization) :
  - **Model placement:** 단일 GPU 메모리 부족 시 여러 장치에 모델 매개변수 분산
  - **Request scheduling 및 Decoding length prediction:** 짧은 요청을 우선 처리하여 전체 지연 시간 감소 (동적 일괄 처리 중 요청 삽입/제거 관리 포함)
  - **KV cache:** 중복 계산 완화 (But, 저장 효율성, 재사용 전략, 압축 문제가 여전히 있음)
  - **Disaggregated architecture:** 프리필(prefill) 단계와 디코딩 단계를 분리하여 각 단계별 최적화 진행
- 클러스터 수준 최적화 (Cluster-Level optimization):
  - **Deployment strategies:** 비용 효율적인 GPU 클러스터 구성 (이기종 하드웨어 활용 포함), 서비스 지향 클러스터 스케줄링
  - **Load Balancing:** 분산된 인스턴스 간 리소스 부족 또는 과부하 방지
  - **Cloud-based LLM Serving:** 로컬 인프라 부족 시 클라우드 활용, 동적 LLM 서비스 수요 충족을 위한 확장
- 새로운 시나리오 (Emerging Scenarios):
  - **Long Context processing**
  - **특정 기술 및 모듈:** 검색 증강 생성(RAG, Retrieval-Augmented Generation), Mixture-of-Experts(MoE), LoRA(Low-Rank Adaptation), 추측 디코딩(Speculative Decoding), Augmented LLMs, Test-Time Reasoning 등 진화하는 요구 사항에 대한 적응성 필요
- 중요한 기타 영역 (Miscellaneous Areas):
  - 하드웨어, 개인 정보 보호, 시뮬레이터, 공정성, 에너지 효율성 등 분야의 전체적인 발전을 위한 기타 영역

<br>

# 2. LLM Inference Serving in Instance
이 섹션에서는 (그림 2와 같이) 배포, 스케줄링, 디코딩 길이 예측, 메모리 관리 및 혁신적인 아키텍처를 다룬다.

<br>

## **2.1 Model Placement**
LLM의 방대한 파라미터 수는 단일 GPU의 용량을 초과하는 경우가 많아, 모델을 여러 GPU에 분산하거나 CPU 메모리 혹은 스토리지로 일부를 옮기는(오프로딩) 전략이 필수적이다. 주요 접근 방식은 다음과 같다.

1. 모델 병렬 처리 (Model Parallelism): 모델 자체를 여러 GPU에 나누어 처리
   - 파이프라인 병렬 처리 (Pipeline Parallelism):
     - **개념:** 모델의 레이어를 여러 GPU에 걸쳐 순차적으로 분산한다. 이를 통해 데이터가 파이프라인처럼 각 GPU를 거치며 처리되어, 순차적인 데이터의 동시 처리가 가능해지고 훈련/추론 속도가 향상된다.
     - **예시:** GPipe, PipeDream, Megatron-LM (Narayanan et al.,2021)
   - 텐서 병렬 처리 (Tensor Parallelism):
     - **개념:** 개별 연산 또는 레이어를 여러 장치에서 병렬로 계산되는 더 작은 하위 텐서로 분할하여 계산 효율성을 높이고 더 큰 모델 크기를 가능하게 한다.
     - **예시:** Megatron-LM (Shoeybi et al., 2020)
   - 기타 특화된 병렬 처리 기술:
     - **시퀀셜 병렬 처리 (Sequential Parallelism):** 긴 컨텍스트 처리 시, LayerNorm 및 Dropout 같은 활성화 함수를 시퀀스 차원을 따라 분할한다.
     - **컨텍스트 병렬 처리 (Context Parallelism):** 모든 레이어를 시퀀스 차원을 따라 분할하여 시퀀셜 병렬 처리를 확장한다.
     - **전문가 병렬 처리 (Expert Parallelism):** sparse MoE 모델의 경우, 전문가 모듈(sparse MoE 컴포넌트)을 GPU에 할당하여 메모리 사용량을 최적화한다.

2. 오프로딩 (Offloading)
   - **개념:** 모델 가중치의 대부분을 CPU 메모리나 스토리지 장치에 저장하고, 추론 시 필요한 부분만 GPU 메모리로 로드하여 처리
   - 주요 기술:
     - **ZeRO-Offload, DeepSpeed-Inference, FlexGen:** 모델 가중치의 대부분을 메모리 또는 스토리지 장치에 저장하고 필요시 GPU로 로드하는 일반적인 오프로딩 기법이다.
     - **PowerInfer:** GPU-CPU 하이브리드 엔진을 사용하여, 자주 사용되는 '핫 뉴런'은 GPU에 미리 로드하여 속도를 높이고, 덜 사용되는 '콜드 뉴런'은 CPU에서 계산하여 GPU 메모리 요구 사항과 데이터 전송량을 줄인다.
     - **TwinPilots:** GPU와 CPU를 트윈 컴퓨팅 엔진으로 통합하고, GPU 및 CPU 메모리를 모두 포함하는 계층적 메모리 아키텍처를 사용하는 새로운 컴퓨팅 패러다임을 제안한다 (비대칭 다중 처리 프레임워크 내).
     - **Park and Egger (2024):** 동적이고 세밀하게 조정된 워크로드 할당을 통해 GPU와 CPU 간의 효율적인 리소스 활용 기술을 제안한다.

<br>

## **2.2 Request Scheduling**
요청 스케줄링은 LLM 추론 서비스의 대기 시간 최적화에 직접적인 영향을 미치며, 요청 간(inter-request) 스케줄링과 요청 내(intra-request) 스케줄링 관점에서 관련 알고리즘을 살펴본다.

1. 요청 간 스케줄링 (Inter-Request Scheduling):
   - **목표:** 여러 요청이 몰릴 때 실행 순서를 정하고 배치 내 요청의 우선순위를 결정
   - **FCFS(First-Come-First-Served):** 대부분의 현재 LLM 솔루션은 선입선출(FCFS) 방식을 사용하나, 이는 먼저 온 긴 요청 때문에 짧은 요청의 처리가 지연되어 전체 대기 시간을 증가시키는 문제(Head-of-Line blocking)가 있다.
   - **최적화 방향:** 짧은 요청에 우선순위를 부여하여 해당 요청의 서비스 수준 목표(SLO) 달성을 목표로 함
   - 주요 기술 및 시스템 (디코딩 길이 예측 활용):
     - **FastServe:** Skip-Join MLFQ(Multi-Level Feedback Queue) 스케줄러를 사용. 우선순위 높은 요청을 먼저 처리하고, 오래 대기한 요청은 우선순위를 높이며, 긴 작업을 선점하여 짧은 요청을 가속화한다.
     - **Fu et al. (2024c):** 예측된 디코딩 시간이 짧은 요청에 우선순위를 부여하여 최단 작업 우선 스케줄링(SJF, Shortest Job First) 방식을 근사한다.
     - **Shahout et al. (2024b):** 남은 디코딩 길이를 동적으로 예측하고 긴 요청의 과도한 선점을 방지하기 위해 선점 비율을 도입하여 최단 잔여 시간 우선 스케줄링(SRTF, Shortest Remaining Time First) 방식을 개선한다. (단, 디코딩 길이 예측 모델 호출 오버헤드가 발생할 수 있음)
     - **Prophet:** PD(Prefill-Decoding) 분리 아키텍처를 활용, 프리필 단계에서는 SJF를, 디코딩 단계에서는 Skip-Join MLFQ를 적용한다.
     - **INFERMAX:** 추론 비용 모델 기반의 전략적 선점이 비선점 방식보다 GPU 비용을 줄일 수 있음을 보여준다.
   - 대조적 접근:
     - **BatchLLM:** 개별 요청 순서보다는 전역 공유(global sharing)를 통해 요청 처리 자체를 우선시한다.
2. 요청 내 스케줄링 (Intra-Request Scheduling):
   - **목표:** 동시에 처리되는 요청 배치의 스케줄링을 통해 요청 도착, 완료 시간, 출력 길이의 가변성을 해결하여 병렬 디코딩 효율성을 향상시킨다.
   - 주요 기술 및 시스템:
     - **Orca:** 반복 수준 스케줄링(iteration-level scheduling)을 도입하여 반복마다 요청을 동적으로 추가/제거할 수 있게 하여 유연성을 높인다.
     - **Dynamic SplitFuse & chunked-prefills:** 프리필 단계를 더 작은 세그먼트로 나누고 이를 디코딩 단계와 병합하여 긴 프롬프트로 인한 지연을 줄이고 프리필 중 디코딩 중단을 방지한다.
     - **SCLS (slice-level scheduling):** 최대 생성 길이를 고정된 크기의 슬라이스(slice)로 나누어 순차적으로 처리함으로써 서비스 시간과 메모리 사용량을 정밀하게 제어한다.

<br>

## **2.3 디코딩 길이 예측 (Decoding Length Prediction)**
생성될 출력 길이의 불확실성이 요청 스케줄링을 복잡하게 만드는 문제를 해결하기 위해 최근 연구들이 세 가지 주요 접근 방식으로 진행되고 있다.

1. 정확한 길이 예측 (Exact Length Prediction):
   - **목표:** 생성될 정확한 토큰 수를 예측합니다.
   - **방법:** BERT 임베딩과 랜덤 포레스트 회귀 모델, 작은 OPT 모델(Hu et al.), 또는 더 간단한 회귀 모델(Qiu et al.) 등을 사용한다.
2. 범위 기반 분류 (Range-Based Classification):
   - **목표:** 요청을 미리 정의된 길이 구간(bin)으로 분류한다.
   - **방법:** Supervised fine-tuning을 하거나, DistilBERT 분류기, short/medium/long 구간 분류, BERT의 CLS 토큰을 FFN으로 처리하여 백분위수 그룹으로 분류(µ-Serve), 토큰 임베딩에 대한 경량 분류기 등을 사용한다.
3. 상대적 순위 예측 (Relative Ranking Prediction):
   - **목표:** 요청들 간의 상대적인 길이 순서를 예측한다.
   - **방법:** 회귀, 분류, 쌍별(pairwise) 비교 방법, 또는 입력 요청만으로 배치 내 상대적 관계 예측을 통해 견고성을 높이고 과적합을 줄인다.
   - **특징:** 배치 내 요청 순서 결정에만 필요하므로 직관적이지만, 일부 요청이 다음 배치로 넘어가면 순위를 다시 계산해야 하는 오버헤드가 발생할 수 있다. (다른 두 방법은 이 문제가 없음)

<br>

## **2.4 KV Cache Optimization**

KV 캐시는 어텐션 메커니즘에서 이전에 계산된 Key와 Value 값을 저장하여 추론 시간 복잡도를 줄이는 데 핵심적이지만, 메모리 관리, 계산 재사용, 압축 효율성에서 여러 문제를 야기합니다.

1. 메모리 관리 (Memory Management):
   - 무손실 저장 기술 (Lossless Storage Techniques):
     - **PagedAttention & vLLM:** OS에서 영감을 받은 페이징 기법을 도입하여 메모리 단편화를 해결하고 공간 낭비를 최소화한다.
     - **DistAttention:** 긴 컨텍스트 처리를 위해 분산된 KV 캐시 처리를 제안한다.
     - **FastDecode:** 분산 처리를 통해 캐시를 CPU 메모리로 오프로드한다.
     - **LayerKV:** 계층별로 KV 캐시를 할당하고 오프로드한다.
     - **KunServe:** 다른 인스턴스의 파이프라인 메커니즘을 통해 모델 파라미터를 제거하여 캐시 공간을 확보한다.
     - **SYMPHONY:** 다중 턴(multi-turn) 대화 패턴을 활용하여 캐시를 동적으로 마이그레이션한다.
     - **InstCache:** LLM 기반 명령어 예측을 통해 응답성을 향상시킨다.
   - 근사 방법 (Approximation Methods):
     - **PQCache:** 임베딩 검색에 사용되는 low-overhead Product Quantization을 활용하여 임베딩을 하위 임베딩으로 분할하고 클러스터링을 적용하여 계산 오버헤드를 줄인다.
     - **InfiniGen:** 주요 KV 캐시 항목의 지능적인 프리페칭을 통해 데이터 전송 오버헤드를 줄이고 성능을 향상시키는 동적 캐시 관리 프레임워크이다.

2. 재사용 전략 (Reuse Strategies):
   - 무손실 재사용 (Lossless Reuse):
     - **PagedAttention:** 페이지 수준 관리를 통해 여러 요청 간 캐시 공유를 가능하게 한다.
     - **Radix tree-based system:** 동적 노드 삭제를 통해 전역 prefix 공유를 구현한다.
     - **CachedAttention:** 턴(turn) 간 캐시 재사용을 통해 대화에서 중복 계산을 최소화한다.
   - 의미론적 인식 재사용 (Semantic-aware Reuse):
     - **GPTCache:** 의미론적 유사성을 사용하여 LLM 출력을 캐싱하고 재사용한다.
     - **SCALM:** 쿼리를 클러스터링하여 의미 있는 의미적 패턴을 찾아낸다.
     - 무손실 전략은 법률, 의료, 코드 생성 등 정확한 템플릿 입력에 이상적이며, 의미론적 인식 전략은 일반적인 대화에 더 적합하다.

3. 압축 기술 (Compression Techniques):
    추론 성능에 미치는 영향을 최소화하면서 메모리 사용량을 줄이기 위해 텐서 양자화 및 컴팩트 표현과 같은 가중치 및 캐시 압축 기술을 사용한다.

   - 양자화 기반 압축 (Quantization-based Compression):
      높은 비트 정밀도에서 낮은 비트 정밀도로 전환하여 메모리를 줄인다.
     - **FlexGen:** 그룹별 양자화(Group-wise Quantization)를 사용하여 추가 I/O 비용 없이 KV 캐시를 4비트로 압축한다.
     - **Kivi:** 캐시에 대한 채널별/토큰별 양자화를 제안한다.
     - **MiniCache:** 인접 레이어 간 KV 캐시 상태의 높은 유사성을 활용하여 레이어 간 캐시를 압축한다.
     - **AWQ:** 중요하지 않은 가중치를 양자화하면 양자화 손실이 줄어든다는 점을 활용한다.
     - **Atom:** 혼합 정밀도, 세분화된 그룹/동적 활성화/캐시 양자화를 사용한다.
     - **QServe:** 알고리즘-시스템 공동 설계를 통해 LLM을 W4A8KV4 정밀도로 양자화하여 GPU 배포 효율성을 향상시킨다.

   - 컴팩트 인코딩 아키텍처 (Compact Encoding Architectures):
      기존의 큰 행렬 대신 더 작은 행렬 표현을 사용한다.
     - **CacheGen:** 사용자 정의 텐서 인코더를 사용하여 KV 캐시를 컴팩트한 bitstream으로 압축하여 디코딩 오버헤드를 최소화하면서 대역폭을 절약한다.

<br>

## **2.5 PD Disaggregation**
프리필/디코딩(PD) 분리는 연산 집약적인 컨텍스트 인코딩과 메모리 집약적인 토큰 생성 단계를 서로 다른 최적화 환경으로 분리하여 LLM 추론의 연산 불균형 문제를 해결하는 접근 방식이다.

- 주요 기술:
  - **DistServe:** 각 단계(프리필, 디코딩)에 대한 리소스 할당 및 병렬 처리를 최적화하고, 대역폭 기반의 전략적 배치를 통해 통신 오버헤드를 최소화한다.
  - **Splitwise:** 비용, 처리량, 전력 최적화를 위해 동종 및 이기종 장치 설계를 탐색한다.
  - **DéjàVu:** 마이크로배치 스와핑 및 상태 복제를 통해 양방향 지연 시간, GPU 과잉 프로비저닝, 느린 복구로 인해 발생하는 파이프라인 버블(유휴 상태)을 해결한다.
  - **Mooncake:** KV 캐시 중심의 분산 아키텍처를 사용하여 유휴 CPU, DRAM, SSD 리소스를 분산된 KV 캐시 스토리지에 활용하고, 높은 부하 시에는 조기 거부(early rejection)를 통해 낭비를 줄인다.
  - **TetriInfer:** 디코딩 핫스팟(특정 부분에 부하 집중)을 피하기 위해 리소스 예측을 사용하는 2단계 스케줄링 알고리즘을 사용한다.
  - **P/DServe:** 세분화된 프리필/디코딩 구성, 동적 조정, 온디맨드 요청 할당, 효율적인 캐시 전송을 통해 LLM 배포 문제를 해결한다.