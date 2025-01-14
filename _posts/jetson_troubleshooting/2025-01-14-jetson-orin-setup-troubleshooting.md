---
title: "Jetson AGX Orin 64GB 환경 구축 A to Z"
date: 2025-01-14 09:58:47 +09:00
modified: 2025-01-14 09:58:47 +09:00
tags: 
    - Jetson
    - On-Device
    - Embodied AI
usemathjax: true
---


지금 다니고 있는 회사에서 Embodied AI 프로젝트를 위해 Jetson AGX Orin 64GB 장비를 도입하게 되었다. 기존에는 주로 RTX4090, A100 등 Ada Lovelace 및 Ampere 아키텍처 기반의 워크스테이션 환경에서 다양한 AI 태스크를 수행해왔다. 하지만 이번에는 Tegra 프로세서 기반의 Jetson 환경에서 작업을 진행하게 되면서 여러 시행착오와 어려움을 겪었다. 이 글에서는 그 과정과 해결책을 공유하고자 한다.

<br>

# 1. 초기 환경 설정 실패: 아키텍처 호환성 문제

처음에는 기존 워크스테이션 환경에서 사용하던 vLLM, TensorRT-LLM, MLC-LLM 등의 추론 라이브러리를 Jetson 환경에서 그대로 사용하려 하였다. 그래서 가상환경을 만들고 `pip install` 명령어를 통해 설치를 시도했지만, 결과는 실패였다.


<figure align="center">
<center><img src="/assets/img/jetson-orin-setup-troubleshooting/0.png"/></center>
<figcaption>jetson platform service diagrams</figcaption>
</figure>

<br>

**문제 원인:**  
기존 워크스테이션 환경에서는 Ubuntu x86_64 아키텍처를 사용했지만, Jetson은 NVIDIA Tegra 프로세서를 기반으로 하는 **Jetson Linux (L4T)** 운영체제를 사용한다. 이는 ARM 아키텍처 (aarch64) 기반이며, vLLM, TensorRT-LLM, MLC-LLM 등의 라이브러리는 Ubuntu의 x86_64 아키텍처에 맞춰 컴파일되고 특정 라이브러리 및 패키지에 의존하기 때문에 Jetson 환경에서 호환성 문제가 발생한다. 따라서, 해당 라이브러리들이 Jetson 환경에 맞게 재컴파일되지 않았거나 필요한 의존성이 충족되지 않으면 설치 및 실행에 실패하게 된다.

<br>

<details> <summary><b>Tegra 프로세서와 L4T (Linux for Tegra)</b></summary>
<div markdown="1">
  <p> <b>Tegra 프로세서란:</b> NVIDIA Tegra는 NVIDIA에서 개발한 시스템 온 칩(System on a Chip, SoC) 시리즈이다. 주로 모바일 및 임베디드 기기용으로 설계되었으며, 강력한 GPU 성능과 에너지 효율성을 결합한 것이 특징이다. Tegra 프로세서는 CPU, GPU, 메모리 컨트롤러, 비디오 처리 장치, 이미지 신호 프로세서(ISP) 등을 하나의 칩에 통합하여 다양한 기능을 제공한다. </p><br>
  <p> <b>L4T(Linux for Tegra):</b> Jetson Linux (L4T)는 NVIDIA Tegra 프로세서를 기반으로 하는 Jetson 임베디드 시스템을 위해 특별히 제작된 Linux 배포판이다. 이 운영체제는 NVIDIA에서 직접 최적화한 Linux 커널을 사용하며, Jetson 하드웨어의 GPU, 멀티미디어, 카메라 인터페이스 등의 기능을 최대한 활용할 수 있도록 특수 드라이버와 라이브러리를 제공한다. JetPack SDK를 통해 운영체제 설치 및 개발 도구를 쉽게 관리할 수 있으며, CUDA, cuDNN, TensorRT 등의 NVIDIA AI 라이브러리를 포함하여 임베디드 AI 및 비전 시스템 개발에 최적화되어 있다. </p> 
</div>
</details>

<br>

# 2. Jetson 환경 점검 및 재구성: `jetson-containers` 기반 환경 구축

그러다 우연히 [jetson-ai-lab](https://www.jetson-ai-lab.com/)에서 소개하는 다양한 LLM 관련 튜토리얼을 참고하면서 모두 [jetson-containers](https://github.com/dusty-nv/jetson-containers) 라는 사전 빌드된 패키지를 사용하는 것을 알 수 있었다. jetson-containers는 Jetson을 위한 최신 AI/ML 패키지를 제공하는 모듈식 컨테이너 빌드 시스템으로, LLM뿐만 아니라 다양한 JetPack/L4T용 패키지를 제공하였다.


<figure align="center">
<center><img src="/assets/img/jetson-orin-setup-troubleshooting/1.png"/></center>
<figcaption>jetson-containers가 지원하는 패키지 목록</figcaption>
</figure>


하지만, 우리가 사용하려는 라이브러리(ex. vLLM, TensorRT-LLM 등)를 설치하기 위해서는 특정 Jetpack 버전 (= L4T 버전) 이상이 필요하였다. 따라서, 우리는 다시 한번 환경을 점검하기로 하였다. 즉, Ubuntu 환경, Jetpack 버전, CUDA 버전, 그리고 torch 버전을 확인하고, jetson-container를 설치하여 필요한 라이브러리를 실행하기로 하였다.

<br>

## 2.1. 현재 환경 확인

**Ubuntu 버전 확인:**

```
$ lsb_release -a
No LSB modules are available.
Distributor ID:	Ubuntu
Description:	Ubuntu 20.04.5 LTS
Release:	20.04
Codename:	focal
```

위 명령어 실행 결과에서 현재 Ubuntu 버전이 20.04 LTS임을 확인할 수 있다.

<br>

**Jetpack 버전 확인:**

```
$ cat /etc/nv_tegra_release

# R35 (release), REVISION: 3.1, GCID: 32827747, BOARD: t186ref, EABI: aarch64, DATE: Sun Mar 19 15:19:21 UTC 2023
```

위 명령어 실행 결과에서 **R35**와 **REVISION 3.1**에 주목해야 한다. [NVIDIA Jetpack Archive](https://developer.nvidia.com/embedded/jetpack-archive)에서 확인해본 결과, 해당 값은 **JetPack 5.1.1** ([L4T 35.3.1]) 버전임을 알 수 있다. 또한 CUDA와 Torch 버전 또한 위 JetPack 버전에 맞게 설치되어야 한다.

<br>

<center><img src="/assets/img/jetson-orin-setup-troubleshooting/2.png"/></center>

<br>

## 2.2. 업그레이드 목표 설정

우리가 사용하려는 라이브러리들은 CUDA 12.2 이상 또는 L4T 36.x.x 이상, 즉 JetPack 6.0 버전 이상을 요구한다. [JetPack 6.0.0 document](https://developer.nvidia.com/embedded/jetpack-sdk-60)를 확인하면 Ubuntu 버전 또한 22.04로 업그레이드해야 한다는 것을 알 수 있다. 따라서, 우리의 업그레이드 목표는 다음과 같다.

- Ubuntu 20.04 LTS → **Ubuntu 22.04 LTS**
- JetPack 5.1.1 → **JetPack 6.0** (JetPack 업그레이드 시 아래 항목들은 자동으로 업그레이드됨)
  - CUDA 11.4 → **CUDA 12.2**
  - Python 3.8.10 → **Python 3.10**

<br>

# 3. Jetson 환경 업그레이드 과정

Jetson 환경 업그레이드 과정은 Jetpack 버전과 Ubuntu 버전에 따라 달라진다.

**업그레이드 방법:**

1. Ubuntu 버전과 Jetpack 버전 모두 안 맞는 경우 (본 문서에서 채택):
   - Ubuntu 및 Jetpack을 모두 업그레이드한다.
   - Ubuntu 22.04 버전의 호스트 PC와 Jetson을 연결하여 호스트 PC에 SDK Manager를 다운로드하고 업그레이드해야 한다.
2. Ubuntu 버전은 맞으나 Jetpack 버전이 안 맞는 경우:
   - 하드웨어 호환성 등의 이유로 Jetson 모듈을 사용하는 캐리어 보드 제조사(공급업체)에서 제공하는 커스터마이징된 BSP(Board Support Package) 파일을 받아야 한다. 
   - 이후 CUDA 업그레이드를 진행한다.

<br>

## 3.1. 방법 1: SDK Manager를 이용한 업그레이드 (Ubuntu 및 Jetpack 모두 업그레이드)

Jetson 환경을 업그레이드하는 방법은 여러 가지가 있지만, 여러 시행착오를 겪어본 결과 Ubuntu 22.04 환경의 호스트 PC를 사용하여 [**NVIDIA SDK Manager**](https://developer.nvidia.com/sdk-manager)를 통해 Jetson 장치를 업그레이드하는 것이 가장 쉽고 안정적인 방법이다. 주의할 점은 Jetson 장치와 호스트 PC를 연결할 때는 반드시 데이터 전송이 가능한 micro USB 5핀 케이블을 사용해야 한다.

SDK Manager를 이용한 설치 과정에 대한 자세한 내용은 NVIDIA 공식 문서[https://docs.nvidia.com/sdk-manager/install-with-sdkm-jetson/index.html](https://docs.nvidia.com/sdk-manager/install-with-sdkm-jetson/index.html)를 참조하면 된다.

<br>

## 3.2. 방법 2: BSP(Board Support Package)를 이용한 업그레이드 (Jetpack만 업그레이드)

Jetson 모듈을 탑재한 캐리어 보드 제조사(공급업체)에서 커스터마이징된 BSP 파일을 제공하는 경우, 해당 BSP 파일을 사용하여 Jetpack을 업그레이드할 수 있다. 이 방법은 Jetson 모듈의 하드웨어 호환성, 특정 기능 지원, 최적화 등을 고려하여 설정되었으므로 안정적인 시스템 운영을 보장한다. BSP 파일은 일반적으로 제조사 웹사이트에서 다운로드할 수 있으며, 제공되는 설치 지침을 따라 업그레이드해야 한다. 여기서도 방법 1과 마찬가지로 Jetson 장치와 호스트 PC를 연결할 때는 반드시 데이터 전송이 가능한 micro USB 5핀 케이블을 사용해야 한다.

<br>

**CUDA 업그레이드:**  
Jetpack을 업그레이드하면 CUDA 업그레이드는 비교적 간단하게 진행된다.

```python
$ sudo apt update
$ sudo apt dist-upgrade
$ sudo reboot
$ sudo apt install nvidia-jetpack
```

<br>

**`nvidia-jetpack` 설치 시 다음과 같은 에러 나는 경우:

```bash
$ sudo apt install nvidia-jetpack
Reading package lists... Done
Building dependency tree
Reading state information... Done
E: Unable to locate package nvidia-jetpack
```

<br>

이는 `/etc/apt/sources.list.d/nvidia-l4t-apt-source.list` 파일에서 릴리즈 파일 설치하는 코드(`deb ...`)가 주석 처리되어 있는 경우 발생한다. 해당 파일을 열어 주석을 해제한 후 다시 `sudo apt install nvidia-jetpack` 명령어를 실행하면 된다.

```bash
$ cat /etc/apt/sources.list.d/nvidia-l4t-apt-source.list
# SPDX-FileCopyrightText: Copyright (c) 2019-2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

deb <https://repo.download.nvidia.com/jetson/common> r35.3 main
deb <https://repo.download.nvidia.com/jetson/t234> r35.3 main
```

(위 코드 예시는 JetPack 5.1.1 버전이며, 6.0은 릴리즈 파일 3개)

<br>

**`nvidia-jetpack` 설치 완료 확인:**

```
$ sudo apt show nvidia-jetpack

Package: nvidia-jetpack
Version: 5.1.1-b56
Priority:standard
...
```

<br>

**`nvidia-jetpack` 설치 후 CUDA 환경 변수 설정:**

```
$ vi ~/.bashrc
# 아래 내용 추가
PATH=/usr/local/cuda/bin:$PATH
LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
$ source ~/.bashrc
```

<br>

**`nvcc --version` 명령어를 통해 CUDA 버전 확인:**

```
$ nvcc --version
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2022 NVIDIA Corporation
Build on Sun_Oct_23_22:16:07_PDT_2022
Cuda compilation tools, release 11.4, V11.4.315
Build cuda_12.2.r12.2/compiler.31964100_0
```

<br>

# 4. PyTorch 설치 및 확인

Jetson 환경에 PyTorch를 설치하는 과정은 JetPack 버전에 맞는 PyTorch whl 파일을 다운로드하여 설치하고, 가상환경에서 앞서 설치한 CUDA가 정상적으로 작동하는지 확인하는 과정으로 이루어진다.

<br>

## 4.1. PyTorch whl 파일 다운로드

JetPack 버전에 맞는 PyTorch whl 파일을 다운로드해야 한다. [PyTorch for Jetson](https://forums.developer.nvidia.com/t/pytorch-for-jetson/72048) 페이지에서 자신의 JetPack 버전에 맞는 PyTorch whl 파일을 찾아 다운로드한다. 여기서는 JetPack 6.0 (L4T R36.3) 버전을 기준으로 설명한다.

<center><img src="/assets/img/jetson-orin-setup-troubleshooting/3.png"/></center>

위 이미지에서 <u>JetPack 6.0 (L4T R36.2 / R36.3) + CUDA 12.2</u> 버전에 맞는 whl 파일 정보를 확인할 수 있다.

- torch 2.3 - `torch-2.3.0-cp310-cp310-linux_aarch64.whl`
- torchvision 0.18 - `torchvision-0.18.0a0+6043bc2-cp310-cp310-linux_aarch64.whl`
- torchaudio 2.3 - `torchaudio-2.3.0+952ea74-cp310-cp310-linux_aarch64.whl`

<br>

## 4.2. 가상환경 생성 및 활성화

먼저, `python3 -m venv` 명령어를 사용하여 PyTorch를 설치할 가상환경을 생성하고 활성화한다.

```
$ python3 -m venv venv  # 가상환경 이름: venv
$ source venv/bin/activate
```

<br>

## 4.3. PyTorch 설치 및 CUDA 확인

다운로드한 whl 파일을 pip install 명령어로 설치한다. JetPack 6.0 (L4T R36.3)에 맞는 `torch-2.3.0-cp310-cp310-linux_aarch64.whl` 파일을 예시로 사용하였다. 

```
$ pip install torch-2.3.0-cp310-cp310-linux_aarch64.whl
```

<br>

PyTorch 설치 후, 다음 파이썬 코드를 실행하여 PyTorch 버전과 CUDA 사용 가능 여부를 확인한다. 

```
$ python -c '
import torch
print(torch.__version__)
print(torch.cuda.is_available())
print(torch.empty((1, 2), device=torch.device("cuda")))
'
```

<br>

정상적으로 설치되었다면, 다음과 유사한 결과가 출력된다.

```
2.3.0a0+41361538.nv23.06
True
tensor([[0., 0.]], device='cuda:0'))
```

<br>

위 결과에서 PyTorch 버전(2.3.0), CUDA 사용 가능 여부 (True), CUDA 장치에서 생성된 텐서가 출력되는 것을 볼 수 있으며, 이를 통해 PyTorch와 CUDA가 Jetson 환경에서 정상적으로 작동하는 것을 확인할 수 있다.

<br>

## 4.4. GPU 메모리 확인 방법

Jetson 장치에서 GPU 메모리 사용량을 확인하는 방법은 데스크톱 환경과는 다소 차이가 있다. 일반적으로 데스크톱 환경에서 사용되는 `nvidia-smi` 명령어는 Jetson의 Tegra 기반 프로세서에서는 사용할 수 없다. 대신 `jetson-stats` 또는 `tegrastats`와 같은 도구를 사용하여 GPU 사용량을 확인할 수 있다.

<br>

### 4.4.1. jetson-stats

[jetson-stats](https://github.com/rbonghi/jetson_stats)는 Jetson 시리즈를 모니터링하고 제어하기 위한 패키지이다. CPU, GPU, Memory, Engine 등 다양한 정보를 한눈에 파악할 수 있다.

```
$ sudo pip3 install -U jetson-stats
$ jtop
```

<br>

### 4.4.2 tegrastats

tegrastats는 Jetson 장치의 시스템 상태를 모니터링하는 또 다른 도구이다. CPU, GPU 사용량, 전력 소비량, 온도 등 다양한 정보를 텍스트 기반으로 표시한다.

```
$ sudo tegrastats
```

<br>

# 5. jetson-containers 활용 및 사용자 정의 Docker 환경 구축

Docker는 이미 설치되었다고 가정하고, jetson-containers를 활용하여 개발 환경을 구축하는 방법을 소개한다. Docker 설치에 대한 자세한 내용은 [공식 문서](https://docs.docker.com/engine/install/ubuntu/)를 참조하면 된다.

<br>

## 5.1. jetson-containers 설치 및 기본 사용법

jetson-containers는 NVIDIA Jetson 장치에서 AI/ML 워크로드를 실행하기 위한 컨테이너 빌드 및 관리 시스템이다. 이 도구를 사용하면 다양한 라이브러리와 환경을 쉽게 컨테이너화하여 사용할 수 있다.

<br>

**설치 방법:**

```
# jetson-containers 클론 및 설치 스크립트 실행
$ git clone https://github.com/dusty-nv/jetson-containers
$ bash jetson-containers/install.sh
```

<br>

**기본 사용법:**

만약 vLLM을 사용하고 싶다면, 다음과 같이 미리 빌드된 컨테이너 이미지를 실행할 수 있다.

```
$ jetson-containers run dustynv/vllm:0.6.3-r36.4.0
```

위 명령어는 `dustynv/vllm:0.6.3-r36.4.0` 이미지를 실행하여 vLLM 환경을 바로 사용할 수 있도록 해줍니다. 이미지 이름의 r36.4.0은 L4T 버전을 나타낸다. [jetson-containers packages](https://github.com/dusty-nv/jetson-containers/tree/master/packages#packages)에서 사용 가능한 이미지들을 확인할 수 있다.

<br>

## 5.2. 사용자 정의 Docker 환경 구축

만약 LLM 추론만 수행한다면 위와 같이 사전 빌드된 이미지를 사용하는 것으로 충분할 수 있다. 하지만, 필요한 라이브러리를 추가로 설치해야 하는 경우가 많아지면 자체적으로 `docker-compose.yml` 파일과 `Dockerfile`을 커스터마이징하는 것이 좋다. 물론 계속해서 빌드 및 삭제를 거치면서 오류를 해결해 나아가야 하는 어려움이 있었지만, 필요한 라이브러리들을 추가로 설치하는 일이 많아지면 자체적으로 Docker 환경을 구성하는 것이 유용하다.

<br>

# 6. 결론
Jetson AGX Orin 64GB 장비를 Embodied AI 프로젝트에 도입하는 과정은 기존 워크스테이션 환경과는 다른 Tegra 프로세서와 Jetson Linux (L4T) 운영체제에 대한 이해가 필수적임을 확인하는 여정이었다. 초기에는 아키텍처 및 운영체제 호환성 문제로 인해 어려움을 겪었지만, Jetson 환경에 대한 이해와 문제 해결을 통해 성공적인 개발 환경을 구축할 수 있었다.  

이러한 시행착오와 경험을 통해 Jetson 프로세서에서 AI/ML 개발을 진행할 때 다음과 같은 점을 고려해야 함을 알 수 있었다.
- Jetson 프로세서의 특성 이해 및 관련 문서 충분히 참고
- Ubuntu와 JetPack (L4T) 버전 간 호환성 유지 및 JetPack 버전에 따른 PyTorch, CUDA 설정 확인
- 라이브러리 및 패키지 설치 시 호환성 면밀히 확인 및 필요에 따라 jetson-containers 활용
- 사용자 정의 Docker 환경 구축 시 라이브러리 간 호환성 문제 고려
- GPU 사용량을 모니터링하여 시스템 자원을 효율적으로 관리  

이러한 노력과 과정을 통해 우리는 Jetson AGX Orin 64GB 장비에서 Embodied AI 프로젝트를 위한 개발 기반을 어느 정도 마련할 수 있었다. 이후에는 추론 속도의 벽에 부딪히는 또 다른 어려움이 있었는데, 추후 기회가 된다면 이 주제에 대해서도 소개하도록 하겠다. 이 글이 Jetson 환경을 처음 접하는 개발자들에게 도움이 되기를 바란다.

<br>

# Reference

- <https://en.wikipedia.org/wiki/Tegra#Linux>
- <https://developer.nvidia.com/embedded/develop/software>
- <https://www.jetson-ai-lab.com/>

<br>
