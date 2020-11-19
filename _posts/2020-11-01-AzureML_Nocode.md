---
layout: post
title: "Azure Machine Learning을 이용한 예측 모델링(코드 X)"
excerpt: "자동 기계학습, 회귀 모델, 분류 모델, 클러스터링 모델 공부"
tags: 
  - Azure ML
  - Automated Machine Learning
  - Regression
  - Classification
  - Clustering
---
## 자동화된 기계 학습
&nbsp; 우선 Azure에서 Machine Learning 작업 영역을 생성한 다음, 컴퓨팅 목록에서 `컴퓨팅 인스턴스`<ins>(데이터 및 모델을 작업하는 데 사용할 수 있는 개발 워크스테이션)</ins>와 `컴퓨팅 클러스트`<ins>(실험 코드의 주문형 처리를 지원하는 확장 가능한 가상 머신 클러스터)</ins>를 만들어줍니다. 

* 가상 머신 유형 : CPU
* 가상 머신 크기 : Standard_DS11_v2
* 나머지는 자신이 하고 싶은데로 ㅎㅎ

<figure class="half">
    <a href="/images/AzureML_Nocode/instance.jpg"><img src="/images/AzureML_Nocode/instance.jpg"></a>
    <a href="/images/AzureML_Nocode/cluster.jpg"><img src="/images/AzureML_Nocode/cluster.jpg"></a>
    <figcaption></figcaption>
</figure>