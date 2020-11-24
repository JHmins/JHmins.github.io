---
layout: post
title: "Azure ML을 이용한 예측 모델링 (코드 X)"
excerpt: "자동 기계학습, 회귀 모델, 분류 모델, 클러스터링 모델 공부"
tags: 
  - Azure ML
  - Automated Machine Learning
  - Regression
  - Classification
  - Clustering
---
### 자동화된 기계 학습
&nbsp; 우선 Azure에서 Machine Learning 작업 영역을 생성한 다음, 컴퓨팅 목록에서 `컴퓨팅 인스턴스`<ins>(데이터 및 모델을 작업하는 데 사용할 수 있는 개발 워크스테이션)</ins>와 `컴퓨팅 클러스트`<ins>(실험 코드의 주문형 처리를 지원하는 확장 가능한 가상 머신 클러스터)</ins>를 만듭니다. 

* 가상 머신 유형 : CPU
* 가상 머신 크기 : Standard_DS11_v2
* 나머지는 자신이 하고 싶은데로 ㅎㅎ

<figure class="half">
    <a href="/images/AzureML_Nocode/instance.jpg"><img src="/images/AzureML_Nocode/instance.jpg"></a>
    <a href="/images/AzureML_Nocode/cluster.jpg"><img src="/images/AzureML_Nocode/cluster.jpg"></a>
    <figcaption></figcaption>
</figure>

### 데이터 세트 만들기
&nbsp; 다음 Azure Machine Learning Studio에서 데이터 세트에서 새로운 데이터 세트를 만들어줍니다. 제가 만들 모델은 지정된 날짜에 예상되는 자전거의 대여수를 예측하는 모델로, 기본 정보는 https://aka.ms/bike-rentals 웹 URL에서 가져와줍니다.\
&nbsp; 데이터 세트 형식은 나중에라도 보기 쉽도록 '표 형식'으로, 파일형식은 '쉼표마다 분리해서 구분하는 형식'으로, 열 머리글은 '첫 번째 파일의 헤더'(ID, day, year, season, ...)를 사용해 만들겠습니다.

<figure class="half">
    <a href="/images/AzureML_Nocode/dataset.jpg"><img src="/images/AzureML_Nocode/dataset.jpg"></a>
    <a href="/images/AzureML_Nocode/dataset2.jpg"><img src="/images/AzureML_Nocode/dataset2.jpg"></a>
    <figcaption></figcaption>
</figure>

### 자동화된 Machine Learning 실행
&nbsp; 데이터 세트를 다 만들었으면 이제는 자동화된 ML에서 새 자동화된 ML을 만들어줍니다.  
