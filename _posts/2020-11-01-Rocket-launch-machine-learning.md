---
layout: post
title: "머신 러닝을 통한 로켓 발사 예측"
excerpt: "Python과 Anaconda를 활용한 데이터 분석 활용"
tags: 
  - Python
  - Anaconda
  - Data science
  - Machine Learning
---
## 머신 러닝을 통한 로켓 발사 예측
머신 러닝을 통한 로켓 발사 예측을 하기 위해서는 우선 파이썬(Python)과 아나콘다(Anaconda)가 설치되어 있어야 합니다. 이 둘은 데이터 과학을 쉽고 편리하게 처리하기 위한 필수 요소들 중 하나입니다.

>'아나콘다(Anaconda)'는 여러가지 수학 및 과학 계산(데이터 과학, 기계 학습 애플리케이션, 대규모 데이터 처리, 예측 분석 등)을 포함하고 있는 패키지로 머신러닝(Machine learning)이나 데이터 분석(Data analysis)를 수월하게 처리하고자 할 때 많이 사용됩니다.

### 아나콘다 다운로드 및 기초 환경 구성
https://www.anaconda.com/products/individual

자신의 컴퓨터 환경에 맞는 아나콘다를 설치한 다음 Anaconda Prompt를 실행해줍니다.
<figure class="half">
    <img src="/images/RocketLaunch/anaconda1.jpg">
    <img src="/images/RocketLaunch/anaconda2.jpg">
    <figcaption>Anaconda Prompt를 찾아서 실행</figcaption>
</figure>

다음과 같이 새로운 아나콘다 환경을 설치 및 구성

```bash
# conda create -n myenv: 이름 'myenv'의 새로운 conda 가상환경 생성
conda create -n myenv python=3.7 pandas numpy jupyter seaborn scikit-learn pydotplus
```
* `Pandas` : 데이터를 구조화된 형식으로 가공 및 분석할 수 있도록 자료구조를 제공하는 패키지
* `Numpy` : 행렬이나 대규모 다차원 배열을 쉽게 처리 할 수 있도록 지원하여 고성능 계산이나 데이터 분석에 유용한 패키지
* `Jupyter` : 라이브 코드, 등식, 시각화와 설명을 위한 텍스트 등을 포함한 문서를 만들고 공유가 가능하며 머신러닝이나 데이터분석 용도로 많이 사용하는 오픈소스 소프트웨어
* `Seaborn` : Matplotlib 기반의 고급 인터페이스를 제공하는 Python 데이터 시각화 라이브러리
* `Scikit-learn` : 회귀(Regression), 분류(Classification), 군집화(clustering), 의사결정 트리(Decision tree) 등의 다양한 머신러닝 알고리즘을 적용할 수 있는 함수들을 제공하는 머신러닝 라이브러리
* `Pydotplus` : 그래프를 생성하는 graphviz의 dot 언어를 파이썬 인터페이스로 제공하는 모듈

```bash
# myenv 환경 실행
conda activate myenv
```

```bash
# AzureML-SDK 설치 및 업그레이드
pip install --upgrade azureml-sdk
```
* `AzureML-SDK` :  Azure Machine Learning 작업 영역에서 기계 학습 및 딥 러닝 모델을 빌드할 수 있게 해줍니다.

```bash
# Excel 파일을 읽을 수 있는 라이브러리 설치
pip install xlrd
```

Visual Studio Code 마켓플레이스에서 Python과 Azure Machine Learning 설치
<figure class="half">
    <img src="/images/RocketLaunch/python.jpg">
    <img src="/images/RocketLaunch/Azure.jpg">
</figure>

<figure>
    <img src="/images/RocketLaunch/same.jpg">
    <figcaption>오른쪽 위의 Jupyter 커널과 왼쪽 아래의 Python 인터프리터가 모두 같은 버전의 <br> Anaconda 환경을 사용하고 있어야 합니다.(주황색으로 표시)</figcaption>
</figure>

### 라이브러리 가져오기, 데이터 정리 및 조작
라이브러리를 가져와 날씨 데이터를 가져와 정리하고, 기계 학습 모델을 만들고 테스트할 수 있도록 구성합니다.

```python
# Pandas library is used for handling tabular data
import pandas as pd
# NumPy is used for handling numerical series operations (addition, multiplication, and ...)
import numpy as np
# Sklearn library contains all the machine learning packages we need to digest and extract patterns from the data
from sklearn import linear_model, model_selection, metrics
from sklearn.model_selection import train_test_split

# Machine learning libraries used to build a decision tree
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree

# Sklearn's preprocessing library is used for processing and cleaning the data 
from sklearn import preprocessing

# for visualizing the tree
import pydotplus
from IPython.display import Image 
```
