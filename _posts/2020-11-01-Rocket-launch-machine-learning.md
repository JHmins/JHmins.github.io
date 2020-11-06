---
layout: post
title: "머신 러닝을 통한 로켓 발사 예측"
excerpt: "Python과 Anaconda를 이용한 데이터 분석 활용"
tags: 
  - Python
  - Anaconda
  - Data science
  - Machine Learning
---
## 머신 러닝을 통한 로켓 발사 예측
머신 러닝을 통한 로켓 발사 예측을 하기 위해서는 우선 파이썬(Python)과 아나콘다(Anaconda)가 설치되어 있어야 합니다. 이 둘은 데이터 과학을 쉽고 편리하게 처리하기 위한 필수 요소들 중 하나입니다.

>'아나콘다(Anaconda)'는 여러가지 수학 및 과학 계산(데이터 과학, 기계 학습 애플리케이션, 대규모 데이터 처리, 예측 분석 등)을 포함하고 있는 패키지로 머신러닝(Machine learning)이나 데이터 분석(Data analysis)를 수월하게 처리하고자 할 때 많이 사용됩니다.

아나콘다 다운로드 및 기초 환경 구성\
https://www.anaconda.com/products/individual

자신의 컴퓨터 환경에 맞는 아나콘다를 설치한 다음 Anaconda Prompt를 실행해줍니다.
<figure class="half">
    <a href="/images/RocketLaunch/anaconda1.jpg"><img src="/images/RocketLaunch/anaconda1.jpg"></a>
    <a href="/images/RocketLaunch/anaconda2.jpg"><img src="/images/RocketLaunch/anaconda2.jpg"></a>
    <figcaption>Anaconda Prompt를 찾아서 실행</figcaption>
</figure>

### 새로운 아나콘다 환경 구성

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
    <a href="/images/RocketLaunch/same.jpg"><img src="/images/RocketLaunch/same.jpg"></a>
    <figcaption>오른쪽 위의 Jupyter 커널과 왼쪽 아래의 Python 인터프리터가 모두 같은 버전의 <br> Anaconda 환경을 사용하고 있어야 합니다.(주황색으로 표시)</figcaption>
</figure>

---

### 라이브러리 가져오기, 데이터 읽기 및 호출
라이브러리를 가져와 날씨 데이터를 가져와 정리하고, 기계 학습 모델을 만들고 테스트할 수 있도록 구성합니다.

```python
# Pandas는 표 데이터를 처리하는 데 사용
import pandas as pd
# NumPy는 수열 연산 작업을 처리하는 데 사용(덧셈, 곱셈 ...)
import numpy as np

# Sklearn 라이브러리는 데이터에서 패턴을 분석하고 추출하는 데 필요한 모든 머신러닝 패키지 포함
from sklearn import linear_model, model_selection, metrics
from sklearn.model_selection import train_test_split

# 의사 결정 트리를 작성하는 데 사용되는 머신 러닝 라이브러리
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree

# 데이터를 처리하고 정리하는 데 사용되는 Sklearn 전처리 라이브러리
from sklearn import preprocessing

# 트리를 시각화하기 위해 사용
import pydotplus
from IPython.display import Image 
```

```python
# pd.read_excel을 통해 데이터를 읽고 변수에 저장
launch_data = pd.read_excel('RocketLaunchDataCompleted.xlsx')

# .head() 함수를 통해 데이터의 상위 5개 행을 출력
launch_data.head()
```

```python
# .columns 함수를 통해 데이터의 모든 열을 볼 수 있음
launch_data.columns
```

---

### 데이터 정리 및 조작(1)
올바르지 않거나 엉망으로 보이는 데이터를 가져와서 값을 변경하거나 삭제하여 정리하는 역활로, 일관되지 않은 데이터를 확인하거나 데이터에 null인 많을 경우 컴퓨터가 혼동을 일으키기 때문에 이러한 작업이 필요합니다.

```python
# .info() 함수를 통해 컬럼명, 데이터 값의 타입 등 데이터에 대한 전반적인 정보 표시
launch_data.info()
```
전체적으로 훑어보다보면 일부 열에 데이터가 누락되어 있어 수정이 필요한 곳이 몇 군데가 보입니다.

```python
# 데이터가 누락되어 있는 곳을 다른 적절한 값으로 변경 
## (Excel 파일에 저장된 데이터가 아니라 launch_data 변수에 저장된 데이터를 변경)

# 'Launched' 열에 데이터가 누락된 곳은 누락 값을 N으로 지정
launch_data['Launched?'].fillna('N',inplace=True)
# 'Crewed or Uncrewed'(유인 or 무인) 정보가 없는 행의 경우 무인으로 가정
launch_data['Crewed or Uncrewed'].fillna('Uncrewed',inplace=True)
# '바람 방향'이 누락되면 unknown으로 표시
launch_data['Wind Direction'].fillna('unknown',inplace=True)
# '컨디션' 데이터가 누락되면 일반적인 날로 간주하고 fair를 사용
launch_data['Condition'].fillna('Fair',inplace=True)
# 다른 누락된 데이터의 경우에는 0 값을 사용
launch_data.fillna(0,inplace=True)

# 변경한 데이터의 상위 5개 행을 출력
launch_data.head()
```

<figure class="half">
    <a href="/images/RocketLaunch/before.jpg"><img src="/images/RocketLaunch/before.jpg"></a>
    <a href="/images/RocketLaunch/after.jpg"><img src="/images/RocketLaunch/after.jpg"></a>
    <figcaption>변경 전(왼쪽)과 변경 후(오른쪽) </figcaption>
</figure>

누락값을 정리한 후 컴퓨터가 계산을 하기 위해서,\
계산은 텍스트보다는 숫자 입력이 적합하므로 모든 텍스트를 숫자로 변환해줍니다.

```python
## 데이터 정리 과정의 일환으로 텍스트 데이터를 숫자로 변환해야 하는 이유는 컴퓨터가 숫자만 이해하기 때문
label_encoder = preprocessing.LabelEncoder()

# 이 3개의 열에는 범주형 텍스트 정보가 있으므로, 이를 숫자로 변환해줄 필요가 있음
launch_data['Crewed or Uncrewed'] = label_encoder.fit_transform(launch_data['Crewed or Uncrewed'])
launch_data['Wind Direction'] = label_encoder.fit_transform(launch_data['Wind Direction'])
launch_data['Condition'] = label_encoder.fit_transform(launch_data['Condition'])
```

데이터를 한 번 살펴보고 잘 정리되었는지를 확인합니다.
```python
launch_data.head()
```

<figure>
    <a href="/images/RocketLaunch/datatheorem.jpg"><img src="/images/RocketLaunch/datatheorem.jpg"></a>
    <figcaption> 정리된 데이터를 보면 'Crewed or Uncrewed'와 'Condition'의 값이 숫자로 바뀌었음을 알 수 있습니다.  </figcaption>
</figure>

이번에는 로켓 발사와 관련하여 필요하지 않거나 사용하지 않을 일부 열들을 제거하고,\
앞으로 사용할 열들만 남겨두도록 합니다.

```python
# 관심 있는 출력(로켓 발사 성공 여부)을 따로 저장
y = launch_data['Launched?']

# 관심 없는 기둥 제거
launch_data.drop(['Name','Date','Time (East Coast)','Location','Launched?','Hist Ave Sea Level Pressure','Sea Level Pressure','Day Length','Notes','Hist Ave Visibility', 'Hist Ave Max Wind Speed'],axis=1, inplace=True)

# 그리고 나머지 데이터들은 입력 데이터로 저장
X = launch_data
```

당연히 로켓 발사 성공 여부를 예측하기 위해서 컴퓨터는 X에 입력된 데이터를 중점으로 살피게 됩니다.

```python
# 기계 학습 알고리즘이 살펴볼 변수 목록:
X.columns
```
