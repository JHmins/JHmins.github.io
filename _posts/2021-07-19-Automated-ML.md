---
layout: post
title: "Automated ML을 통한 데이터 자동시각화(EDA)"
excerpt: "Sweetviz와 Pandas를 이용한 데이터 자동시각화"
tags: 
  - Python
  - Sweetviz
  - Pandas
  - Data science
  - Machine Learning
---

## Automated ML을 통한 데이터 자동시각화(EDA)
&nbsp; 이번에는 `Sweetviz`를 사용한 간단한 데이터 자동시각화(EDA)를 해보았습니다. `Sweetviz`는 간단한 코드 몇 줄만으로 '<ins>데이터를 빠르게 시각화하여 데이터 세트를 비교 (대상 분석, 비교, 기능 분석, 상관 관계) 하는데 중점을 두고 있는 라이브러리</ins>' 중 하나입니다. 사용할 데이터 자료는 데이터 사이언스를 공부하는 사람들은 한 번쯤은 들어봤을 Kaggle의 타이타닉 데이터셋을 사용하였습니다. 

---

```python
import sweetviz as sv
import pandas as pd

# 모델을 학습시키기 위한 train 데이터를 불러옵니다.
df_train = pd.read_csv("train.csv")

# analyze 함수를 통해 리포트를 생성합니다.
my_report = sv.analyze(df_train)
my_report.show_html()
```
&nbsp; 위의 코드를 실행시켜보면

<figure class="half">
    <a href="/images/AutomatedML/terminal1.jpg"><img src="/images/AutomatedML/terminal1.jpg"></a>
    <a href="/images/AutomatedML/terminal2.jpg"><img src="/images/AutomatedML/terminal2.jpg"></a>
</figure>

&nbsp; 몇 초만에, 자동으로 데이터 분석,비교가 된 리포트가 SWEETVIZ_REPORT.html 파일로 만들어져서 나옵니다.

<figure>
    <a href="/images/AutomatedML/report1.jpg"><img src="/images/AutomatedML/report1.jpg"></a>
    <figcaption> html 파일로 만들어진 리포트 </figcaption>
</figure>

---

&nbsp; Sweetviz는 받은 데이터를 빠르게 시각화하여 간단히 훑어보기에는 좋은 라이브러리입니다. 또한, `compare 함수`를 사용하여 두 개의 데이터들을 비교하는 것도 가능합니다. 간단한 코드와 리포트 분석 결과 내용은 다른 EDA 라이브러리와 비교해보아도 나쁘지 않고, 괜찮은 것 같습니다. 그리고 계속 꾸준한 업데이트가 되고 있기 때문에 앞으로의 행보가 무척 기대됩니다ㅎㅎ
