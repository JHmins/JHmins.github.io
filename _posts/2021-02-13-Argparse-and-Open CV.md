---
layout: post
title: "Argparse 명령줄 인수와 간단한 Open CV 예제"
excerpt: "Argparse에 대한 간략한 설명과 간단한 이미지 윤곽 인식"
tags: 
  - Python
  - Open CV
  - Image recognition
  - Artificial Intelligence
---
## Argparse 명령줄 인수와 간단한 Open CV 예제
&nbsp; `Argparse` 모듈은 Python에서 많이 사용되는 명령줄 인수 구문 중 하나로, '<ins>인자값을 입력받고 이와 관련된 간단한 도움말과 사용법 작성이 가능합니다.</ins>' 그리고 만약에 사용자가 프로그램에 잘못된 인자를 줄 때 에러를 통해 알려주는 역활도 가능합니다.

### Argparse 간단한 설명

```python
# argparse 모듈 가져오기
import argparse

# argparse.ArgumentParser() 함수를 통해 parser를 생성합니다.
ap = argparse.ArgumentParser()

# 인수를 추가하고 인수의 조건을 설정합니다. 저는 required=True를 통해, parse_args()의 옵션에 그 명령행이 없으면 에러를 보고하도록 설정했습니다.
ap.add_argument("-n", "--name", required=True, help="사용자의 이름")

# parse_args를 통해 인자 문자열을 객체로 변환하고 namespace의 어트리뷰트로 설정합니다.
args = vars(ap.parse_args())

# 그리고 출력
print ("안녕하세요 {}씨, 만나서 반갑습니다.".format(args["name"]))
```

<figure>
    <a href="/images/Argparse/sample.jpg"><img src="/images/Argparse/sample.jpg"></a>
    <figcaption> 터미널에서 한 번 실행시켜 봅시다. </figcaption>
</figure>

Argparse 모듈에 대해서 더욱 자세한 설명을 원한다면\
[https://docs.python.org/ko/3/library/argparse.html](https://docs.python.org/ko/3/library/argparse.html)

### Argparse 명령줄 인수를 활용한 간단한 Open CV 예제

&nbsp; Argparse 명령줄 인수의 특성 때문에 머신러닝이나 딥러닝을 하다보면 자주 보게 됩니다. 우선은 Argparse 명령줄 인수를 이용하여 이미지 윤곽을 찾아내는 간단한 Open CV 예제를 해보았습니다.

```python
import argparse
import imutils
import cv2

# 인수 구문 분석기를 구성하고 인수를 구문 분석합니다.(위에 설명 확인)
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input", required=True,
	help="입력하려는 이미지 경로")
ap.add_argument("-o", "--output", required=True,
	help="출력하려는 이미지 경로")
args = vars(ap.parse_args())

# 디스크에서 입력 이미지를 로드합니다.
image = cv2.imread(args["input"])

# 이미지를 회색조, 흐리게 및 임계값으로 변환
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray, (5,5), 0)
thresh = cv2.threshold(blurred, 60, 255, cv2.THRESH_BINARY)[1]

# 이미지에서 윤곽을 추출
cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
	cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)

# 윤곽선을 반복하여 입력 영상에 그립니다.
for c in cnts:
	cv2.drawContours(image, [c], -1, (0, 0, 255), 2)

# 이미지의 총 모양 수를 표시
text = "I found {} total shapes".format(len(cnts))
cv2.putText(image, text, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2 )

# 출력 이미지를 디스크에 기록
cv2.imwrite(args["output"], image)
```

<figure class="half">
    <a href="/images/Argparse/input1.png"><img src="/images/Argparse/input1.png"></a>
    <a href="/images/Argparse/output1.png"><img src="/images/Argparse/output1.png"></a>
    <figcaption>넣은 그림(왼쪽)과 나온 그림(오른쪽) </figcaption>
</figure>

이 코드는 단순히 이미지 윤곽을 추출만 하는 내용으로, 스스로 학습하는 능력이 없기 때문에 아직까지는 정확도가 많이 부족한 점이 있습니다. 이는 나중에 공부할 머신러닝, 딥러닝 등을 이용하여 겪은 경험을 통해 자동으로 개선이 가능하게 코드를 수정해보도록 하겠습니다.
