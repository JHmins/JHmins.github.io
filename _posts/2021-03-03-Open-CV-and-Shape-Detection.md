---
layout: post
title: "Open CV 모양(도형) 감지"
excerpt: "Open CV를 이용한 모양(도형) 감지"
tags: 
  - Python
  - Open CV
  - Shape recognition
  - Artificial Intelligence
---
## Open CV 모양(도형) 감지
&nbsp; 이번에는 Open CV를 활용하여, 모양(도형)을 감지할 수 있도록 해보았습니다. 이번에는 실행할 main.py와는 별도로 클래스의 구현을 저장할 .py를 추가로 만들 것이기 때문에 별도의 패키지를 만들어 놓았습니다. 

```
:\Project
    - ShapeDetect
        --- shapedetector.py
        --- __init__.py             
    - shapes_detection.py
    - 그 외 jpg,png
```

&nbsp; 폴더(디렉터리) 안에 `_init_.py` 파일이 있는 이유는 있어야지 해당 폴더를 패키지로 인식하기 때문입니다. `_init_.py`는 비워둘 수도 있고, 패키지를 초기화하는 역할도 가능합니다. (파이썬 3.3 이후로는 `_init_.py` 파일이 없어도 패키지로 인식이 가능함.) 일단 저는 먼저 클래스의 구현을 저장할 shapeDetector.py를 만들어보겠습니다.

---

```python
import cv2

class ShapeDetector:
	def __init__(self):
		pass

	def detect(self, contour):
		# 도형 이름을 초기화하고 윤곽선을 근사화합니다.
		shape = "unidentified"
		epsilon = cv2.arcLength(contour, True)
		approx = cv2.approxPolyDP(contour, epsilon * 0.04, True)
        
        # 꼭짓점 3개면 그 도형은 삼각형입니다.
		if len(approx) == 3:
			shape = "Triangle"

		# 정점이 4개인 경우 그 도형은 정사각형 또는 직사각형입니다.
		elif len(approx) == 4:

			# 윤곽선의 경계 상자를 계산하고 이를 사용하여 종횡비를 계산합니다.
			(x, y, w, h) = cv2.boundingRect(approx)
			ar = w / float(h)

			# 정사각형은 대략 1과 같은 종횡비를 갖습니다. 그 외는 직사각형입니다.
			shape = "Square" if ar >= 0.95 and ar <= 1.05 else "Rectangle"

		# 꼭짓점 5개면 그 도형은 오각형입니다.
		elif len(approx) == 5:
			shape = "Pentagon"

		# 그 외에는 원이라고 가정합니다.
		else:
			shape = "Circle"

		# 도형의 이름을 반환합니다.
		return shape
```
&nbsp; 저는 클래스의 구현을 저장할 ShapeDetector.py를 만들었고, 저는 모양 감지를 하기 위해서 `윤곽 근사(Contour approximation)`를 이용했습니다.
> 윤곽 근사(Contour approximation)란 `Ramer-Douglas-Peucker 알고리즘`을 이용하여 '지정한 정밀도에 따라 정점의 수가 적은 모양으로 윤곽 모양을 근사' 하는 방법입니다. `근사치 정확도 값(epsilon)`은 입력할 다각형과 반환할 근사화된 다각형 사이 최대 편차를 고려해 다각형을 근사하는 원리입니다. 이러한 근사치 정확도의 값은 작을수록, 근사를 더욱 적게 하기 때문에 원본 윤곽과 유사해진다고 합니다.

```python
	# contour는 검출한 윤곽선들이 저장되는 Numpy 배열, True는 검출한 윤곽선이 닫혀있는지, False는 열려있는지를 의미합니다.
    epsilon = cv2.arcLength(contour, True) 

	# 저는 근사치 정확도를 위해 윤곽선 전체 길이의 4%로 활용했습니다.
	approx = cv2.approxPolyDP(contour, epsilon * 0.04, True) 
```

---

&nbsp; 이제는 본격적으로 실행할 shapes_detection.py 을 만들어보겠습니다.

```python
from .ShapeDetect.shapedetector import shapedetector
import argparse
import imutils
import cv2

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
	help="입력하려는 이미지 경로")
args = vars(ap.parse_args())

# 도형이 더 잘 근사화될 수 있도록, 이미지를 더 작은 크기로 조정합니다.
image = cv2.imread(args["image"])
resized = imutils.resize(image, width=300)
ratio = image.shape[0] / float(resized.shape[0])

# 사이즈 조정한 이미지를 회색조로 변환하고, 살짝 흐리게 한 다음 임계값을 설정합니다.
gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray, (5, 5), 0)
thresh = cv2.threshold(blurred, 60, 255, cv2.THRESH_BINARY)[1]

# 임계값(이진) 이미지에서 윤곽선을 찾고, Shapedetector를 초기화합니다.
cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)
sd = ShapeDetector()

# 윤곽선을 반복합니다.
for c in cnts:
	# 윤곽선의 중심을 계산한 다음, 윤곽선을 사용하여 모양의 이름을 감지합니다.
	M = cv2.moments(c)
	cX = int((M["m10"] / M["m00"]) * ratio)
	cY = int((M["m01"] / M["m00"]) * ratio)
	shape = sd.detect(c)

	# 윤곽선 (x, y)에 크기 조정 비율을 곱한 후, 윤곽선과 모양의 이름을 이미지에 그립니다.
	c = c.astype("float")
	c *= ratio
	c = c.astype("int")
	cv2.drawContours(image, [c], -1, (0, 255, 0), 2)
	cv2.putText(image, shape, (cX, cY), cv2.FONT_HERSHEY_SIMPLEX,
		0.5, (255, 255, 255), 2)

	# 출력할 이미지 표시
	cv2.imshow("Image", image)
	cv2.waitKey(0)
```

&nbsp; 이러한 영상처리를 하는 코드들을 보면 주로 전처리 과정에서 사이즈를 조절하거나, `Grayscale(그레이스케일)` 작업과 `Gaussian blur(가우시안 블러)` 작업을 하게 되는데, 그 이유는 보통 이미지 노이즈를 줄이고 세부 사항을 줄여 컴퓨터가 이진화 처리를 하기에 편리하기 때문입니다.

<figure class="half">
    <a href="/images/ShapeDetection/sample.png"><img src="/images/ShapeDetection/sample.png"></a>
    <a href="/images/ShapeDetection/sampleout.jpg"><img src="/images/ShapeDetection/sampleout.jpg"></a>
    <figcaption>샘플 이미지(왼쪽)과 출력 후(오른쪽) </figcaption>
</figure>

> Tip. 저는 영어로 출력했지만, 만약에 Open CV로 한글을 출력하고 싶으면 PIL(Python Image Library)을 사용해 한글을 출력해주면 됩니다. 그냥 하면 한글이 깨지기 때문에 한글을 출력하고 싶을 경우 꼭 사용해줘야 합니다.

 ---

&nbsp; P.S. 위에 있는 코드대로 실행을 시키면 아래와 같은 에러가 발생하게 됩니다.\
" ImportError: attempted relative import with no known parent package "은 메인 모듈에서 상대경로를 사용하다가 발생하는 오류로

```python
from .ShapeDetect.shapedetector import shapedetector
```

&nbsp; 그럴 경우에는 코드를 상대 경로에서 절대 경로로 수정해주면됩니다.

```python
# 지금 현재 있는 코드에서 코드를 추가해줍니다.

# 현재 모듈의 절대경로를 알아내어 추가하는 방식
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 그리고 앞에 있는 . 한 개 지움
from ShapeDetect.shapedetector import shapedetector
```
