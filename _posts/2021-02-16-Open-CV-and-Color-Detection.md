---
layout: post
title: "Open CV 색상 감지"
excerpt: "Open CV를 이용한 지정한 색상 감지"
tags: 
  - Python
  - Open CV
  - Numpy
  - Color recognition
  - Artificial Intelligence
---
## Open CV 색상 감지
&nbsp; 이번에는 영상 처리 라이브러리 중 하나인 Open CV를 이용하여, 특정 지정한 색상을 감지할 수 있도록 해보았습니다.

```python
import numpy as np
import argparse
import cv2

# 명령행 인수 구문 분석을 처리
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", help = "입력하려는 이미지 경로")
args = vars(ap.parse_args())

# 이미지를 로드
image = cv2.imread(args["image"])

# 색상 경계값(하한,상한)을 미리 정의해줍니다.
boundaries = [
	([17, 15, 100], [50, 56, 200]),
	([86, 31, 4], [220, 88, 50]),
	([25, 146, 190], [62, 174, 250]),
	([103, 86, 65], [145, 133, 128]) 
]

# 경계값을 반복합니다.
for (lower, upper) in boundaries:

	# 경계값에서 Numpy 배열을 생성합니다.
	lower = np.array(lower, dtype = "uint8")
	upper = np.array(upper, dtype = "uint8")

	# 지정된 경계값 내에서 색상을 찾아 마스크 적용해줍니다.
	mask = cv2.inRange(image, lower, upper)
	output = cv2.bitwise_and(image, image, mask = mask)

	# 이미지를 보여줍니다.
	cv2.imshow("images", np.hstack([image, output]))
	cv2.waitKey(0)
```

작성한 코드는 대략 이렇습니다. 여기서 중요한 점은

> &nbsp; 색상의 경계값을 정할 때 Open CV는 RGB 형식이 아닌 BGR 형식을 사용한다는 점입니다. `boundaries = ([17, 15, 100], [50, 56, 200])`은 "B>=17, G>=15, R>=100"에서 ~ "B<=50, G<=56, R<=200" 사이의 색상을 적색으로 간주하게 됩니다. 다른 색상들도 이와 마찬가지입니다.

```python
boundaries = [
	([17, 15, 100], [50, 56, 200]),  -> 적색
	([86, 31, 4], [220, 88, 50]),    -> 청색
	([25, 146, 190], [62, 174, 250]), -> 노랑색
	([103, 86, 65], [145, 133, 128])  -> 회색
]
```

> &nbsp; 또한 cv2.inRange 함수에서 `첫 번째 인수는 우리가 색 감지를 수행할 이미지이고, 두 번째 인수는 탐지하려는 색상의 하한이며, 세 번째 인수는 탐지하려는 색상의 상한입니다.`
여기서 범위에 들어가는 부분은 값 그대로, 나머지 부분은 0으로 채워져서 결과값을 반환하게 됩니다. (따라서, 우리가 지정한 범위 이외에 부분은 검은색으로 처리된후 나오게 됩니다.)

```python
  # 이미지에서 특정 색상만 추출하기 위한 임계값으로 사용합니다.
  mask = cv2.inRange(image, lower, upper)

  # mask와 원본 이미지를 비트 연산합니다.
  output = cv2.bitwise_and(image, image, mask = mask)
```

<figure>
    <a href="/images/ColorDetection/shoes.jpg"><img src="/images/ColorDetection/shoes.jpg"></a>
    <figcaption> 넣고자 하는 하는 샘플 이미지 파일 </figcaption>
</figure>

<figure class="third">
	<a href="/images/ColorDetection/shoes_red.jpg"><img src="/images/ColorDetection/shoes_red.jpg"></a>
	<a href="/images/ColorDetection/shoes_blue.jpg"><img src="/images/ColorDetection/shoes_blue.jpg"></a>
	<a href="/images/ColorDetection/shoes_gray.jpg"><img src="/images/ColorDetection/shoes_gray.jpg"></a>
	<figcaption>감지해서 나온 색상들 빨강 - 파랑 - 회색 순서대로(노랑은 실수로 빠졌네요ㅎㅎ)</figcaption>
</figure>

&nbsp; `cv2.inRange` 함수는 매우 효율적이지만, 이미지에 적용하기 위해서는 반드시 색상 경계를 알아야 한다는 단점이 있습니다. 그리고 인터넷을 통해 찾아본 결과 음영을 포함한 넓은 색상을 넣기 위해서는 HSV 색 공간을 사용하는 것이 훨씬 쉽다고 합니다. (HSV에서 고른 색상을 다시 RGB로 변환하여 경계값에 넣어 사용하시면 됩니다.)