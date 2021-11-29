# ML Note

> 궁금했던 부분들 해결한 것 정리

<br>
 

#### 그래프 상에서 설명하는 역전파는 쉽게 이해할 수 있는데, 이를 행렬계산 수식으로 구현한 것도 직관적으로 이해할 수 있을까 ?
----
  ( W.shape = (output node, input feature), X.shape(feature, data) 일때, dW = (1/m) * dot_product(dZ, X.T) )

  - **w, x, z 에서 확장될 수 있는 요소(feature , data, output node) 들이 가지는 의미와 dot product 의 특성을 비교**해서 생각하면 된다.

      dw 가 x 와 dz 의 곱으로 이루어진다는 것은 변함이 없지만, 계산할 대상 (feature, output node) 와 계산할 때 더해야 하는 대상(data) 이 늘어났을 때 이를 한꺼번에 계산할 수 있도록 하는 것이 행렬계산이 하는 역할. 따라서 X, 와 dZ 에서 data 를 의미하는 차원이 서로 더해지도록 하고 feature 와 output 을 의미하는 차원이 남도록 하려면 dZ*X.T 의 형태로 dot product 를 해야 하는 것.
<br><br>        


#### 보통 gradient descent 의 원리에 대해 설명하는 자료들은 2차원 상황을 예시로 드는데, n차원 이상의 환경에서도 gradient 의 반대 방향은 항상 아래로 내려가는 방향일까?
----
  - 미분 가능하고 연속적인 함수라면 gradient 방향은 가장 가파르게 올라가는 방향(steepest ascent) 임.  ([Why is gradient the direction of steepest ascent?](https://math.stackexchange.com/questions/223252/why-is-gradient-the-direction-of-steepest-ascent))

      기본적으로 경사하강법을 사용할 때, **minimum point 를 지나쳐서 다시 cost 가 올라가는 경우를 제외하고는** 항상 하강 해야 함.

      실제 학습 시킬때 cost 가 진동하면서 내려가는 이유는 Dropout 이나 mini batch 의 사용 등으로 인한 것.
<br><br>

#### Momentun vs RMSProp
----
  - 둘다 gradient 에 대한 지수가중 평균 식을 사용하기 때문에 비슷해 보일 수 있지만 Moment에서는 그냥 **w 벡터** 를 사용하고 RMSProp 은 **w 벡터의 요소별 제곱** 을 사용하기 때문에 Momentum 에서는 V_t 가 진동을 많이 하고 있는 차원일수록 작아지고, RMSProp 에서는 반대로 커짐.

      따라서 업데이트 식에서 Momentum 의 V_t 는 그냥 들어가고 RMSProp 의 V_t 는 분모로 들어간다.
<br><br>

#### Dropout VS L2 Regularization
----  
  - L2 Regularization 을 사용하면 W 를 작게 만드는게 목표이기 때문에 W 중에 0에 가까워지는 원소가 많아지고, Dropout 을 사용하면 W가 분산 되어서 오히려 0에 가까운 원소가 적어지는 생각을 하면서, 그럼 서로 반대되는 효과를 내는 것이 아닌가? 라고 봤었는데, 실제로는 둘 다  **특정 가중치 값이 지나치게 영향력을 갖는 것을 방해하여** 모델이 과도하게 복잡해지는 걸 막는다는 공통점을 가지고 있다. 방법이 다르기 때문에 어떤 부분에서는 서로 다른 역할을 하는 것처럼 보일 수 있지만 큰 목표에서는 동일함.
<br><br>
        
#### least square VS gradient descent
----  
  - 둘 다 Cost function 을 최적화 하는데 쓰이는 방법. **least  square** 는 한번의 계산으로 최적의 파라미터를 바로 찾을 수 있지만 최적의 파라미터를 구하는 식이 연립 방정식의 형태로 정리가 되어야 한다는 제한이 있고, **gradient descent** 는 파라미터와 cost function 사이의 도함수를 구하거나 근사할 수 있으면 사용할 수 있다는 면에서 다양한 목적 함수와 모델을 대상으로 사용할 수 있지만 여러 번의 step 을 거쳐야 한다는 단점이 있다.

      실제로 라이브러리를 보면 sklearn 의 단순 선형 회귀 등의 모델은 내부적으로 least square 를 사용하고, 딥러닝 관련 라이브러리는 gradient descent 사용.
<br><br>

#### Regularization VS Normalization
----  
  - **둘다 한국어로는 "정규화"** 라고 번역이 되어서 헷깔렸는데

      Regularization 은 Overfitting 을 막는 것을 목표로 하고, Normalizaiton 은 scale 이 큰 feature 의 영향이 비대해지지 않게 방지하는 것을 것을 목표로 함.
<br><br>

#### 딥러닝 순전파, 역전파를 최대한 행렬계산으로 하는 것이 유리한 이유
----  
  - CPU 혹은 GPU 의 **SIMD**(Single Instruction Multiple Data) 을 활용하기 위함임. GPU 가 병렬 계산을 CPU 보다 잘 해서 학습에 유리하다고 하는 이유가 이것 때문.
<br><br>

#### 비선형 활성화 함수의 역할

---

- 이에 대해서는 보통 *"**활성화 함수가 없다면 신경 망은 입력 값에 대해 선형식만을 계산하기 때문에 은닉층이 없는 것이나 다름이 없다.**"* 라고 설명이 되어 있다. ([Why must a nonlinear activation function be used in a backpropagation neural network?](https://stackoverflow.com/questions/9782071/why-must-a-nonlinear-activation-function-be-used-in-a-backpropagation-neural-net))
- (개인적인 생각)
    
    다층 신경망을 쓰는 이유의 예시로 알려져 있는 **perceptron 의 XOR 문제**에서도 사실 step function 이라는 활성화 함수가 없다면 아무리 층을 많이 쌓아도 해결 불가능하다. 활성화 함수가 입력 값의 범위에 따라서 이를 의미 있는 정보로 바꿔주기 때문에 (0 or 1) 추가적인 층을 사용하는 것이 도움이 되는 것.
    
    비선형적인 활성화함수라는건 입력 값의 범위에 따라 다른 양상의 출력값이 나오도록 함으로써  **입력 값이 가지는 특성을 변환해서 표현해주는** 역할을 하는 것 같다.
