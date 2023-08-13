---
layout: post
title: "deeplearningSession1"
---
# 머신러닝
머신러닝은, 기존의, input이 있고, output이 있어, 그 중간의 과정을 직접 코딩하는 기존 문제해결방법과 상이되는, 답을 유추해줄 수 있는 최적의 함수를 찾는 것이다. 즉, 정확하게 맞추는 것이 아니라, 예측을 하는 것이다. 이 머신러닝은, 점점 더 input과 output이 복잡해지는 빅데이터의 시대가 옴으로써, 대두되었다.
## 예
y = w0 + w1*x1 + w2*x2 + w3*x3 + .......+wn*xn
# 딥러닝
딥러닝은 이 머신러닝 분야 중 하나의 분야로써, F(X) = w0 + w1*x1 + w2*x2 + w3*x3 + .......+wn*xn 와 같은 식이 있을 때, 딥러닝이 학습하는 것은 가중치 W값이라고 생각하면 된다.
# RSS(Residual Sum of Square)
오류 값의 제곱을 구해서 더하는 방식. 일반적으로 미분 등의 계산을 편리하게 하기 위해서 RSS 방식으로 오류 합을 구한다. 즉, Error**2 = RSS 이다.
# 경사하강법
점진적으로 반복적인 계산을 통해, W 파라미터 값을 업데이트 하면서 오류 값이 최소가 되는 W 파라미터를 구하는 방식이다. 오류 값을 점점 줄이는 방향으로 업데이트 한다. 이는, 미분 값을 이용하는데, 미분 값이 0으로 향해서 간다면, 최소 값이 된다는 아이디어에서 착안한다.
# GD(Gradient Descent) vs SGD vs Mini - Batch GD
* GD : 전체 학습 데이터를 기반으로 GD 계산
* SGD : 전체 학습 데이터 중 한 건만 임의로 선택하여 GD 계산
* Mini - Batch GD : 전체 학습 데이터 중 특정 크기 만큼(Batch 크기) 임의로 선택해서 GD 계산
# 경사하강법의 문제점
* 1. learning rate 크기에 따른 이슈
너무 작으면, 최소 점에 수렴하는데 너무 오랜 시간이 걸림. 너무 크면, 최소점을 찾지 못하거나 오히려 발산될 수 있음.
* 2. 전역 최소점(Global Minimum)과 국소 최소점(Local Minimum) 이슈
전역최소점이 아니라, 국소 최소점으로 수렴할 수 있음.
# 코드
```python
from sklearn.datasets import load_boston
boston = load_boston()
bostonDF = pd.DataFrame(boston.data, columns=boston.feature_names)
bostonDF['PRICE'] = boston.target
print(bostonDF.shape)
bostonDF.head()
#gradient_descent() 함수에서 반복적으로 호출되면서 update될 weight/bias 값을 계산하는 함수.
#rm은 RM(방 개수), lstat(하위계층 비율), target은 price임. 전체 array가 다 입력됨
#반환 값은 weight와 bias가 update되어야 할 값과 mean squared error 값을 loss로 반환
def get_update_weights_value(bias,w1,w2,rm,lstat,target,learning_rate = 0.01):
    # 전체 데이터 수
    N = len(target)
    # 예측 값
    predicted = w1 * rm + w2 * lstat + bias
    diff = target - predicted
    bias_factor = np.ones((N,))
    w1_update = -(2/N)*learning_rate*(np.dot(rm.T,diff))
    w2_update = -(2/N)*learning_rate*(np.dot(lstat.T,diff))
    bias_update = -(2/N)*learning_rate*(np.dot(bias_factor.T, diff))
    mse_loss = np.mean(np.square(diff))
    return bias_update, w1_update, w2_update, mse_loss
#RM, LSTAT feature array와 PRICE target array 를 입력 받아서 iter_epochs 수만큼 반복적으로 Weight, Bias를 update 적용
def gradient_descent(features, target, iter_epochs = 1000, verbose = True):
    # w1, w2 는 numpy array 연산을 위해 1차원 array로 변환하되, 초기 값은 0으로 설정
    # bias도 1차원 array로 변환하되, 초기 값은 1로 설정
    w1 = np.zeros((1,))
    w2 = np.zeros((1,))
    bias = np.ones((1,))
    print("최초 w1, w2, bias : ", w1, w2, bias)
    # learning_rate와 RM, LSTAT 피처 지정, 호출 시 numpy array 형태로 RM과 LSTAT으로 된 2차원 feature가 입력됨.
    learning_rate = 0.01
    rm = features[:,0]
    lstat = features[:,1]
    #iter_epochs 수 만큼 반복하면서 weight와 bias update 수행
    for i in range(iter_epochs):
        # weight/bias update 값 계산
        bias_update, w1_update, w2_update, loss = get_update_weights_value(bias, w1,w2,rm,lstat,target,learning_rate)
        # weight/bias의 update 적용
        w1 = w1 - w1_update
        w2 = w2 - w2_update
        bias = bias - bias_update
        if verbose:
            print("Epoch:",i+1,'/',iter_epochs)
            print("w1",w1,"w2",w2,"bias:",bias,"loss:",loss)
    return w1,w2,bias
# Gradient Descent 적용
# 신경망은 데이터를 정규화/표준화 작업을 미리 선행해 주어야 함
# 이를 위해 사이킷런의 MinMaxScalar를 이용하여 개별 feature 값은 0~1사이 값으로 변환 후 학습 적용
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaled_features = scaler.fit_transform(bostonDF[['RM','LSTAT']])
w1, w2, bias = gradient_descent(scaled_features, bostonDF['PRICE'].values, iter_epochs = 1000, verbose = True)
print('#### 최종 w1, w2, bias ####')
print(w1,w2,bias)
predicted = scaled_features[:,0]*w1 + scaled_features[:,1]*w2 + bias
bostonDF['PREDICTED_PRICE'] = predicted
bostonDF.head(10)
# Keras를 이용하여 보스턴 주택가격 모델 학습 및 예측
# Dense Layer를 이용하여 퍼셉트론 구현. units는 1로 설정.
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
model = Sequential([
    # 단 하나의 units 설정. input_shape은 2차원, 회귀이므로 activation은 설정하지 않음
    # weight와 bias 초기화는 kernel_inbitializer와 bias_initializer를 이용
    Dense(1, input_shape = (2,), activation = None, kernel_initializer = "zeros", bias_initializer = "ones")
])
model.compile(optimizer = Adam(learning_rate = 0.01), loss = "mse", metrics = ['mse'])
model.fit(scaled_features, bostonDF['PRICE'].values, epochs=1000)
```