import torch
import torch.nn as nn
import torch.optim as optim

torch.manual_seed(42) #난수 생성 시드값 고정
X = torch.randn(100,1) #정규분포에서 무작위로 샘플링된 데이터를 생성
# 100개의 행과 1개의 열로 이루어진 2차원 텐서를 의미함
# 즉, 100개의 샘플을 갖는 열벡터를 생성하는 것을 의미.
#torch.randn 함수는 평균이 0이고 표준 편차가 1인 정규 분포로부터 데이터를 생성함
# 반환되는 값이 보통 -3과 3사이에 있음.
#그러나 표준 편차가 1이므로 보통 범위가 조금 더 넓게 잡힘.
y = 3 * X + 2 + 0.1 * torch.randn(100,1) 
# 목표 데이터는 3X + 2임. 그 뒤는 노이즈를 추가하여
#실제 데이터 상에서 불확실성을 모방하기 위해 추가하는 것임.
# 실제 데이터는 노이즈가 존재. 이를 통해 선형 회귀 모델의 학습을 시뮬레이션 해보는 것.


#선형 회귀 모델 단순하게 정의
class LinearRegressionModel(nn.Module): #nn모듈 상속하여 정의
    def __init__(self):
        super(LinearRegressionModel, self).__init__()
        self.linear = nn.Linear(1,1) #입력 차원 1, 출력 차원 1

    def forward(self,x):
        return self.linear(x)
    
model = LinearRegressionModel() #인스턴스 생성

#손실 함수와 옵티마이저 정의
criterion =nn.MSELoss()  #평균 제곱 오차
optimizer = optim.SGD(model.parameters(), lr=0.01) #확률적 경사 하강법? 옵티마이저 

#학습 과정 
epochs = 100 #100번의 epoch 동안 모델을 학습한다. 

for epoch in range(epochs):
    #순전파 단계
    outputs = model(X) #모델이 예측한 값 outputs
    
    #손실 계산 
    loss = criterion(outputs, y) #예측값과 y 사이의 손실을 계산하는 criterion

    #역전파 단계와 가중치 업데이트
    optimizer.zero_grad() 
    loss.backward() 
    optimizer.step()

    if (epoch+1) % 10 == 0: #10번째 에코마다 손실값을 출력함. ㅁ
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')

# 최종 모델의 파라미터 출력
print("모델의 가중치:", model.linear.weight.item())
print("모델의 편향:", model.linear.bias.item())




