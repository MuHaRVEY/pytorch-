import torch.nn as nn
 #모든 신경망 모델은 nn.Module클래스를 상속하여 정의된다.
 #이는 forward pass와 backward pass 등의 학습 알고리즘을 구현하는데 사용됨.
       #torch.nn은 다양한 종류의 레이어를 제공함.
        #선형 레이어 / 합성곱 레이어 / 순환 신경망 레이어 등이 있음
        # 입력 차원: 10, 출력 차원: 5
        #이 코드에서 사용하고 있는 nn.Linear는 선형 레이어임.

class SimpleModel(nn.Module):
    def __init__(self): #생성자 매서드
        super(SimpleModel,self).__init__() #부모 클래스인 nn.Module의 생성자를 호출함
        self.fc = nn.Linear(10,5) 

    def forward(self,x): #순전파란 입력 데이터를 받아 출력을 계산하는 과정을 의미함
        x = self.fc(x) #단순히 x를 입력받아 출력을 계산하는 과정을 의미하고 있음
        return x        #x를 slef.fc라는 선형 레이어에 통과시켜 결과를 반환하는 것
    
#모델 인스턴스 생성
model = SimpleModel() #정의한 신경망 모델의 인스턴스
print(model) #모델 구조 출력

#출력 결과
# SimpleModel(
#   (fc): Linear(in_features=10, out_features=5, bias=True)
# )
