#파이토치 공부하기

import torch
from torch import nn
from torch.utils.data import DataLoader #데이터 작업을 위한 기본 요소 1 - DataSet을 iterable 객체로 감쌈
from torchvision import datasets # 데이터 작업을 위한 기본 요소 2 - 샘플과 정답 label을 저장 
from torchvision.transforms import ToTensor

#파이토치는 TorchText, TorchVision 및 TorchAudio 같은 도메인 특화 라이브러리를 데이터셋과 함께 제공
#torchvision.datasets 모듈을 실제 vision 데이터에 대한 데이터셋을 제공

#학습 데이터 내려받기
training_data = datasets.FashionMNIST(
    root ="data",
    train=True,
    download = True,
    transform=ToTensor(),
)

#공개 데이터셋에서 테스트 데이터를 내려받습니다.
test_data = datasets.FashionMNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor(),
)

batch_size = 64

#데이터 로더를 생성합니다.
train_dataloader = DataLoader(training_data, batch_size = batch_size)
test_dataloader = DataLoader(test_data, batch_size = batch_size)

for X,y in test_dataloader:  #입력 데이터 X, 목표 값 (레이블) y
  print(f"Shape of X [N,C,H,W]: {X.shape}")
  print(f"Shape of y: {y.shape} {y.dtype}")
  break


device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using {device} device")

class NeuralNetWork(nn.Module):
  def __init__(self):
    super().__init__()
    self.flatten = nn.Flatten()
    self.linear_relu_stack = nn.Sequential(
      nn.Linear(28*28, 512),
      nn.ReLU(),
      nn.Linear(512,512),
      nn.ReLU(),
      nn.Linear(512,10)
    )

  def forward(self, x):
    x = self.flatten(x)
    logits = self.linear_relu_stack(x)
    return logits

model = NeuralNetWork().to(device)
print(model)  

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

def train(dataloader, model, loss_fn, optimizer):
  size = len(dataloader.dataset)
  for batch, (X,y) in enumerate(dataloader):  #enumerate() = 순서가 있는 자료형을 입력으로 하였을 때, 인덱스와 값을 함께 리턴
    X,y = X.to(device), y.to(device) #GPU 혹은 CPU의 디바이스로 입력 데이터와 레이블을 이동, 같은 디바이스에 놓아야 효율적인 연산이 가능하기 때문

    #예측 오류 계산
    pred = model(X)
    loss = loss_fn(pred, y)
    
    #역전파? == 입력 데이터에 대해 예측한 결과와 실제 결과 간의 오차를 계산, 이 오차를 최소화 하기 위해 가중치를 조정하는 과정 
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    
    if batch % 100 == 0:
      loss, current = loss.item(), (batch+1) * len(X)
      print(f"loss : {loss:>7f} [{current:>5d} / {size:>5d}]")

def test(dataloader, model, loss_fn):
  size = len(dataloader.dataset)
  num_batches = len(dataloader)
  model.eval()
  test_loss, correct =0, 0
  with torch.no_grad():
    for X,y in dataloader:
      X,y = X.to(device), y.to(device)
      pred = model(X)
      test_loss += loss_fn(pred, y).item()
      correct += (pred.argmax(1) ==y).type(torch.float).sum().item()
  test_loss /= num_batches
  correct /= size
  print(f"Test Error : \n Accuracy : {(100*correct) :>0.1f}%, Avg loss: {test_loss :>8f} \n")

epochs = 5
for t in range(epochs):
  print(f"Epochs {t+1}\n----------------------")
  train(train_dataloader, model, loss_fn, optimizer)
  test(test_dataloader,model,loss_fn)
print("Done!")


torch.save(model.state_dict(), "model.pth")
print("Saved Pytorch Model State to model.pth")

model = NeuralNetWork().to(device)
model.load_state_dict(torch.load("model.pth"))
