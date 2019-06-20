
"""

    *미리보기
    
    1) 초기설정                         : 필요한 상수, 변수, 데이터 저장할 리스트 초기화
    2) spike 생성하기                   : spike발생확률 지저하고 100s 동안의 spike를 발생시켜 저장하고 각 spike의 ISI값 저장
    3) ISI분포도 확인                   : 저장한 ISI값을 바탕으로 ISI의 분포 시각화
    4) ISI값의 평균, 표준편차 구하기     : ISI값의 평균, 표준편차를 구하고 이론값과 비교해보기
    5) autocorrelation 확인하기         : spike발생 간의 규칙이 있는지, 아니면 랜덤한지 convolution을 통해 확인

"""


import numpy as np
import matplotlib.pyplot as plt

"""

    1) 초기설정


    1. firing_rate(Hz) : 10, 20, 30, 40, 50
    2. spike[] : 매 1ms마다 spike가 발생했는지 발생하지 않았는지를 기록하는 리스트
                spike발생시 1, 발생안했을 시 0
                총 100 * 1000 ms(10만)의 기록이 담겨져있음
    3. ISI[] : spike간의 간격을 기록해두는 리스트 (ms단위)
                총 spike의 개수가 ISI리스트 길이와 동일
    4. x_range : ISI분포도 확인시 ISI값의 한계값 설정
                1 ~ 1000ms에 해당하는 ISI 개수만 저장하게끔 1000으로 설정 --> 그 이상은 0에 수렴하기 때문에 의미 없다 판단
                (필요시 원하는 값으로 설정)
    5. distribution[] : distribution[k]에는 k ms의 ISI가 나타난 횟수를 기록해둠
    6. ms : 1ms를 s로 표현하기 위한 값 (1ms = 0.001s)
    6. dt : 1ms에 해당하는 값으로 0.001s를 나타냄
                매 1ms마다 측정하는 시간간격

"""

firing_rate = 10

spike = []
ISI = []

x_range = 100
distribution = [0 for i in range(x_range)]


ms = 0.001
dt = ms



"""

    2) spike 생성하기
    
    
    1. spike발생 확률 : probability(t) = a * dt
    2. 총 100 * 1000 ms 만큼 진행 -> 10만번의 spike generate시도 진행
    3. spike가 일정확률로 발생함을 구현하기 위해 0 ~ 1 사이 값의 균등분포를 따르는 랜덤값 rand를 매 반복마다 새로 뽑는다.
    4. 처음 dt값이 1ms로 초기화 되어있고 매 반복마다 dt값이 1ms추가 되거나 1ms로 다시 초기화 되거나 한다.
    5. 매 반복에서의 dt값으로 dt시간 동안에 spike가 1번 일어날 확률을 랜덤으로 뽑은 a와 비교
        4-1. rand > probability(dt)  : spike확률이 랜덤값을 못넘었으므로 spike 실패를 나타냄
                                    spike[t]에 0을 저장하고 dt값은 1ms 증가
        4-2. rand <= probability(dt) : spike확률이 랜덤값을 넘었으므로 spike 성공을 나타냄
                                    ISI에 현재 dt값을 저장                                 -> 이후 ISI 분포를 확인하는데 사용
                                    spike[t]에 1을 저장하고 dt값은 1ms로 초기화
    5. 모든 반복이 끝나고 spike[t] : t시간에서 spike의 유무
                           ISI[i] : i번째 spike가 일어날때 까지의 ISI의 크기  
                        
"""

def probability(t):
    return firing_rate * ms

for e in range(5): 
    for t in range(100000):
    
        rand = np.random.uniform(0, 1, 1)
        
        if rand > probability(dt):
            spike.append(0)
            dt = dt + ms
        else:
            spike.append(1)
            if(t%200 != 0):
                ISI.append(dt)
            dt = ms
            
            
            
    """
        spike가 나타난 형태를 볼때 '#' 풀어서 확인
        ISI 값이 얼마인지 확인할 때 '#' 풀어서 확인
    """     
    
    #print(spike)
    #print(ISI)
    
    
    
    """
    
        3) ISI분포도 확인
        
        
        1. 저장된 모든 ISI값을 조사하여 각 ISI값이 얼마나 나타났는지를 distribution[]에 저장
            (1~1000ms까지의 값만 확인하였음. 필요시  distribution 리스트 크기를 늘려서 사용)
        2. 저장된 distribution값은 나타난 횟수이므로 이를 % 비율로 바꿔준다
        3. x_lable은 ISI값으로 맞춰준다.
        4. matplotlib로 x : x_lable, y : distribution 로 분포도 그리고 출력
    
    """
   
    for j in ISI:
        if int(j*1000) < x_range:
            distribution[int(j*1000)] += 1
     
    
    for i in distribution:
        i = i / len(ISI) *100
    
    
    
    
    
    plt.plot(distribution)
    firing_rate += 10
x_lable = [i*ms for i in range(1,x_range+1)]
plt.show()



"""

    4) ISI값의 평균, 표준편차 구하기
    
    
    1. 평균구하기 : 저장한 ISI값을 모두 더하여 나타난 spike개수(=ISI개수)로 나눔
    2. 표준편차구하기 : 1.에서 구한 평균값을 각 ISI값에서 뺀것을 제곱한다. 이를 모두 더하여 spike개수(=ISI개수)로 나눔

"""

#평균구하기
sum1 = 0
for i in ISI:
    sum1 += i
mean = sum1 / len(ISI)

#표준편차구하기
sum1 = 0
for i in ISI:
    sum1 += (i - mean) * (i - mean) 
standard_deviation = sum1 / len(ISI)

#평균, 표준편차 출력
print("mean = {}".format(mean))
print("standard deviation = {}".format(standard_deviation))



"""

    5) autocorrelation 확인하기
    
    
    1. search_range : 확인하고자하는 범위 지정  (1s로 맞춰 놓았고 필요시 변경)
    2. Filter : spike리스트 중 0 ~ 1000ms를 Filter로 지정하여  일정 시간 이후(타우)의 spike함수(로우)와 Convolution한다.
                (Filter크기는 필요시 조절)
    3. x[t] : t ms 이후의 spike함수를 저장한다.
    4. 총 1000번, 즉 1초까지 Filter를 옮겼을 때 spike함수와 Filter와의 correlation을 보기 위해 x[t] 총 1000번 반복 저장
    5. correlation : correlation은 Filter함수와 spike함수의 convolution이므로 이를 행렬곱으로 표현함
                    Filter(1 x 1000 행렬) * spike함수(1000 x 1000 행렬)
                    결과 : correlation[t]는 Filter를 t ms만큼 이동했을때 convolution 값
                        -> t = 0일때 Filter와 Filter의 convolution이기 때문에 최대값을 가지고 나머지는 불규칙적인 값이 나옴
                            -> spike는 특정한 규칙을 가지고 나타나지 않고 랜덤하게 나타나는 것을 확인

"""

search_range = 1000
filter_size = 1000

Filter = spike[0:filter_size]
x = []
for t in range(search_range):
    x.append(spike[t: t+filter_size])
#spike[0:100]과의 autocorrelation이 1이 나오는 지 확인
correlation = np.dot(Filter,np.transpose(x))

plt.plot(correlation)
plt.show()


