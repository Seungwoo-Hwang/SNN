

import math
import numpy as np
import matplotlib.pyplot as plt

'''
    
    1) 변수
    
        1. firing_rate1, firing_rate2 : Npre1, Npre2의 FiringRate(Hz)
                -> Npre1과 Npre2의 spike 발생 확률과 관련 (firing_rate * dt)
        2. num_ch1, num_ch2 : Npre1, Npre2의 AMPAR Channel 개수
                -> Npre1, Npre2로 부터 전달받는 current와 관련(Nc * c * Ps)
        3. c_max : maximum synpatic current
                -> Npre1, Npre2로 부터 전달받는 current와 관련(Nc * c * Ps)
        4. p_max : P0
                -> presynaptic에서 spike 발생했을 때 Ps에 관련
        5. beta_s : Channel Closing Rate (1/tau_s)
                -> Decay시의 Ps에 관련
        6. tau, resist, tref, resting_potential, uth, uhyper, dt는 task3에서 사용한 것과 같음
            모르겠으면 다시 보길 바람
        
'''

#새로운 상수
firing_rate1 = 20
firing_rate2 = 40

num_ch1 = 1000
num_ch2 = 2000
nA = 0.000000001
c_max = 0.001 * nA
p_max = 0.4
tau_s = 15*0.001
beta_s = 1/tau_s
#여기까지

tau = 40 * 0.001
resist = 100 * 1000000
tref = 10 * 0.001
ures = 0
uth = 50 * 0.001
uhyper = 0
dt = 0.001




'''

    2) potential 함수 설정

        여기도 task3랑 같음

'''

def potential(ut):
    #return (1- dt/tau)*ut + dt/tau*resist*i_prev
    #return 1/(1+tau/dt) * (tau/dt*ut + resist*i_cur)
    return 1/(1/2 + tau/dt) * ((tau/dt - 1/2)*ut + resist/2*(i_prev + i_cur))



'''

    3) spike 발생
        
        추가 변수
        1. prob1, prob2 : probability of channel open of Npre1, Nprev2
                -> Npre1, Npre2로 부터 전달받는 current와 관련(Nc * c * Ps)
        2. 나머지 변수들은 task3와 같음
        
        spike발생
        1. task1에서 처럼 random.uniform으로 rand값 하나를 뽑아서 
            Npre1, Npre2가 spike할 확률(a*dt)보다 작은 값이 나오면 spike가 일어남으로 인한 prob1, prob2값 변경
            더 큰값이 나오면 spike가 일어나지 않았으므로 prob1, prob2는 decay시킴        
        2. i_noise : normal distribution의 Noise Current이다.
        3. i_cur : task3에서는 값을 우리가 직접 설정해주었지만 여기선 Npre1, Npre2의 spike로 인한 current가 된다.
                    i = 각 pre시냅스에서 오는 (Nc * c * Ps)값의 합
        4. task3와 같이 subthreshold potential이 threshold potential넘으면 spike, 안넘으면 ut최신화 진행
        
'''

#새로운 변수
prob1 = 0;
prob2 = 0;
#여기까지

tmax = 10
nt = 1 + int(tmax / dt)

ut = ures
spike_time = []
tstamp = -100

i_prev = num_ch1*c_max*prob1 + num_ch2*c_max*prob2

for t in range(1, nt):
    #추가 과정
    rand = np.random.uniform(0, 1, 1)
    if rand < firing_rate1 * dt:
        prob1 = prob1 + p_max*(1-prob1)
    else:
        prob1 = prob1 * math.exp(-beta_s * dt)
              
    rand = np.random.uniform(0, 1, 1)
    if rand < firing_rate2 * dt:
        prob2 = prob2 + p_max*(1-prob2)
    else:
        prob2 = prob2 * math.exp(-beta_s * dt)
        
        
    i_noise = np.random.normal(0.1*nA, 0.2*nA , 1)
    i_cur = num_ch1*c_max*prob1 + num_ch2*c_max*prob2 + i_noise
    #여기까지
    
    if (potential(ut) >= uth) and ((t - tstamp)*dt > tref):
        ut = uhyper
        tstamp = t;
        spike_time.append(t)
    else:
        ut = potential(ut)
    
    i_prev = i_cur
   
    
    
'''

    4) 결과확인
    
        1. Spike Count Rate : 전체 spike 발생 횟수 / 측정시간 
        2. ISI 분포 확인 : task3에서 했던것 처럼 진행

'''

print(len(spike_time) / 10)


x_lable = [i*0.001 for i in range(1,1001)]
ISI = []
distribution = [0 for i in range(1000)]
total = 0

for k in range(1, len(spike_time)):
    ISI.append((spike_time[k] - spike_time[k-1]) * 0.001)

for j in ISI:
    if int(j*1000) < 1000:
        distribution[int(j*1000)] += 1
    total += j

average = total / len(ISI)
total = 0
for j in ISI:
    total += (j-average)*(j-average)
 
plt.scatter(x_lable, distribution)
plt.show()

print("mean = {}".format(average))
print("std = {}".format(total/len(ISI)))