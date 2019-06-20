import numpy as np
import matplotlib.pyplot as plt

"""
    
    1) i가 constant일때

    초기 상수, 변수 설정
        
        문제에서 주어진 상수
        1. tau : Membrane Potential time constant로 40ms로 설정
        2. resist : Resistance of the leakage channel (R)로 100M옴으로 설정
        3. tref : Refactory time으로 10ms로 설정
        4. ures : Resting Potential로 0V로 설정
        5. uth : Threshold Potential for Spiking으로 50mV로 설정
        6. uhyper : Hyperpolarized Potential로 0V로 설정
                    spike발생시 potential이 초기화 되는 값
        7. dt : Simulation time-step으로 1ms로 설정
        
        추가 상수
        1. nt : 총 simulating 할 횟수로 10초를 dt로 나눈 값에 1을 더한 값
        2. i : t시간에서의 Input Current로 0.5, 0.6, 0.7, 0.8, 0.9, 1.0nA로 변경하면서 측정
        3. nA : i_cur가 nA단위 이므로 이를 위한 보정값
                    
"""

tau = 40 * 0.001
resist = 100 * 1000000
tref = 0.01
ures = 0;
uth = 50 * 0.001
uhyper = 0
dt = 0.001

tmax = 10
nt = 1 + int(tmax / dt)
i = [0.5, 0.6, 0.7, 0.8, 0.9, 1]
nA = 0.000000001



"""

    t시간에서 potential 구하기
        1. ut : t-1시간에서의 potential
        2. potential(ut) : t시간에서의 potential로 t-1시간의 potential이 ut에 저장되어 있으므로
                             ut를 인자로 받아서 t시간의 potential을 구하고 ut를 potential(ut)로 최신화
                             
                             potential을 구하기 위한 numerical solution 각각을 구현해 놨으므로 필요한것 사용
                             1> Explicit Method
                             2> Implicit Method
                             3> Crank-Nicolson Method

"""

def potential(ut):
    #return (1- dt/tau)*ut + dt/tau*resist*i_prev
    #return 1/(1+tau/dt) * (tau/dt*ut + resist*i_cur)
    return 1/(1/2 + tau/dt) * ((tau/dt - 1/2)*ut + resist/2*(i_prev + i_cur))



"""

    spike 발생시키기
        
        1. spike_rate = 매 i에서 spike_rate을 저장할 list
        2. ut : t시간에서의 potential을 구하기위해 t-1의 potential을 저장
                처음엔 ut를 ures(=0)으로 초기화
                내부 for문에서 ut = potential(ut)로 최신화 시켜준 뒤에 다음 step에 넘겨준다.
        3. i_cur : t시간에서의 Input Current로 0.5, 0.6, 0.7, 0.8, 0.9, 1.0nA로 변경하면서 측정
        4. i_prev : t-1시간에서의 Input Current로 i가 constant일 때에는 i_cur과 동일
        5. spike_time : spike가 일어난 시간을 저장해놓은 list  
        6. tstamp : spike가 일어난 뒤에 Refactory time을 적용하기 위함
                    time stamp로 spike가 일어난 시점의 시간을 임시저장한다. 따라서 spike가 일어나면 그 시점의 시간으로 최신화 시켜준다.
                    -100으로 초기화 시킴으로써 처음 시작할땐 Refactoring time 적용을 피해줌  
        7. potential(ut)가 uth보다 크고 => t시간에서의 potential이 threshold potential보다 크고
           (t - tstamp)*dt > tref보다 크면 => (t - tstamp)dt 시간이 Refactoring time보다 크다면 spike발생
           
           1> ut를 uhyper로 초기화
           2> tstamp를 spike 발생시간인 t로 최신화
           3> spike_time에 spike발생시간 저장
           4> 조건 만족 못할 시 ut를 potential(ut)로 최신화
        8. spike_rate list에 총 발생한 spike 개수를 10s으로 나눈 값(len(spike_time)/10)을 저장한다.
        
"""

spike_rate = []

for j in i:
    ut = ures
    i_cur = j*nA
    i_prev = j*nA
    spike_time = []
    tstamp = -100
    
    for t in range(1, nt):
        if (potential(ut) >= uth) and ((t - tstamp)*dt > tref):
            ut = uhyper
            tstamp = t;
            spike_time.append(t)
        else:
            ut = potential(ut)
            
        i_prev = i_cur
    
    spike_rate.append(len(spike_time) / tmax)



"""

    결과 확인
    
        1. 각 i에 따른 spike rate을 저장해 놓은 spike_rate을 확인한다.
        2. input current에 따른 spike_rate을 확인한다.

"""
print("1) Constant Current")
print(spike_rate)
plt.plot(i, spike_rate)
plt.ylabel("Spike Count Rate")
plt.xlabel("Input Current(nA)")
plt.show()
print()



"""

    2) i가 constant가 아닐 때
    
    변수, 상수 재초기화
        1. spike_rate list 초기화
        2. nt를 100초 동안의 횟수로 초기화
        3. 매 for문 시작시 ut, tstamp, spike_rate list 초기화
    
    spike 발생시키기
        1. i_cur과 i_prev가 normal distribution을 가지고 나타나므로 mean과 std지정 
        2. 처음 t=0일 때의 current값을 i_prev에다가 normal distribution으로 random하게 하나 지정
        3. 이후 1)번과 동일한 방법으로 spike를 발생시키나 처음 루프 시작시에 i_cur를 하나 random으로 뽑고
            루프 마지막에 i_prev를 i_cur로 최신화 시켜줌
        8. spike_rate list에 총 발생한 spike 개수를 100s으로 나눈 값(len(spike_time)/100)을 저장한다.

"""

spike_rate = []
tmax = 100
nt = 1 + int(tmax / dt)


m = [0.45, 0.47, 0.49, 0.5, 0.6, 0.7]         
std = 0.2 * nA




for mean in m:
    ut = ures
    spike_time = []
    tstamp = -100
    mean *= nA
    
    i_prev = np.random.normal(mean, std, 1)
    
    for t in range(1, nt):
        i_cur = np.random.normal(mean, std, 1)
        
        if (potential(ut) >= uth) and ((t - tstamp)*dt > tref):
            ut = uhyper
            tstamp = t;
            spike_time.append(t)
        else:
            ut = potential(ut)
            
        i_prev = i_cur
    
    spike_rate.append(len(spike_time) / tmax)
    
    
    '''
    distribution 구하기
    '''
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
    
    print("{}nA Result".format(mean / nA))
    print("mean = {}".format(average))
    print("std = {}".format(total/len(ISI)))
    
"""

    결과 확인
    
        1. 각 i에 따른 spike rate을 저장해 놓은 spike_rate을 확인한다.
        2. input current에 따른 spike_rate을 확인한다.

"""

print()
print("2) Normal Distribution Current")
print(spike_rate)
plt.plot(m, spike_rate)
plt.ylabel("Mean Spike Rate")
plt.xlabel("Mean Input Current(nA)")
plt.show()
print()



    


spike_rate = []
tmax = 10
nt = 1 + int(tmax / dt)

for mean in m:
    
    mean *= nA
    sum = 0
    for trial in range(100):
        ut = ures
        spike_time = []
        tstamp = -100
        
        
        i_prev = np.random.normal(mean, std, 1)
        
        for t in range(1, nt):
            i_cur = np.random.normal(mean, std, 1)
            
            if (potential(ut) >= uth) and ((t - tstamp)*dt > tref):
                ut = uhyper
                tstamp = t;
                spike_time.append(t)
            else:
                ut = potential(ut)
                
            i_prev = i_cur
        
        sum += len(spike_time)
        
        
    
    spike_rate.append(sum / (nt*100*dt))


print("3) Normal Distribution Current")
print(spike_rate)
plt.plot(m, spike_rate)
plt.ylabel("Firing Rate(Hz)")
plt.xlabel("Mean Input Current(nA)")
plt.show()
print()




