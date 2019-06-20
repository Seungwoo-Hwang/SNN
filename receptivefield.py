
import math
import numpy as np
import matplotlib.pyplot as plt


'''

    image 픽셀 정보 불러오기
    
        image load후 픽셀 최대 값인 255로 나눠준다
        -> spike의 세기를 0~1로 보기 때문에 input값을 0~1값 사이로 맞추기 위한 전처
        
'''

img_matrix = np.loadtxt('lena.txt')
img_matrix = img_matrix / 255



'''

    초기 설정
    
        1. field_size : ReceptiveField의 크기 (filter의 크기)   ---> 4, 8
        2. resize : ReceptiveField를 거치고 난 후 resize된 image크기
                    image를 field_size pixel씩 잘랐으므로 512 / field_size
                    
'''

#field_size = 4
field_size = 8
resize = int(512 / field_size)



'''

    GaborFunction으로 Receptive_field 구현
        
        1. sig_x, sig_y : GaborFunction에서 x,y의 표준편차
        2. k : GaborFunction에서 각 파수
        3. setha : 
        4. GaborFunction 함수 : 실제 V1세포가 실험적으로 가지고 있는 ReceptiveField의 weigth값 분포가 GaborDistribution임
        5. Receptive_field : Receptive_field를 (field_size x field_size) 크기의 2차원 리스트로 표현
                            4.에서 구현한 GarborFunction을 Receptive_field에 대입
                            filter의 중심을 (0,0)으로 두기 위해 x, y값에 각각 field_size/2를 
                                
'''

#sig_x = 0.2
#sig_y = 0.2
#k = 1
#setha = math.pi / 2

sig_x = 1
sig_y = 1
k = 1
setha = math.pi / 2

def GaborFunction(x, y):
    return 1/(2*math.pi*sig_x*sig_y)*math.exp(-x*x/(2*sig_x*sig_x) -y*y/(2*sig_y*sig_y))*math.cos(k*x - setha)


Receptive_field = np.empty((field_size,field_size))
for i in range(field_size):
    for j in range(field_size):
        Receptive_field[i][j] = GaborFunction(i-field_size/2, -j+field_size/2)

'''
    Receptive_field 확인
'''
#print(Receptive_field)
#plt.imshow(Receptive_field)



'''

    각 neuron의 firing_rate구하기
    
        1. result_matrix : 각 neuron의 fring_rate을 (512/field_size x 512/field_size)크기의 2차원 리스트에 저장할 것임
        2. mt : 원본 이미지를 (field_size x field_size)로 잘라서 임시 저장
        3. firing_rate : V1 cell에서 이미지 신호를 input으로 받았을 때 firing_rate은 각 ReceptiveField의 이미지와 담당 뉴런의 weigth의 곱의 합

'''
result_matrix = np.empty((resize, resize))


for i in range(0, resize):
    for j in range(0, resize):
        mt = img_matrix[i*field_size : (i+1)*field_size, j*field_size : (j+1)*field_size]
        result_matrix[i][j] = np.sum(np.multiply(Receptive_field, mt))



'''
    result저장 및 result확인
'''
#np.savetxt('result.txt', result_matrix)
print(result_matrix)
plt.imshow(result_matrix)