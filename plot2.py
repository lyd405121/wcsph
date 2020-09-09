import numpy as np
import matplotlib.pyplot as plt
 
#定义x、y散点坐标
x    = [50.0, 500.0, 5000.0]
num1 = [2.35,3.7,8.7]
num2 = [2.9,5.5,13.1]

y1 = np.array(num1)
y2 = np.array(num2)

#用2次多项式拟合
f1 = np.polyfit(x, y1, 2)
p1 = np.poly1d(f1)
yvals1 = p1(x)  #拟合y值
 
f2 = np.polyfit(x, y2, 2)
p2 = np.poly1d(f2)
yvals2 = p2(x)


fig1, ax1 = plt.subplots()
ax1.set_xscale("log")
ax1.set_xlim(1e1, 1e4)



#绘图
plot1 = plt.plot(x, y1, 's')
plot2 = plt.plot(x, yvals1, 'r')
plot3 = plt.plot(x, y2, 's')
plot4 = plt.plot(x, yvals2, 'g')

plt.text(1000.0, 5.0,'precondition_cg',fontsize=10)
plt.text(800.0, 10.0,'cg',fontsize=10)

plt.xlabel('viscosity factor')
plt.ylabel('average iter')
plt.legend(loc=2, bbox_to_anchor=(1.05,1.0),borderaxespad = 0.)
plt.title('average iter')


plt.show()
