import matplotlib.pyplot as plt
import matplotlib

matplotlib.rcParams['font.sans-serif'] = ['SimHei']
matplotlib.rcParams['axes.unicode_minus'] = False

price = [0.0001, 0.001, 0.005]
"""
绘制水平条形图方法barh
参数一：y轴
参数二：x轴
"""
plt.barh(range(3), price, height=0.7, color='steelblue', alpha=0.8)      # 从下往上画
plt.yticks(range(3), ['sesph', 'pcisph', 'iisph'])
plt.xlim(0.0,0.01)
plt.xlabel("second")
plt.title("max step time size for each algrithm")
for x, y in enumerate(price):
    plt.text(y + 0.2, x - 0.1, '%s' % y)
plt.show()