import matplotlib
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
import openpyxl
import skimage.io as io
import skimage.transform as trans
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['SimHei']  # 使用'SimHei'字体
matplotlib.rcParams['axes.unicode_minus'] = False

workbook = openpyxl.load_workbook('C:/Users/13183/PycharmProjects/pythonProject/左侧数据（整理）xlsx.xlsx')
sheetsource = workbook['原始温度数据']

for i in sheetsource.iter_rows(min_row=2,min_col=1, max_col=37):
    sheetname = str(i[0].value)
    sheetname = sheetname.split(' ')[0]
    sheetname = sheetname + '(17张图)'
    sheet = workbook[sheetname]

    for n in range(0,17):
        print(n)
        title = str(sheetname) + str(f" {3500+n*100}mm")#(3500-5100)

        temperatures = []
        for row in sheet[f'L{3+n*4}':f'BF{6+n*4}']:#['L7':'BF10'] ['L(4n-1)':'BF(4n+2)']
            temperatures.append([cell.value for cell in row])
        for row in sheet[f'L{3+n*4}':f'BF{3+n*4}']:#['L7':'BF7'] ['L(4n-1)':'BF(4n-1)']
            temperatures.append([cell.value for cell in row])
        temperatures=np.array(temperatures)

        # 角度和半径
        theta = np.array([0, np.pi/2, np.pi, 3*np.pi/2, 2*np.pi])  # 角度
        r = np.linspace(0, 4600, temperatures.shape[1])  # 半径

        # 创建一个网格用于插值
        theta_grid, r_grid = np.meshgrid(np.linspace(0, 2*np.pi, 360), r)

        # 对温度数据进行插值
        temperatures_grid = griddata((np.repeat(theta, len(r)), np.tile(r, len(theta))), temperatures.flatten(), (theta_grid, r_grid), method='cubic')

        # 创建一个极坐标图形
        fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})

        # 绘制等值线图
        contour = ax.contourf(theta_grid, r_grid, temperatures_grid,levels=np.linspace(-30, 30, 60))

        # 添加颜色条
        fig.colorbar(contour, label='温度 (°C)')

        # 添加红色的零度线
        contour = ax.contour(theta_grid, r_grid, temperatures_grid, levels=[0], colors='red')

        # 在半径为1350的圆上画7个等间距的小圆圈
        circle_theta = np.linspace(0, 2 * np.pi, 1)
        circle_r = np.repeat(0, 1)
        ax.scatter(circle_theta, circle_r, edgecolors='black', facecolors='none')

        circle_theta = np.linspace(0, 2 * np.pi, 8)
        circle_r = np.repeat(1350, 8)
        ax.scatter(circle_theta, circle_r, edgecolors='black', facecolors='none')

        circle_theta = np.linspace(0, 2 * np.pi, 16)
        circle_r = np.repeat(2700, 16)
        ax.scatter(circle_theta, circle_r, edgecolors='black', facecolors='none')

        circle_theta = np.linspace(0, 2 * np.pi, 35)
        circle_r = np.repeat(3900, 35)
        ax.scatter(circle_theta, circle_r, edgecolors='black', facecolors='none')

        # 添加标题
        plt.title(title)

        # 添加图例
        legend_elements = [matplotlib.lines.Line2D([0], [0], color='red', lw=2, label='零度线'),
                           matplotlib.lines.Line2D([0], [0], marker='o',color='black', markerfacecolor='none', markersize=10,linestyle='None',
                                                   label='冻结管')]
        ax.legend(handles=legend_elements, loc='upper right',bbox_to_anchor=(1.2,1.15))

        # 保存图像
        plt.savefig(f"./可视化/{3500+n*100}mm/" + title + '.png',)
    # plt.show()