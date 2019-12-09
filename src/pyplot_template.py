from matplotlib import pyplot as plt


# 이미지 표시 그래프의 구조를 만드는 함수
def plt_arch(plot_number, title, src):
    plt.subplot(plot_number), plt.imshow(src, cmap='gray')
    plt.title(title), plt.xticks([]), plt.yticks([])
