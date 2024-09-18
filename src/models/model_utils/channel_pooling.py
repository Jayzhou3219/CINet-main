import torch
import numpy as np
from torch.autograd import Variable
from torch.nn.modules.module import Module
import torch.nn.functional as F



def create_gaussian_heatmap_template(size, kernlen=81, nsig=4, normalize=True):
	""" Create a big gaussian heatmap template to later get patches out """
	template = np.zeros([size, size])
	kernel = gkern(kernlen=kernlen, nsig=nsig)
	m = kernel.shape[0]
	x_low = template.shape[1] // 2 - int(np.floor(m / 2))
	x_up = template.shape[1] // 2 + int(np.ceil(m / 2))
	y_low = template.shape[0] // 2 - int(np.floor(m / 2))
	y_up = template.shape[0] // 2 + int(np.ceil(m / 2))
	template[y_low:y_up, x_low:x_up] = kernel
	if normalize:
		template = template / template.max()
	return template


def gkern(kernlen=31, nsig=4):
	"""	creates gaussian kernel with side length l and a sigma of sig """
	ax = np.linspace(-(kernlen - 1) / 2., (kernlen - 1) / 2., kernlen)
	xx, yy = np.meshgrid(ax, ax)
	kernel = np.exp(-0.5 * (np.square(xx) + np.square(yy)) / np.square(nsig))
	return kernel / np.sum(kernel)


def get_patch(template, traj, H, W):
	x = np.round(abs(traj[:, 0])).astype('int')  # np.round(data,decimal):取整函数
	y = np.round(abs(traj[:, 1])).astype('int')

	x_low = abs(template.shape[1] // 2 - x)  # .shape[1]:输出矩阵的列数  # // : 取整除,返回商的整数部分（向下取整）
	x_up = abs(template.shape[1] // 2 + W - x)
	y_low = abs(template.shape[0] // 2 - y)  # .shape[0]:输出矩阵的行数
	y_up = abs(template.shape[0] // 2 + H - y)

	patch = [template[y_l:y_u, x_l:x_u] for x_l, x_u, y_l, y_u in zip(x_low, x_up, y_low, y_up)]

	return patch


def create_dist_mat(size, normalize=True):
	""" Create a big distance matrix template to later get patches out """
	middle = size // 2
	dist_mat = np.linalg.norm(np.indices([size, size]) - np.array([middle, middle])[:,None,None], axis=0)  # np.linalg.norm(x,ord=None,axis=0):axis=0表示,按列向量处理，求多个列向量的范数
	if normalize:                                             # np.indices的作用就是返回一个给定形状数组的序号网格数组
		dist_mat = dist_mat / dist_mat.max() * 2
	return dist_mat


class my_channel_pooling(Module):

    def __init__(self, kernel_size, stride):
        super(my_channel_pooling, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride
    
    def forward(self, input):

        input = input.transpose(3, 1)  # 如维度为(32, 8, 64, 64) 交换为 (32, 64, 64, 8)
        
        input = F.max_pool2d(input, self.kernel_size, self.stride)  # shape:(32, 64, 64, 8)
        
        input = input.transpose(3, 1).contiguous()  # shape:(32, 8, 64, 64)
        
        return input
