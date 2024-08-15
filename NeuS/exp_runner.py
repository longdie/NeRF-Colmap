import os
import time
import logging
import argparse
import numpy as np
import cv2 as cv
import trimesh # trimesh: 用于处理和操作三维网格。
import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter # 用于在 TensorBoard 中记录训练过程。
from shutil import copyfile
from icecream import ic
from tqdm import tqdm
from pyhocon import ConfigFactory
from models.dataset_mvdiff import Dataset
from models.fields import RenderingNetwork, SDFNetwork, SingleVarianceNetwork, NeRF
from models.renderer import NeuSRenderer
import pdb
import math


def ranking_loss(error, penalize_ratio=0.7, type='mean'):
    """
    ranking_loss 是一个自定义损失函数，按误差排序，并仅对前 penalize_ratio 百分比的误差计算损失。
    error: 输入误差
    penalize_ratio: 惩罚的比例，即取误差中最小的部分进行计算
    type: 决定返回的损失类型，可以是 mean 或 sum
    """
    error, indices = torch.sort(error)
    # only sum relatively small errors
    s_error = torch.index_select(error, 0, index=indices[: int(penalize_ratio * indices.shape[0])])
    if type == 'mean':
        return torch.mean(s_error)
    elif type == 'sum':
        return torch.sum(s_error)


class Runner:
    """
    Runner 类用于管理模型的训练、验证和保存
    """
    def __init__(self, conf_path, mode='train', case='CASE_NAME', is_continue=False, data_dir=None):
        """
        __init__ 方法是类的构造函数，初始化运行环境和模型参数。
        conf_path: 配置文件路径
        mode: 模式，默认为 'train'
        case: 案例名称，用于替换配置文件中的占位符
        is_continue: 是否从断点继续训练
        data_dir: 数据目录路径
        """
        self.device = torch.device('cuda')
        
        # 打开并读取配置文件，将其中的占位符 'CASE_NAME' 替换为具体的案例名称
        self.conf_path = conf_path
        f = open(self.conf_path)
        conf_text = f.read()
        conf_text = conf_text.replace('CASE_NAME', case)
        f.close()

        # 使用 ConfigFactory.parse_string 解析配置文件内容，生成配置字典 self.conf
        self.conf = ConfigFactory.parse_string(conf_text)
        self.conf['dataset']['data_dir'] = data_dir
        self.conf['dataset.data_dir'] = self.conf['dataset.data_dir'].replace('CASE_NAME', case)
        # 设置实验目录 self.base_exp_dir，用于存储模型和日志文件
        self.base_exp_dir = self.conf['general.base_exp_dir']
        os.makedirs(self.base_exp_dir, exist_ok=True)
        # 使用 Dataset 类加载数据集，并用 DataLoader 封装为可迭代对象。
        self.dataset = Dataset(self.conf['dataset'])
        self.dataloader = torch.utils.data.DataLoader(
            self.dataset,
            # 设置 batch_size 为配置文件中的值，并启动多线程加载数据（num_workers=64）
            batch_size=self.conf['train']['batch_size'],
            shuffle=True,
            num_workers=64,
        )
        # 初始化训练步骤 self.iter_step 为 1。
        self.iter_step = 1

        # Training parameters
        self.end_iter = self.conf.get_int('train.end_iter') # 训练的总迭代次数
        self.save_freq = self.conf.get_int('train.save_freq') # 模型保存频率
        self.report_freq = self.conf.get_int('train.report_freq') # 训练信息报告频率
        self.val_freq = self.conf.get_int('train.val_freq') #  验证频率
        self.val_mesh_freq = self.conf.get_int('train.val_mesh_freq') # 验证网格的频率
        self.batch_size = self.conf.get_int('train.batch_size') # 批次大小
        self.validate_resolution_level = self.conf.get_int('train.validate_resolution_level') # 验证图像的分辨率等级
        self.learning_rate = self.conf.get_float('train.learning_rate') # 初始学习率
        self.learning_rate_alpha = self.conf.get_float('train.learning_rate_alpha') # 学习率衰减的控制参数
        self.use_white_bkgd = self.conf.get_bool('train.use_white_bkgd') # 是否使用白色背景
        self.warm_up_end = self.conf.get_float('train.warm_up_end', default=0.0) # 学习率预热结束时的迭代次数
        self.anneal_end = self.conf.get_float('train.anneal_end', default=0.0) # 学习率退火结束时的迭代次数

        # 读取各个损失项的权重参数，并设置模型的其他状态变量。
        self.color_weight = self.conf.get_float('train.color_weight')
        self.igr_weight = self.conf.get_float('train.igr_weight')
        self.mask_weight = self.conf.get_float('train.mask_weight')
        self.normal_weight = self.conf.get_float('train.normal_weight')
        self.sparse_weight = self.conf.get_float('train.sparse_weight')
        self.is_continue = is_continue
        self.mode = mode
        self.model_list = []
        self.writer = None

        # Networks: 初始化多个神经网络模块
        params_to_train_slow = []
        self.nerf_outside = NeRF(**self.conf['model.nerf']).to(self.device)
        self.sdf_network = SDFNetwork(**self.conf['model.sdf_network']).to(self.device)
        self.deviation_network = SingleVarianceNetwork(**self.conf['model.variance_network']).to(self.device)
        self.color_network = RenderingNetwork(**self.conf['model.rendering_network']).to(self.device)
        # params_to_train += list(self.nerf_outside.parameters())
        # 将 SDFNetwork 和 SingleVarianceNetwork 的参数加入 params_to_train_slow，这些参数会以较慢的学习率进行更新
        params_to_train_slow += list(self.sdf_network.parameters())
        params_to_train_slow += list(self.deviation_network.parameters())
        # params_to_train += list(self.color_network.parameters())

        # 定义优化器 Adam，以不同的学习率分别更新缓慢训练的参数和渲染网络的参数
        self.optimizer = torch.optim.Adam(
            [{'params': params_to_train_slow}, {'params': self.color_network.parameters(), 'lr': self.learning_rate * 2}], lr=self.learning_rate
        )
        # 初始化 NeuSRenderer，用于执行实际的渲染操作
        self.renderer = NeuSRenderer(
            self.nerf_outside, self.sdf_network, self.deviation_network, self.color_network, **self.conf['model.neus_renderer']
        )

        # Load checkpoint
        latest_model_name = None # 初始化一个变量，用于存储找到的最新检查点文件的名称
        if is_continue: # 是否需要从已有的检查点继续训练
            # 获取 checkpoints 目录中的所有文件名列表。这个目录通常用来保存训练过程中生成的检查点文件
            model_list_raw = os.listdir(os.path.join(self.base_exp_dir, 'checkpoints'))
            model_list = [] # 初始化一个空列表 model_list，用于存储满足条件的模型文件
            # 遍历 model_list_raw 中的每个文件名
            for model_name in model_list_raw: 
                # 检查文件名是否以 .pth 结尾（即模型文件的标准扩展名）
                if model_name[-3:] == 'pth' and int(model_name[5:-4]) <= self.end_iter:
                    # 将 model_name 添加到 model_list 中
                    model_list.append(model_name)
            # 对符合条件的模型文件名进行排序。通常，文件名中包含的迭代次数决定了排序顺序，这样最后一个文件就是训练中最近生成的检查点文件
            model_list.sort()
            # 将 model_list 中的最后一个文件名赋值给 latest_model_name，这是最新的检查点文件
            latest_model_name = model_list[-1]

        if latest_model_name is not None: # 检查是否找到了符合条件的检查点文件
            logging.info('Find checkpoint: {}'.format(latest_model_name)) # # 记录找到的检查点文件名
            # 调用 self.load_checkpoint() 方法加载找到的检查点文件。此方法通常会恢复模型的参数、优化器状态等
            self.load_checkpoint(latest_model_name)

        # Backup codes and configs for debug
        if self.mode[:5] == 'train': # 检查当前模式是否为训练模式
            self.file_backup() # 调用 file_backup 方法，将当前的代码和配置文件备份

    def train(self):
        # SummaryWriter 用于将训练中的各种统计数据记录到指定日志目录中，便于后续使用工具（如TensorBoard）可视化
        self.writer = SummaryWriter(log_dir=os.path.join(self.base_exp_dir, 'logs'))
        self.update_learning_rate() # 更新学习率，通常是根据训练进度调整学习率
        res_step = self.end_iter - self.iter_step # 计算剩余的训练步数，即从当前步到设定的结束步之间的步数
        image_perm = self.get_image_perm() # 获取图像的随机排列，用于在训练过程中随机选择图像

        # 计算需要训练的epoch数量
        num_train_epochs = math.ceil(res_step / len(self.dataloader))

        print("training ", num_train_epochs, " epoches") # 打印需要训练的epoch数量

        for epoch in range(num_train_epochs):
            # 进入训练的epoch循环，并打印当前epoch
            print("epoch ", epoch)
            for iter_i, data in enumerate(self.dataloader):
                # 进入每个epoch的训练循环，通过 dataloader 获取数据并将其移动到GPU上
                data = data.cuda()
                # 数据切片操作
                rays_o, rays_d, true_rgb, mask, true_normal, cosines = (
                    data[:, :3], # 光线的原点 rays_o
                    data[:, 3:6], # 方向 rays_d
                    data[:, 6:9], # 真实的 RGB 值 true_rgb
                    data[:, 9:10], # 掩码 mask
                    data[:, 10:13], # 真实法线 true_normal
                    data[:, 13:], # 角度余弦 cosines
                )
                
                near, far = self.dataset.get_near_far() # 获取光线的近远平面值，用于确定光线在场景中的射程范围
                # 处理背景颜色
                background_rgb = None
                if self.use_white_bkgd:
                    background_rgb = torch.ones([1, 3])
                # 掩码处理
                if self.mask_weight > 0.0:
                    mask = (mask > 0.5).float()
                else:
                    mask = torch.ones_like(mask)
                # 余弦值处理:将余弦值大于 -0.1 的部分设为 0，然后将掩码与余弦值的条件结合，生成一个新的掩码
                cosines[cosines > -0.1] = 0
                mask = ((mask > 0) & (cosines < -0.1)).to(torch.float32)
                # 计算掩码值的和，并加上一个小值 1e-5 以避免除零错误
                mask_sum = mask.sum() + 1e-5
                # 使用渲染器 self.renderer 渲染场景，并返回渲染结果
                render_out = self.renderer.render(
                    rays_o, rays_d, near, far, background_rgb=background_rgb, cos_anneal_ratio=self.get_cos_anneal_ratio()
                )
                # 渲染输出分割
                color_fine = render_out['color_fine'] # 渲染的颜色
                s_val = render_out['s_val'] # 统计值
                cdf_fine = render_out['cdf_fine'] # 累积分布函数
                gradient_error = render_out['gradient_error'] # 梯度误差
                weight_max = render_out['weight_max'] # 最大权重 
                weight_sum = render_out['weight_sum'] # 权重和

                # Loss: 计算颜色误差 color_errors，然后用自定义的 ranking_loss 函数计算颜色损失 color_fine_loss
                color_errors = (color_fine - true_rgb).abs().sum(dim=1)
                color_fine_loss = ranking_loss(color_errors[mask[:, 0] > 0])
                # 计算峰值信噪比（PSNR），它是一个衡量重建图像质量的指标
                psnr = 20.0 * torch.log10(1.0 / (((color_fine - true_rgb) ** 2 * mask).sum() / (mask_sum * 3.0)).sqrt())
                
                eikonal_loss = gradient_error # 梯度误差
                # 掩码损失 mask_loss 是基于二值交叉熵和 ranking_loss 计算的
                mask_errors = F.binary_cross_entropy(weight_sum.clip(1e-3, 1.0 - 1e-3), mask, reduction='none')
                mask_loss = ranking_loss(mask_errors[:, 0], penalize_ratio=0.8)

                def feasible(key):
                    return (key in render_out) and (render_out[key] is not None)

                # calculate normal loss 法线损失
                # 计算法线损失 normal_loss，它依赖于法线的余弦相似度、掩码以及一个自定义的惩罚函数
                n_samples = self.renderer.n_samples + self.renderer.n_importance
                normals = render_out['gradients'] * render_out['weights'][:, :n_samples, None]
                if feasible('inside_sphere'):
                    normals = normals * render_out['inside_sphere'][..., None]
                normals = normals.sum(dim=1)
 
                normal_errors = 1 - F.cosine_similarity(normals, true_normal, dim=1)
                normal_errors = normal_errors * torch.exp(cosines.abs()[:, 0]) / torch.exp(cosines.abs()).sum()
                normal_loss = ranking_loss(normal_errors[mask[:, 0] > 0], penalize_ratio=0.9, type='sum')
                # 获取稀疏损失 sparse_loss，通常与正则化相关
                sparse_loss = render_out['sparse_loss']
                # 总损失计算
                loss = (
                    color_fine_loss * self.color_weight
                    + eikonal_loss * self.igr_weight
                    + sparse_loss * self.sparse_weight
                    + mask_loss * self.mask_weight
                    + normal_loss * self.normal_weight
                )
                # 反向传播与优化
                # 清零梯度，计算损失的梯度并执行优化器步骤，更新模型权重
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                # 记录损失和统计数据到日志中
                self.writer.add_scalar('Loss/loss', loss, self.iter_step)
                self.writer.add_scalar('Loss/color_loss', color_fine_loss, self.iter_step)
                self.writer.add_scalar('Loss/eikonal_loss', eikonal_loss, self.iter_step)
                self.writer.add_scalar('Statistics/s_val', s_val.mean(), self.iter_step)
                self.writer.add_scalar('Statistics/cdf', (cdf_fine[:, :1] * mask).sum() / mask_sum, self.iter_step)
                self.writer.add_scalar('Statistics/weight_max', (weight_max * mask).sum() / mask_sum, self.iter_step)
                self.writer.add_scalar('Statistics/psnr', psnr, self.iter_step)
                # 日志打印与验证、检查点保存
                if self.iter_step % self.report_freq == 0:
                    print(self.base_exp_dir)
                    print(
                        'iter:{:8>d} loss = {:4>f} color_ls = {:4>f} eik_ls = {:4>f} normal_ls = {:4>f} mask_ls = {:4>f} sparse_ls = {:4>f} lr={:5>f}'.format(
                            self.iter_step,
                            loss,
                            color_fine_loss,
                            eikonal_loss,
                            normal_loss,
                            mask_loss,
                            sparse_loss,
                            self.optimizer.param_groups[0]['lr'],
                        )
                    )
                    print('iter:{:8>d} s_val = {:4>f}'.format(self.iter_step, s_val.mean()))
                if self.iter_step % self.val_mesh_freq == 0:
                    self.validate_mesh(resolution=256)
                # 更新学习率和图像排列
                self.update_learning_rate()

                self.iter_step += 1

                if self.iter_step % self.val_freq == 0:
                    self.validate_image(idx=0)
                    self.validate_image(idx=1)
                    self.validate_image(idx=2)
                    self.validate_image(idx=3)

                if self.iter_step % self.save_freq == 0:
                    self.save_checkpoint()

                if self.iter_step % len(image_perm) == 0:
                    image_perm = self.get_image_perm()

    def get_image_perm(self):
        """
        生成一个随机排列的图像索引序列
        该函数返回一个 torch.Tensor，包含从 0 到 self.dataset.n_images - 1 的随机排列整数
        """
        return torch.randperm(self.dataset.n_images)

    def get_cos_anneal_ratio(self):
        """
        根据当前的训练进度，返回一个用于调整余弦衰减的比例值
        self.anneal_end 是一个预设的参数，表示余弦衰减过程结束时的迭代步数。它决定了衰减的时间跨度
        函数最终返回一个在 0.0 到 1.0 之间的值，该值随着 self.iter_step 的增加逐渐从 0.0 增长到 1.0，用于调整训练过程中的余弦衰减比例
        """
        if self.anneal_end == 0.0: # 返回 1.0 表示没有进行任何衰减，余弦衰减比例始终保持最大值
            return 1.0
        else:
            # self.iter_step / self.anneal_end 计算当前迭代步数相对于衰减结束步数的比例
            return np.min([1.0, self.iter_step / self.anneal_end])

    def update_learning_rate(self):
        '''
        根据训练的进展情况动态地调整学习率
        1. Warm-up阶段的学习率调整
        2. 热身阶段后的学习率调整
        '''
        # 这里判断当前的训练步数 self.iter_step 是否小于 self.warm_up_end，即判断是否处于训练的“热身”阶段（Warm-up Phase）
        if self.iter_step < self.warm_up_end: 
            # 学习率的调整因子 learning_factor 将根据当前步数线性增加
            learning_factor = self.iter_step / self.warm_up_end
        # 开始进入实际训练阶段，此时学习率因子 learning_factor 由一个余弦退火函数（cosine annealing function）控制
        else:
            alpha = self.learning_rate_alpha
            # 计算当前步数在热身阶段结束后所占的进度比例 progress，即训练的相对进度
            progress = (self.iter_step - self.warm_up_end) / (self.end_iter - self.warm_up_end)
            # 学习率因子 learning_factor 根据余弦函数计算
            learning_factor = (np.cos(np.pi * progress) + 1.0) * 0.5 * (1 - alpha) + alpha

        # 更新学习率
        for g in self.optimizer.param_groups:
            g['lr'] = self.learning_rate * learning_factor

    def file_backup(self):
        """
        这个函数的作用是备份指定目录中的Python文件以及配置信息，方便以后恢复或者追踪代码和配置的历史记录
        """
        dir_lis = self.conf['general.recording'] # 获取备份目录列表
        # 创建主备份目录
        os.makedirs(os.path.join(self.base_exp_dir, 'recording'), exist_ok=True)
        # 遍历并创建每个目录的备份
        for dir_name in dir_lis:
            cur_dir = os.path.join(self.base_exp_dir, 'recording', dir_name)
            os.makedirs(cur_dir, exist_ok=True)
            files = os.listdir(dir_name)
            for f_name in files:
                if f_name[-3:] == '.py':
                    copyfile(os.path.join(dir_name, f_name), os.path.join(cur_dir, f_name))
        # 备份配置文件
        copyfile(self.conf_path, os.path.join(self.base_exp_dir, 'recording', 'config.conf'))

    def load_checkpoint(self, checkpoint_name):
        """
        这个函数的作用是加载一个训练过程中的检查点（checkpoint），恢复模型、优化器以及训练的步数（iteration step）等状态，使训练能够从中断的地方继续
        """
        # 加载检查点文件
        checkpoint = torch.load(os.path.join(self.base_exp_dir, 'checkpoints', checkpoint_name), map_location=self.device)
        # 恢复模型的状态
        self.nerf_outside.load_state_dict(checkpoint['nerf'])
        self.sdf_network.load_state_dict(checkpoint['sdf_network_fine'])
        self.deviation_network.load_state_dict(checkpoint['variance_network_fine'])
        self.color_network.load_state_dict(checkpoint['color_network_fine'])
        # 恢复优化器的状态
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        # 恢复训练步数
        self.iter_step = checkpoint['iter_step']
        # 记录日志信息
        logging.info('End')

    def save_checkpoint(self):
        """
        这个函数的主要作用是将当前训练的状态保存为一个检查点（checkpoint），以便后续可以从这个状态继续训练
        """
        checkpoint = {
            'nerf': self.nerf_outside.state_dict(),
            'sdf_network_fine': self.sdf_network.state_dict(),
            'variance_network_fine': self.deviation_network.state_dict(),
            'color_network_fine': self.color_network.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'iter_step': self.iter_step,
        }
        # 创建存储检查点的目录
        os.makedirs(os.path.join(self.base_exp_dir, 'checkpoints'), exist_ok=True)
        torch.save(checkpoint, os.path.join(self.base_exp_dir, 'checkpoints', 'ckpt_{:0>6d}.pth'.format(self.iter_step)))

    def validate_image(self, idx=-1, resolution_level=-1):
        """
        这个函数的主要功能是对模型生成的图像和相关输出进行验证，并将这些结果保存为图像文件。它会从数据集中随机选择或指定一个图像索引 idx，然后生成相应的射线，
        并通过渲染器计算得到颜色、法线和权重掩码等输出。最终，结果会被保存到本地目录中
        """
        # 图像索引 idx 的设置
        if idx < 0:
            idx = np.random.randint(self.dataset.n_images)

        print('Validate: iter: {}, camera: {}'.format(self.iter_step, idx))
        # 设置分辨率级别
        if resolution_level < 0:
            resolution_level = self.validate_resolution_level
        # 生成射线:根据指定的图像索引 idx 和分辨率级别 resolution_level，生成射线的原点 rays_o 和方向 rays_d。然后，将这些射线按批次大小分割为多个小批次，以便后续处理
        rays_o, rays_d = self.dataset.gen_rays_at(idx, resolution_level=resolution_level)
        H, W, _ = rays_o.shape
        rays_o = rays_o.reshape(-1, 3).split(self.batch_size)
        rays_d = rays_d.reshape(-1, 3).split(self.batch_size)
        # 初始化输出列表
        out_rgb_fine = []
        out_normal_fine = []
        out_mask = []
        # 遍历射线批次并渲染
        for rays_o_batch, rays_d_batch in zip(rays_o, rays_d):
            near, far = self.dataset.get_near_far()
            background_rgb = torch.ones([1, 3]) if self.use_white_bkgd else None

            render_out = self.renderer.render(
                rays_o_batch, rays_d_batch, near, far, cos_anneal_ratio=self.get_cos_anneal_ratio(), background_rgb=background_rgb
            )

            def feasible(key):
                return (key in render_out) and (render_out[key] is not None)
            # 保存颜色、法线和掩码输出
            if feasible('color_fine'):
                out_rgb_fine.append(render_out['color_fine'].detach().cpu().numpy())
            if feasible('gradients') and feasible('weights'):
                n_samples = self.renderer.n_samples + self.renderer.n_importance
                normals = render_out['gradients'] * render_out['weights'][:, :n_samples, None]
                if feasible('inside_sphere'):
                    normals = normals * render_out['inside_sphere'][..., None]
                normals = normals.sum(dim=1).detach().cpu().numpy()
                out_normal_fine.append(normals)

            if feasible('weight_sum'):
                out_mask.append(render_out['weight_sum'].detach().clip(0, 1).cpu().numpy())

            del render_out
        
        img_fine = None
        if len(out_rgb_fine) > 0:
            img_fine = (np.concatenate(out_rgb_fine, axis=0).reshape([H, W, 3, -1]) * 256).clip(0, 255)

        mask_map = None
        if len(out_mask) > 0:
            mask_map = (np.concatenate(out_mask, axis=0).reshape([H, W, -1]) * 256).clip(0, 255)

        normal_img = None
        if len(out_normal_fine) > 0:
            normal_img = np.concatenate(out_normal_fine, axis=0)
            rot = np.linalg.inv(self.dataset.pose_all[idx, :3, :3].detach().cpu().numpy())
            normal_img = (np.matmul(rot[None, :, :], normal_img[:, :, None]).reshape([H, W, 3, -1]) * 128 + 128).clip(0, 255)
        
        # 处理和保存图像
        os.makedirs(os.path.join(self.base_exp_dir, 'validations_fine'), exist_ok=True)
        os.makedirs(os.path.join(self.base_exp_dir, 'normals'), exist_ok=True)

        for i in range(img_fine.shape[-1]):
            if len(out_rgb_fine) > 0:
                cv.imwrite(
                    os.path.join(self.base_exp_dir, 'validations_fine', '{:0>8d}_{}_{}.png'.format(self.iter_step, i, idx)),
                    np.concatenate(
                        [
                            img_fine[..., i],
                            self.dataset.image_at(idx, resolution_level=resolution_level),
                            self.dataset.mask_at(idx, resolution_level=resolution_level),
                        ]
                    ),
                )
            if len(out_normal_fine) > 0:
                cv.imwrite(
                    os.path.join(self.base_exp_dir, 'normals', '{:0>8d}_{}_{}.png'.format(self.iter_step, i, idx)),
                    np.concatenate([normal_img[..., i], self.dataset.normal_cam_at(idx, resolution_level=resolution_level)])[:, :, ::-1],
                )
            if len(out_mask) > 0:
                cv.imwrite(os.path.join(self.base_exp_dir, 'normals', '{:0>8d}_{}_{}_mask.png'.format(self.iter_step, i, idx)), mask_map[..., i])

    def save_maps(self, idx, img_idx, resolution_level=1):
        """
        save_maps 函数用于渲染指定视角的图像、法线图和掩码，并将结果保存为带有透明通道的 PNG 文件
        """
        # 定义视角类型
        view_types = ['front', 'back', 'left', 'right']
        print('Validate: iter: {}, camera: {}'.format(self.iter_step, idx))
        # 生成射线
        rays_o, rays_d = self.dataset.gen_rays_at(idx, resolution_level=resolution_level)
        H, W, _ = rays_o.shape
        rays_o = rays_o.reshape(-1, 3).split(self.batch_size)
        rays_d = rays_d.reshape(-1, 3).split(self.batch_size)

        out_rgb_fine = []
        out_normal_fine = []
        out_mask = []
        # 遍历射线批次并渲染
        for rays_o_batch, rays_d_batch in zip(rays_o, rays_d):
            
            near, far = self.dataset.get_near_far()
            background_rgb = torch.ones([1, 3]) if self.use_white_bkgd else None

            render_out = self.renderer.render(
                rays_o_batch, rays_d_batch, near, far, cos_anneal_ratio=self.get_cos_anneal_ratio(), background_rgb=background_rgb
            )

            def feasible(key):
                return (key in render_out) and (render_out[key] is not None)

            if feasible('color_fine'):
                out_rgb_fine.append(render_out['color_fine'].detach().cpu().numpy())
            if feasible('gradients') and feasible('weights'):
                n_samples = self.renderer.n_samples + self.renderer.n_importance
                normals = render_out['gradients'] * render_out['weights'][:, :n_samples, None]
                if feasible('inside_sphere'):
                    normals = normals * render_out['inside_sphere'][..., None]
                normals = normals.sum(dim=1).detach().cpu().numpy()
                out_normal_fine.append(normals)

            if feasible('weight_sum'):
                out_mask.append(render_out['weight_sum'].detach().clip(0, 1).cpu().numpy())

            del render_out

        img_fine = None
        if len(out_rgb_fine) > 0:
            img_fine = (np.concatenate(out_rgb_fine, axis=0).reshape([H, W, 3]) * 256).clip(0, 255)

        mask_map = None
        if len(out_mask) > 0:
            mask_map = (np.concatenate(out_mask, axis=0).reshape([H, W, 1]) * 256).clip(0, 255)

        normal_img = None
        if len(out_normal_fine) > 0:
            normal_img = np.concatenate(out_normal_fine, axis=0)
            # rot = np.linalg.inv(self.dataset.pose_all[idx, :3, :3].detach().cpu().numpy())
            world_normal_img = (normal_img.reshape([H, W, 3]) * 128 + 128).clip(0, 255)

        os.makedirs(os.path.join(self.base_exp_dir, 'coarse_maps'), exist_ok=True)
        img_rgba = np.concatenate([img_fine[:, :, ::-1], mask_map], axis=-1)
        normal_rgba = np.concatenate([world_normal_img[:, :, ::-1], mask_map], axis=-1)

        cv.imwrite(os.path.join(self.base_exp_dir, 'coarse_maps', "normals_mlp_%03d_%s.png" % (img_idx, view_types[idx])), img_rgba)
        cv.imwrite(os.path.join(self.base_exp_dir, 'coarse_maps', "normals_grad_%03d_%s.png" % (img_idx, view_types[idx])), normal_rgba)

    def render_novel_image(self, idx_0, idx_1, ratio, resolution_level):
        """
        render_novel_image 函数用于在两台相机之间插值生成新视角的图像。
        这种技术通常用于生成从不同视角观察物体的中间视图，以实现视角间的平滑过渡
        idx_0 和 idx_1：指定两个相机的索引
        ratio：插值系数，用于在两个视角之间进行线性插值。ratio 的值范围通常在 [0, 1] 之间，0 表示第一个视角，1 表示第二个视角
        resolution_level：渲染图像的分辨率级别
        """
        # 生成插值后的射线
        rays_o, rays_d = self.dataset.gen_rays_between(idx_0, idx_1, ratio, resolution_level=resolution_level)
        H, W, _ = rays_o.shape
        rays_o = rays_o.reshape(-1, 3).split(self.batch_size)
        rays_d = rays_d.reshape(-1, 3).split(self.batch_size)

        out_rgb_fine = []
        # 遍历射线批次并渲染
        for rays_o_batch, rays_d_batch in zip(rays_o, rays_d):
            near, far = self.dataset.get_near_far()
            background_rgb = torch.ones([1, 3]) if self.use_white_bkgd else None

            render_out = self.renderer.render(
                rays_o_batch, rays_d_batch, near, far, cos_anneal_ratio=self.get_cos_anneal_ratio(), background_rgb=background_rgb
            )

            out_rgb_fine.append(render_out['color_fine'].detach().cpu().numpy())

            del render_out
        # 处理渲染结果并生成图像
        img_fine = (np.concatenate(out_rgb_fine, axis=0).reshape([H, W, 3]) * 256).clip(0, 255).astype(np.uint8)
        return img_fine

    def validate_mesh(self, world_space=False, resolution=64, threshold=0.0):
        """
        validate_mesh 函数用于验证和导出3D网格模型。它提取当前模型的几何形状，
        并根据指定的分辨率和阈值生成网格数据。最后将生成的网格保存到文件中.
        world_space：布尔值，指示是否将顶点坐标转换为世界坐标系
        resolution：整数值，指定提取网格的分辨率，越高的分辨率会生成越细致的网格
        threshold：浮点值，指定提取几何体的阈值，通常用于控制提取网格的细节水平
        """
        # 获取物体边界框的最小和最大坐标
        bound_min = torch.tensor(self.dataset.object_bbox_min, dtype=torch.float32)
        bound_max = torch.tensor(self.dataset.object_bbox_max, dtype=torch.float32)
        # 提取几何体
        vertices, triangles, vertex_colors = self.renderer.extract_geometry(bound_min, bound_max, resolution=resolution, threshold=threshold)
        # 创建存储网格的目录
        os.makedirs(os.path.join(self.base_exp_dir, 'meshes'), exist_ok=True)
        # 根据世界坐标系进行顶点变换（如果需要）
        if world_space:
            vertices = vertices * self.dataset.scale_mats_np[0][0, 0] + self.dataset.scale_mats_np[0][:3, 3][None]
        # 创建 Trimesh 对象
        mesh = trimesh.Trimesh(vertices, triangles, vertex_colors=vertex_colors)
        # 导出网格文件
        mesh.export(os.path.join(self.base_exp_dir, 'meshes', 'tmp.glb'))

        logging.info('End')

    def interpolate_view(self, img_idx_0, img_idx_1):
        """
        interpolate_view 函数用于在两个视角之间插值生成一段视频。这在生成基于视角变化的动画或演示时非常有用
        """
        images = []
        n_frames = 60
        # 生成插值图像
        for i in range(n_frames):
            print(i)
            images.append(self.render_novel_image(img_idx_0, img_idx_1, np.sin(((i / n_frames) - 0.5) * np.pi) * 0.5 + 0.5, resolution_level=4))
        # 生成逆向播放的图像
        for i in range(n_frames):
            images.append(images[n_frames - i - 1])
        # 设置视频编码器与输出路径
        fourcc = cv.VideoWriter_fourcc(*'mp4v')
        video_dir = os.path.join(self.base_exp_dir, 'render')
        os.makedirs(video_dir, exist_ok=True)
        h, w, _ = images[0].shape
        writer = cv.VideoWriter(os.path.join(video_dir, '{:0>8d}_{}_{}.mp4'.format(self.iter_step, img_idx_0, img_idx_1)), fourcc, 30, (w, h))
        # 将生成的图像写入视频文件
        for image in images:
            writer.write(image)
        # 释放视频写入器
        writer.release()


if __name__ == '__main__':
    print('Hello Wooden') # 打印 "Hello Wooden" 作为程序启动的确认

    torch.set_default_tensor_type('torch.FloatTensor') # 设置PyTorch中默认的张量类型为 FloatTensor
    # 设置日志格式
    FORMAT = "[%(filename)s:%(lineno)s - %(funcName)20s() ] %(message)s"
    logging.basicConfig(level=logging.DEBUG, format=FORMAT)
    # 定义命令行参数
    parser = argparse.ArgumentParser()
    parser.add_argument('--conf', type=str, default='./confs/base.conf')
    parser.add_argument('--mode', type=str, default='train')
    parser.add_argument('--mcube_threshold', type=float, default=0.0)
    parser.add_argument('--is_continue', default=False, action="store_true")
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--case', type=str, default='')
    parser.add_argument('--data_dir', type=str, default='')
    # 解析命令行参数
    args = parser.parse_args()
    # 设置GPU设备
    torch.cuda.set_device(args.gpu)
    # 实例化Runner对象
    runner = Runner(args.conf, args.mode, args.case, args.is_continue, args.data_dir)
    # 根据模式执行任务
    if args.mode == 'train':
        runner.train()
        runner.validate_mesh(world_space=False, resolution=256, threshold=args.mcube_threshold)
    elif args.mode == 'save_maps': # 
        for i in range(4):
            runner.save_maps(idx=i, img_idx=runner.dataset.object_viewidx)
    elif args.mode == 'validate_mesh': # 验证网格模式
        runner.validate_mesh(world_space=False, resolution=512, threshold=args.mcube_threshold)
    elif args.mode.startswith('interpolate'):  # 视角插值模式
        _, img_idx_0, img_idx_1 = args.mode.split('_')
        img_idx_0 = int(img_idx_0)
        img_idx_1 = int(img_idx_1)
        runner.interpolate_view(img_idx_0, img_idx_1)
