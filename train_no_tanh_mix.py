# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import sys, os, time
sys.path.append('utils')
sys.path.append('models')
from utils.data import CelebA, RandomNoiseGenerator
from models.model_transformer_mixmlp import Generator, Discriminator
import argparse
import numpy as np
# from scipy.misc import imsave
import imageio
from utils.logger import Logger

class PGGAN():
    def __init__(self, G, D, data, noise, opts):
        self.G = G
        self.D = D
        self.data = data
        self.noise = noise
        self.opts = opts
        self.current_time = time.strftime('%Y-%m-%d %H%M%S')
        self.logger = Logger('./logs/' + self.current_time + "/")
        gpu = self.opts['gpu']
        self.use_cuda = len(gpu) > 0
        self.gpus = gpu
        os.environ['CUDA_VISIBLE_DEVICES'] = gpu

        current_time = time.strftime('%Y-%m-%d %H%M%S')
        self.opts['sample_dir'] = os.path.join(os.path.join(self.opts['exp_dir'], current_time), 'samples')
        self.opts['ckpt_dir'] = os.path.join(os.path.join(self.opts['exp_dir'], current_time), 'ckpts')
        os.makedirs(self.opts['sample_dir'])
        os.makedirs(self.opts['ckpt_dir'])

        # self.bs_map = {2**R: self.get_bs(2**R) for R in range(2, 11)}
        self.bs_map=  {4: 128, 8: 64, 16: 32, 32: 16, 64: 4, 128: 2, 256:2}
        # self.rows_map = {32: 8, 16: 4, 8: 4, 4: 2, 2: 2}
        self.rows_map = {128: 8, 128: 8, 64: 8, 32: 4, 16: 4,8:4,4:4,2:4}
        # save opts
        with open(os.path.join(os.path.join(self.opts['exp_dir'], current_time), 'options.txt'), 'w') as f:
            for k, v in self.opts.items():
                print('%s: %s' % (k, v), file=f)
            print('batch_size_map: %s' % self.bs_map, file=f)
            print(G, file=f)
            print(D, file=f)

    def get_bs(self, resolution):
        R = int(np.log2(resolution))
        if R < 7:
            bs = 32 / 2**(max(0, R-4))
        else:
            bs = 8 / 2**(min(2, R-7))
        return int(bs)
    def compute_wgan_gp(self,fake,real,cur_level,LAMBDA):
        if self.opts['gan'] == 'wgan_gp':
            real_data = real
            fake_data = fake
            netD = self.D
            # alpha = torch.rand(real.size(0),1,1, 1)
            # alpha = alpha.expand(real_data.size())
            alpha = torch.cuda.FloatTensor(np.random.random((real_data.size(0),1,1,1)))
            # alpha = alpha.cuda()
            # alpha = alpha.to(device)#cuda() #gpu) #if use_cuda else alpha
            # print('real_data shape is {}'.format(real_data.shape))
            # print('fake_data shape is {}'.format(fake_data.shape))
            interpolates = alpha * real_data + ((1 - alpha) * fake_data)
            # print('interpolates shape is {}'.format(interpolates.shape))
            # interpolates = interpolates.to(device)#.cuda()
            interpolates = interpolates.cuda()
            interpolates = torch.autograd.Variable(interpolates, requires_grad=True)
            disc_interpolates = netD(interpolates,cur_level=cur_level)
            gradients = torch.autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                                grad_outputs=torch.ones(disc_interpolates.size()).cuda(),#.cuda(), #if use_cuda else torch.ones(
                                    #disc_interpolates.size()),
                                create_graph=True, retain_graph=True, only_inputs=True)[0]#LAMBDA = 1
            gradients = gradients.view(gradients.size(0),-1)
            gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * (LAMBDA/(int(cur_level)**2))
        else:
            gradient_penalty = 0.0
        
        return gradient_penalty

    def register_on_gpu(self):
        if len(self.gpus) > 1:
            self.G = nn.DataParallel(self.G)
            self.D = nn.DataParallel(self.D)
        if self.use_cuda:
            self.G.cuda()
            self.D.cuda()

    def create_optimizer(self):
        self.optim_G = optim.Adam(self.G.parameters(), lr=self.opts['g_lr_max'], betas=(self.opts['beta1'], self.opts['beta2']))
        self.optim_D = optim.Adam(self.D.parameters(), lr=self.opts['d_lr_max'], betas=(self.opts['beta1'], self.opts['beta2']))

    def create_criterion(self):
        # w is for gan
        if self.opts['gan'] == 'lsgan':
            self.adv_criterion = lambda p,t,w: torch.mean((p-t)**2)  # sigmoid is applied here
        elif self.opts['gan'] == 'wgan_gp':
            # self.adv_criterion = self.wgn_gp_criterion(10)
            self.adv_criterion = lambda p,t,w: (-1)*torch.mean(p)+1e-8 if t else torch.mean(p)+1e-8
        elif self.opts['gan'] == 'gan':
            lambda p,t,w: -w*(torch.mean(t*torch.log(p+1e-8)) + torch.mean((1-t)*torch.log(1-p+1e-8)))
        else:
            raise ValueError('Invalid/Unsupported GAN: %s.' % self.opts['gan'])


    def compute_adv_loss(self, prediction, target, w):
        # print('compute_adv_loss value prediction{} target{} w{}'.format(prediction,target,w))
        return self.adv_criterion(prediction, target, w)

    def compute_additional_g_loss(self,cur_level):#reconstruct loss
        # alpha=100
        # loss = nn.MSELoss()
        # netG = self.G
        # real = self.real
        # rec_loss = alpha*loss(netG(self.z.detach(), cur_level=cur_level),real)
        # return rec_loss
        return 0.0

    def compute_additional_d_loss(self):  # drifting loss and gradient penalty, weighting inside this function
        return 0.0

    def _get_data(self, d):
        return d.data.item() if isinstance(d, Variable) else d

    def compute_G_loss(self):
        g_adv_loss = self.compute_adv_loss(self.d_fake, True, 1)
        g_add_loss = self.compute_additional_g_loss(self.level)
        self.g_adv_loss = self._get_data(g_adv_loss)
        self.g_add_loss = self._get_data(g_add_loss)
        return g_adv_loss + g_add_loss

    def compute_D_loss(self):
        self.d_adv_loss_real = self.compute_adv_loss(self.d_real, True, 0.5)
        # self.d_adv_loss_fake = self.compute_adv_loss(self.d_fake, False, 0.5) * self.opts['fake_weight']
        self.d_adv_loss_fake = self.compute_adv_loss(self.d_fake, False, 0.5)
        self.d_adv_gp = self.compute_wgan_gp(self.fake,self.real,self.level,10)
        d_adv_loss = self.d_adv_loss_real + self.d_adv_loss_fake+self.d_adv_gp
        d_add_loss = self.compute_additional_d_loss()
        self.d_adv_loss = self._get_data(d_adv_loss)
        self.d_add_loss = self._get_data(d_add_loss)
        self.d_adv_gp_loss = self._get_data(self.d_adv_gp)

        return d_adv_loss + d_add_loss

    def _rampup(self, epoch, rampup_length):
        if epoch < rampup_length:
            p = max(0.0, float(epoch)) / float(rampup_length)
            p = 1.0 - p
            return np.exp(-p*p*5.0)
        else:
            return 1.0

    def _rampdown_linear(self, epoch, num_epochs, rampdown_length):
        if epoch >= num_epochs - rampdown_length:
            return float(num_epochs - epoch) / rampdown_length
        else:
            return 1.0

    def update_lr(self, cur_nimg):
        for param_group in self.optim_G.param_groups:
            # lrate_coef = self._rampup(cur_nimg / 1000.0, self.opts['rampup_kimg'])
            # lrate_coef *= self._rampdown_linear(cur_nimg / 1000.0, self.opts['total_kimg'], self.opts['rampdown_kimg'])
            param_group['lr'] = (0.2 **(cur_nimg-1))  * self.opts['g_lr_max']
        for param_group in self.optim_D.param_groups:
            # lrate_coef = self._rampup(cur_nimg / 1000.0, self.opts['rampup_kimg'])
            # lrate_coef *= self._rampdown_linear(cur_nimg / 1000.0, self.opts['total_kimg'], self.opts['rampdown_kimg'])
            param_group['lr'] = (0.2 **(cur_nimg-1)) * self.opts['d_lr_max']

    def postprocess(self):
        # TODO: weight cliping or others
        pass

    def _numpy2var(self, x):
        var = Variable(torch.from_numpy(x))
        if self.use_cuda:
            var = var.cuda()
        return var

    def _var2numpy(self, var):
        if self.use_cuda:
            return var.cpu().data.numpy()
        return var.data.numpy()

    def add_noise(self, x):
        # TODO: support more method of adding noise.
        if self.opts.get('no_noise', False):
            print('i am here')
            return x
        print('i am noise')
        if hasattr(self, '_d_'):
            print(dir(self))
            print('_d_ {}'.format(self._d_))
            self._d_ = self._d_ * 0.9 + torch.mean(self.d_real).data.item() * 0.1
        else:
            self._d_ = 0.0
        print('self._d_{}'.format(self._d_))
        strength = 0.2 * max(0, self._d_ - 0.5)**2
        noise = self._numpy2var(np.random.randn(*x.size()).astype(np.float32) * strength)
        return x + noise

    def preprocess(self, z, real):
        self.z = self._numpy2var(z)
        self.real = self._numpy2var(real)
        # self.gt = self._numpy2var(x_gt)

    def forward_G(self, cur_level):
        self.d_fake = self.D(self.fake, cur_level=cur_level)
        # return self.d_fake
    
    def forward_D(self, cur_level, detach=True):
##################################for z input
        # self.fake = self.G(self.z, cur_level=cur_level)
        # # self.d_real = self.D(self.add_noise(self.real), cur_level=cur_level)
        # self.d_real = self.D(self.real, cur_level=cur_level)
        # # print('self.fake {}'.format(self.fake.shape))
        # self.d_fake = self.D(self.fake.detach() if detach else self.fake, cur_level=cur_level)
        # # return self.d_real,self.d_fake
        # # print('d_real', self.d_real.view(-1))
        # # print('d_fake', self.d_fake.view(-1))
        # # print(self.fake[0].view(-1))
        ################################for sr input
        self.fake = self.G(self.z, cur_level=cur_level)
        self.d_real = self.D(self.real, cur_level=cur_level)
        # print('self.fake {}'.format(self.fake.shape))
        self.d_fake = self.D(self.fake.detach() if detach else self.fake, cur_level=cur_level)

    def backward_G(self):
        g_loss = self.compute_G_loss()
        g_loss.backward()
        self.optim_G.step()
        self.g_loss = self._get_data(g_loss)

    def backward_D(self, retain_graph=False):
        d_loss = self.compute_D_loss()
        d_loss.backward(retain_graph=retain_graph)
        # for p in self.D.parameters():
        #     # print('i am clip')
        #     p.data.clamp_(-1, 1)
        # for tag, value in self.D.named_parameters():
        #     if(value.grad is not None):
        #         # print('i clicp')
        #         value.clamp_(-5, 5)#wgan-clip weight ->[-5 5]          
        self.optim_D.step()
        self.d_loss = self._get_data(d_loss)

    def report(self, it, num_it, phase, resol):
        formation = 'Iter[%d|%d], %s, %s, G: %.3f, D: %.3f, G_adv: %.3f, G_add: %.3f, D_adv: %.3f,D_gp: %.3f, D_add: %.3f'
        values = (it, num_it, phase, resol, self.g_loss, self.d_loss, self.g_adv_loss, self.g_add_loss, self.d_adv_loss, self.d_adv_gp_loss,self.d_add_loss)
        print(formation % values)

    def tensorboard(self, it, num_it, phase, resol, samples):
        # (1) Log the scalar values
        prefix = str(resol)+'/'+phase+'/'
        info = {prefix + 'G_loss': self.g_loss,
                prefix + 'G_adv_loss': self.g_adv_loss,
                prefix + 'G_add_loss': self.g_add_loss,
                prefix + 'D_loss': self.d_loss,
                prefix + 'D_adv_loss': self.d_adv_loss,
                prefix + 'D_add_loss': self.d_add_loss,
                prefix + 'D_adv_loss_fake': self._get_data(self.d_adv_loss_fake),
                prefix + 'D_adv_loss_real': self._get_data(self.d_adv_loss_real)}
        print('tensorboard lt is{}'.format(it))
        for tag, value in info.items():
            self.logger.scalar_summary(tag, value, it)

        # (2) Log values and gradients of the parameters (histogram)
        for tag, value in self.G.named_parameters():
            # print('tensor board G tag{} value{}'.format(tag,value.shape))
            tag = tag.replace('.', '/')
            self.logger.histo_summary('G/' + prefix +tag, self._var2numpy(value), it)
            if value.grad is not None:
                self.logger.histo_summary('G/' + prefix +tag + '/grad', self._var2numpy(value.grad), it)

        for tag, value in self.D.named_parameters():
            # print('tensor board D tag{} value{}'.format(tag,value.shape))
            tag = tag.replace('.', '/')
            self.logger.histo_summary('D/' + prefix + tag, self._var2numpy(value), it)
            if value.grad is not None:
                self.logger.histo_summary('D/' + prefix + tag + '/grad',
                                          self._var2numpy(value.grad), it)

        # (3) Log the images
        # info = {'images': samples[:10]}
        # for tag, images in info.items():
        #     logger.image_summary(tag, images, it)

    def train(self):
        # prepare
        self.create_optimizer()
        self.create_criterion()
        self.register_on_gpu()

        to_level = int(np.log2(self.opts['target_resol']))
        from_level = int(np.log2(self.opts['first_resol']))
        assert 2**to_level == self.opts['target_resol'] and 2**from_level == self.opts['first_resol'] and to_level >= from_level >= 2
        cur_level = from_level
        tensorboard_lt=0
        for R in range(from_level-1, to_level-1):
            batch_size = self.bs_map[2 ** (R+1)]
            train_kimg = int(self.opts['train_kimg'] * 1000)
            transition_kimg = int(self.opts['transition_kimg'] * 1000)
            if R == to_level-1:
                transition_kimg = 0
            cur_nimg = 0
            _len = len(str(train_kimg + transition_kimg))
            _num_it = (train_kimg + transition_kimg) // batch_size
            for it in range(_num_it):
                # determined current level: int for stabilizing and float for fading in
                cur_level = R + float(max(cur_nimg-train_kimg, 0)) / transition_kimg 
                cur_resol = 2 ** int(np.ceil(cur_level+1))
                phase = 'stabilize' if int(cur_level) == cur_level else 'fade_in'
                self.level = cur_level
                # get a batch noise and real images
                z = self.noise(batch_size)
                x = self.data(batch_size, cur_resol, cur_level)

                # preprocess
                self.preprocess(z, x)
                # update D
                self.optim_D.zero_grad()
                self.forward_D(cur_level, detach=True)  # TODO: feed gdrop_strength
                self.backward_D()

                # update G
                self.optim_G.zero_grad()
                self.forward_G(cur_level)
                self.backward_G()
                
                # report 
                self.report(it, _num_it, phase, cur_resol)
                
                cur_nimg += batch_size

                # sampling
                samples = []
                if (it % self.opts['sample_freq'] == 0) or it == _num_it-1 :
                # or abs(self.g_loss)>500:
                    samples = self.sample('')
                    imageio.imsave(os.path.join(self.opts['sample_dir'],
                                        '%dx%d-%s-%s.png' % (cur_resol, cur_resol, phase, str(it).zfill(6))), samples)

                # ===tensorboard visualization===
                if (it % self.opts['sample_freq'] == 0) or it == _num_it - 1: 
                # or abs(self.g_loss)>50000:
                    self.tensorboard(it+tensorboard_lt, _num_it, phase, cur_resol, samples)

                # save model
                if (it % self.opts['save_freq'] == 0 and it > 0) or it == _num_it-1: 
                # or abs(self.g_loss)>50000:
                    self.save(os.path.join(self.opts['ckpt_dir'], '%dx%d-%s-%s' % (cur_resol, cur_resol, phase, str(it).zfill(6))))
            tensorboard_lt=tensorboard_lt+_num_it
            self.update_lr(R)

    def sample(self, file_name):
        batch_size = self.z.size(0)
        n_row = self.rows_map[batch_size]
        n_col = int(np.ceil(batch_size / float(n_row)))
        samples = []
        i = j = 0
        for row in range(n_row):
            one_row = []
            # fake
            for col in range(n_col):
                one_row.append(self.fake[i].cpu().data.numpy())
                i += 1
            # real
            for col in range(n_col):
                one_row.append(self.real[j].cpu().data.numpy())
                j += 1
            samples += [np.concatenate(one_row, axis=2)]
        samples = np.concatenate(samples, axis=1).transpose([1, 2, 0])

        half = samples.shape[1] // 2
        samples[:,:half,:] = samples[:,:half,:] - np.min(samples[:,:half,:])
        samples[:,:half,:] = samples[:,:half,:] / np.max(samples[:,:half,:])
        samples[:,half:,:] = samples[:,half:,:] - np.min(samples[:,half:,:])
        samples[:,half:,:] = samples[:,half:,:] / np.max(samples[:,half:,:])
        return samples

    def save(self, file_name):
        g_file = file_name + '-G.pth'
        d_file = file_name + '-D.pth'
        torch.save(self.G.state_dict(), g_file)
        torch.save(self.D.state_dict(), d_file)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', default='', type=str, help='gpu(s) to use.')
    parser.add_argument('--train_kimg', default=600, type=float, help='# * 1000 real samples for each stabilizing training phase.')
    parser.add_argument('--transition_kimg', default=600, type=float, help='# * 1000 real samples for each fading in phase.')
    parser.add_argument('--total_kimg', default=10000, type=float, help='total_kimg: a param to compute lr.')
    parser.add_argument('--g_lr_max', default=1e-4, type=float, help='Generator learning rate')
    parser.add_argument('--d_lr_max', default=4e-4, type=float, help='Discriminator learning rate')
    parser.add_argument('--fake_weight', default=0.1, type=float, help="weight of fake images' loss of D")
    parser.add_argument('--beta1', default=0, type=float, help='beta1 for adam')
    parser.add_argument('--beta2', default=0.99, type=float, help='beta2 for adam')
    parser.add_argument('--gan', default='lsgan', type=str, help='model: lsgan/wgan_gp/gan, currently only support lsgan or gan with no_noise option.')
    parser.add_argument('--first_resol', default=4, type=int, help='first resolution')
    parser.add_argument('--target_resol', default=256, type=int, help='target resolution')
    parser.add_argument('--drift', default=1e-3, type=float, help='drift, only available for wgan_gp.')
    parser.add_argument('--sample_freq', default=300, type=int, help='sampling frequency.')
    parser.add_argument('--save_freq', default=5000, type=int, help='save model frequency.')
    parser.add_argument('--exp_dir', default='./exp', type=str, help='experiment dir.')
    parser.add_argument('--no_noise', action='store_true', help='do not add noise to real data.')
    parser.add_argument('--rampup_kimg', default=10000, type=float, help='rampup_kimg.')
    parser.add_argument('--rampdown_kimg', default=10000, type=float, help='rampdown_kimg.')
    # TODO: support conditional inputs

    args = parser.parse_args()
    opts = {k:v for k,v in args._get_kwargs()}

    latent_size = 512
    sigmoid_at_end = args.gan in ['lsgan', 'gan']

    G = Generator(num_channels=3, latent_size=latent_size, resolution=args.target_resol, fmap_max=latent_size, fmap_base=8192, tanh_at_end=False)
    D = Discriminator(num_channels=3, resolution=args.target_resol, fmap_max=latent_size, fmap_base=8192, sigmoid_at_end=sigmoid_at_end)
    print(G)
    print(D)
    data = CelebA()
    noise = RandomNoiseGenerator(latent_size, 'gaussian')
    pggan = PGGAN(G, D, data, noise, opts)
    pggan.train()
