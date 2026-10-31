from __future__ import print_function
import argparse
import os
#from torchvision.utils import save_image
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
from utils.data import get_training_set
from utils.networks import define_G, define_D, GANLoss, get_scheduler, update_learning_rate
from utils.Perceptual import PerceptualLoss

# Training settings
parser = argparse.ArgumentParser(description='Residual Nets')
parser.add_argument('--dataset', type=str,default='cuhk', help='[xm2vts, ar, cuhk, cuhk_feret, WildSketch]')
parser.add_argument('--output_path', type=str,default='cuhk_cvq', help='Name of the subfolder under the ./checkpoint/codebook/ directory for saving outputs')
parser.add_argument('--batchSize', type=int, default=1, help='training batch size')
parser.add_argument('--testBatchSize', type=int, default=1, help='testing batch size')
parser.add_argument('--input_nc', type=int, default=3, help='input image channels')
parser.add_argument('--output_nc', type=int, default=3, help='output image channels')
parser.add_argument('--ngf', type=int, default=64, help='generator filters in first conv layer')
parser.add_argument('--epoch_count', type=int, default=1, help='the starting epoch count')
parser.add_argument('--niter', type=int, default=300, help='# of iter at starting learning rate')
parser.add_argument('--niter_decay', type=int, default=300, help='# of iter to linearly decay learning rate to zero')
parser.add_argument('--ndf', type=int, default=64, help='discriminator filters in first conv layer')
parser.add_argument('--lr', type=float, default=0.0002, help='initial learning rate for adam')
parser.add_argument('--lr_policy', type=str, default='lambda', help='learning rate policy: lambda|step|plateau|cosine')
parser.add_argument('--lr_decay_iters', type=int, default=50, help='multiply by a gamma every lr_decay_iters iterations')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
parser.add_argument('--vgg', type=str,default=True, help='use vgg loss?')
parser.add_argument('--cuda', type=str,default=True, help='use cuda?')
parser.add_argument('--visual', action='store_true', help='visualize the result?')
parser.add_argument('--threads', type=int, default=4, help='number of threads for data loader to use')
parser.add_argument('--seed', type=int, default=123, help='random seed to use. Default=123')
# CHANGES.md 2026-09-26 (paths / device / resume only): data from data/datasets, outputs under outputs/codebook, resumable state every epoch
parser.add_argument('--data_root', type=str, default=None, help='default <repo>/data/datasets')
parser.add_argument('--out_dir', type=str, default=None, help='default <repo>/outputs/codebook/<dataset>/<output_path>')
parser.add_argument('--device', type=str, default='cuda:0')
parser.add_argument('--resume', action='store_true', help='continue from <out_dir>/state.pt')

opt = parser.parse_args()

print(opt)

if opt.cuda and not torch.cuda.is_available():
    raise Exception("No GPU found, please run without --cuda")

cudnn.benchmark = True

torch.manual_seed(opt.seed)
if opt.cuda:
    torch.cuda.manual_seed(opt.seed)

print('===> Loading datasets')
DECP_DIR = os.path.dirname(os.path.abspath(__file__))
root_path = opt.data_root or os.path.join(os.path.dirname(DECP_DIR), 'data', 'datasets')
out_dir = opt.out_dir or os.path.join(os.path.dirname(DECP_DIR), 'outputs', 'codebook', opt.dataset, opt.output_path)
os.makedirs(out_dir, exist_ok=True)
train_set = get_training_set(os.path.join(root_path, opt.dataset))

training_data_loader = DataLoader(dataset=train_set, num_workers=opt.threads, batch_size=opt.batchSize, shuffle=True)


device = torch.device(opt.device if opt.cuda else "cpu")
# stage1_weights = f"./checkpoint/codebook/{opt.output_path}/netG_a_model_epoch_last.pth"
net_g_a = define_G(True, 'instance', 'normal', 0.02, gpu_ids=device, num_embeddings=1024)
# stage1_weights = f"./checkpoint/codebook/{opt.output_path}/netG_b_model_epoch_last.pth"
net_g_b = define_G(True, 'instance', 'normal', 0.02, gpu_ids=device, num_embeddings=1024)
net_d_a = define_D(opt.input_nc, opt.ndf, 'basic', gpu_ids=device)
net_d_b = define_D(opt.input_nc, opt.ndf, 'basic', gpu_ids=device)

# setup optimizer
criterionGAN = GANLoss().to(device)
if opt.vgg:
    criterionVGG = PerceptualLoss().to(device)

optimizer_net_g_a = optim.Adam(filter(lambda p: p.requires_grad,net_g_a.parameters()), lr=opt.lr, betas=(opt.beta1, 0.999))
optimizer_net_g_b = optim.Adam(filter(lambda p: p.requires_grad,net_g_b.parameters()), lr=opt.lr, betas=(opt.beta1, 0.999))
optimizer_net_d_a = optim.Adam(filter(lambda p: p.requires_grad,net_d_a.parameters()), lr=opt.lr, betas=(opt.beta1, 0.999))
optimizer_net_d_b = optim.Adam(filter(lambda p: p.requires_grad,net_d_b.parameters()), lr=opt.lr, betas=(opt.beta1, 0.999))


a_scheduler = get_scheduler(optimizer_net_g_a, opt)
b_scheduler = get_scheduler(optimizer_net_g_b, opt)

# set_detect_anomaly(True) removed: debug-only slowdown (F25), no numerical effect
state_path = os.path.join(out_dir, 'state.pt')
if opt.resume and os.path.isfile(state_path):
    st = torch.load(state_path, map_location=device)
    for m, k in ((net_g_a, 'g_a'), (net_g_b, 'g_b'), (net_d_a, 'd_a'), (net_d_b, 'd_b'), (optimizer_net_g_a, 'o_g_a'), (optimizer_net_g_b, 'o_g_b'),
                 (optimizer_net_d_a, 'o_d_a'), (optimizer_net_d_b, 'o_d_b'), (a_scheduler, 's_a'), (b_scheduler, 's_b')):
        m.load_state_dict(st[k])
    opt.epoch_count = st['epoch'] + 1
    print('resumed from epoch', st['epoch'])
for epoch in range(opt.epoch_count, opt.niter + opt.niter_decay + 1):
    # train
    for iteration, batch in enumerate(training_data_loader, 1):
        # a是photo,b是sketch

        # forward
        real_a, real_b = batch[0].to(device), batch[1].to(device)

        rec_a, q_loss_a = net_g_a(real_a)

        rec_b, q_loss_b = net_g_b(real_b)

        ## train net_d_a  
        optimizer_net_d_a.zero_grad()
        # train with rec
        pred_rec_a = net_d_a.forward(rec_a.detach())
        loss_d_a_rec = criterionGAN(pred_rec_a, False)
        # train with real
        pred_real_a = net_d_a.forward(real_a)
        loss_d_a_real = criterionGAN(pred_real_a, True)
        # Combined D loss
        loss_d_a = (loss_d_a_rec + loss_d_a_real) * 0.5
        loss_d_a.backward()
        optimizer_net_d_a.step()

        ## train net_d_b
        optimizer_net_d_b.zero_grad()
        # train with rec
        pred_rec_b = net_d_b.forward(rec_b.detach())
        loss_d_b_rec = criterionGAN(pred_rec_b, False)
        # train with real
        pred_real_b = net_d_b.forward(real_b)
        loss_d_b_real = criterionGAN(pred_real_b, True)
        # Combined D loss
        loss_d_b = (loss_d_b_rec + loss_d_b_real) * 0.5
        loss_d_b.backward()
        optimizer_net_d_b.step()

        # train net_g_a
        optimizer_net_g_a.zero_grad()      
        # gan loss
        pred_a = net_d_a.forward(rec_a)
        loss_g_a = criterionGAN(pred_a, True)
        # vgg loss
        loss_g_vgg_a = criterionVGG(rec_a, real_a)*0.5 
        # loss_l1_a = criterionL1(rec_a,real_a)
        loss_g_a_ = loss_g_a + loss_g_vgg_a + q_loss_a*2
        loss_g_a_.backward()    
        optimizer_net_g_a.step()

        # train net_g_b
        optimizer_net_g_b.zero_grad()
        # gan loss
        pred_b = net_d_b.forward(rec_b)
        loss_g_b = criterionGAN(pred_b, True)
        # vgg loss
        loss_g_vgg_b = criterionVGG(rec_b, real_b)*0.5 
        # loss_l1_b = criterionL1(rec_b,real_b)
        loss_g_b_ = loss_g_b  + loss_g_vgg_b + q_loss_b*2
        loss_g_b_.backward()
        optimizer_net_g_b.step()

        # print loss
        print("===> Epoch[{}]({}/{}): loss_g_a: {:.4f}, loss_g_vgg_a:{:.4f}, q_loss_a:{:.4f}, loss_g_b: {:.4f}, loss_g_vgg_b:{:.4f}, q_loss_b:{:.4f}".format(
            epoch, iteration, len(training_data_loader), loss_g_a.item(), loss_g_vgg_a.item(),q_loss_a.item()*2, loss_g_b.item(), loss_g_vgg_b.item(), q_loss_b.item()*2))
    
    update_learning_rate(a_scheduler, optimizer_net_g_a)
    update_learning_rate(b_scheduler, optimizer_net_g_b)

    if epoch % 50 == 0:
        torch.save(net_g_a.state_dict(), os.path.join(out_dir, 'netG_a_model_epoch_{}.pth'.format(epoch)))
        torch.save(net_g_b.state_dict(), os.path.join(out_dir, 'netG_b_model_epoch_{}.pth'.format(epoch)))
        torch.save(net_d_a.state_dict(), os.path.join(out_dir, 'netD_a_model_epoch_{}.pth'.format(epoch)))
        torch.save(net_d_b.state_dict(), os.path.join(out_dir, 'netD_b_model_epoch_{}.pth'.format(epoch)))
        print("Checkpoint saved to {}".format(out_dir))
    torch.save({'epoch': epoch, 'g_a': net_g_a.state_dict(), 'g_b': net_g_b.state_dict(), 'd_a': net_d_a.state_dict(), 'd_b': net_d_b.state_dict(),
                'o_g_a': optimizer_net_g_a.state_dict(), 'o_g_b': optimizer_net_g_b.state_dict(), 'o_d_a': optimizer_net_d_a.state_dict(),
                'o_d_b': optimizer_net_d_b.state_dict(), 's_a': a_scheduler.state_dict(), 's_b': b_scheduler.state_dict()}, state_path + '.tmp')
    os.replace(state_path + '.tmp', state_path)

# final weights under the names the rest of the project expects (data/codebooks/<id>_cvq/netG_{a,b}_model.pth)
torch.save(net_g_a.state_dict(), os.path.join(out_dir, 'netG_a_model.pth'))
torch.save(net_g_b.state_dict(), os.path.join(out_dir, 'netG_b_model.pth'))
print('final codebooks written to', out_dir)
