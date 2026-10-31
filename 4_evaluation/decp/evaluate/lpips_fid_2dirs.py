import argparse
import os
import lpips
import torch
parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('-d0','--dir0', type=str, default='/results/cufs/b2a')
parser.add_argument('-d1','--dir1', type=str, default='/data/cufs/a')

parser.add_argument('-v','--version', type=str, default='0.1')
parser.add_argument('--use_gpu', default='True', help='turn on flag to use GPU')

opt = parser.parse_args()
device = torch.device("cuda:2")

## Initializing the model
loss_fn1 = lpips.LPIPS(net='alex',version=opt.version)
loss_fn2 = lpips.LPIPS(net='squeeze',version=opt.version)
loss_fn3 = lpips.LPIPS(net='vgg',version=opt.version)
if(opt.use_gpu):
	loss_fn1.to(device)
	loss_fn2.to(device)
	loss_fn3.to(device)

# a2b crawl directories
#f = open(opt.outa2b,'w')
files = os.listdir(opt.dir0)
dist1_a2b = 0
dist2_a2b,dist3_a2b = 0,0
count_a2b = 0
with torch.no_grad():
	for file in files:
		if(os.path.exists(os.path.join(opt.dir1,file))):
			# Load images
			img0 = lpips.im2tensor(lpips.load_image(os.path.join(opt.dir0,file))) # RGB image from [-1,1]
			img1 = lpips.im2tensor(lpips.load_image(os.path.join(opt.dir1,file)))

			if(opt.use_gpu):
				img0 = img0.to(device)
				img1 = img1.to(device)

			# Compute distance
			dist01_a2b = loss_fn1.forward(img0,img1)
			dist02_a2b = loss_fn2.forward(img0,img1)
			dist03_a2b = loss_fn3.forward(img0,img1)
			dist1_a2b+=dist01_a2b
			dist2_a2b+=dist02_a2b
			dist3_a2b+=dist03_a2b
			count_a2b+=1
			# print('%s: %.3f'%(file,dist01_a2b))
	#		f.writelines('%s: %.6f\n'%(file,dist01_a2b))
			del img0,img1,dist01_a2b
			torch.cuda.empty_cache()
	print('a2b:\nalex:%.4f\nsqueeze:%.4f\nvgg:%.4f\n'%(dist1_a2b/count_a2b,dist2_a2b/count_a2b,dist3_a2b/count_a2b))

# import subprocess
# def compute_fid(path1, path2):
#     command = ['python', '-m', 'pytorch_fid', path1, path2, '--device', 'cuda:0']
#     result = subprocess.run(command, capture_output=True, text=True)
#     return result.stdout
# print(compute_fid(opt.dir0, opt.dir1))
from calculate import calculate_metrics
mean_metrics = calculate_metrics(opt.dir0, opt.dir1)
for metric, value in mean_metrics.items():
        print(metric + ": " + str(value))