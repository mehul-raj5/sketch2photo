from argparse import ArgumentParser
import time
import pickle
import numpy as np
import os
import json
import sys
from PIL import Image
import multiprocessing as mp
import math
import torch
import torchvision.transforms as trans
import warnings
warnings.filterwarnings("ignore")

sys.path.append("./")

from evaluate.models.mtcnn.mtcnn import MTCNN
from evaluate.models.encoders.model_irse import IR_101
CIRCULAR_FACE_PATH = 'pretrained_models/CurricularFace_Backbone.pth'


def chunks(lst, n):
	"""Yield successive n-sized chunks from lst."""
	for i in range(0, len(lst), n):
		yield lst[i:i + n]


def extract_on_paths(file_paths):
	facenet = IR_101(input_size=112)
	facenet.load_state_dict(torch.load(CIRCULAR_FACE_PATH))
	facenet.cuda()
	facenet.eval()
	mtcnn = MTCNN()
	id_transform = trans.Compose([
		trans.ToTensor(),
		trans.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
	])

	pid = mp.current_process().name
	print('\t{} is starting to extract on {} images'.format(pid, len(file_paths)))
	tot_count = len(file_paths)
	count = 0

	scores_dict = {}
	for res_path, gt_path in file_paths:
		count += 1
		if count % 100 == 0:
			print('{} done with {}/{}'.format(pid, count, tot_count))
		if True:
			input_im = Image.open(res_path)
			if input_im.mode == 'L':
				input_im = input_im.convert('RGB')
			# print(input_im.size)
			input_im, _ = mtcnn.align(input_im)
			if input_im is None:
				print('{} skipping {}'.format(pid, res_path))
				continue

			input_id = facenet(id_transform(input_im).unsqueeze(0).cuda())[0]
			
			if 'photo' in os.path.basename(gt_path) and 'FS2K' in gt_path:
				gt_path_basename = os.path.basename(gt_path).replace('photo', 'sketch').replace('image', 'sketch')
				gt_path = os.path.join(os.path.dirname(gt_path), gt_path_basename)
			# print(res_path, gt_path)
			result_im = Image.open(gt_path)
			if result_im.mode == 'L':
				result_im = result_im.convert('RGB')
			result_im, _ = mtcnn.align(result_im)
			if result_im is None:
				print('{} skipping {}'.format(pid, gt_path))
				continue

			result_id = facenet(id_transform(result_im).unsqueeze(0).cuda())[0]
			score = float(input_id.dot(result_id))
			scores_dict[os.path.basename(gt_path)] = score

	return scores_dict


def parse_args():
	parser = ArgumentParser(add_help=False)
	parser.add_argument('--num_threads', type=int, default=4)
	parser.add_argument('--data_path', type=str,  default='results')
	parser.add_argument('--gt_path', type=str, default='gt_images')
	args = parser.parse_args()
	return args


def run(args):
	file_paths = []
	for f in os.listdir(args.data_path):
		image_path = os.path.join(args.data_path, f)
		gt_path = os.path.join(args.gt_path, f)
		if f.endswith(".jpg") or f.endswith('.png'):
			# file_paths.append([image_path, gt_path.replace('.png','.jpg')])
			file_paths.append([image_path, gt_path])

	file_chunks = list(chunks(file_paths, int(math.ceil(len(file_paths) / args.num_threads))))
	pool = mp.Pool(args.num_threads)
	print('Running on {} paths\nHere we goooo'.format(len(file_paths)))

	tic = time.time()
	results = pool.map(extract_on_paths, file_chunks)
	scores_dict = {}
	for d in results:
		scores_dict.update(d)

	all_scores = list(scores_dict.values())

	mean = np.mean(all_scores)
	std = np.std(all_scores)
	#result_str = 'New Average score is {:.2f}+-{:.2f}'.format(mean, std)
	#print(result_str)

	f = open(os.path.join(args.data_path, "result.txt"),'a')
	f.write('Time: {}'.format(time.strftime('%Y-%m-%d %H:%M:%S ',time.localtime(time.time()))))
	f.write('ID(CurricularFace) Score between {} and {}: {} \n'.format(args.data_path, args.gt_path, mean))

	toc = time.time()
	print('Mischief managed in {}s'.format(toc - tic))
	print('ID Score between {} and {}:'.format(args.data_path, args.gt_path), mean)


if __name__ == '__main__':
	args = parse_args()
	with torch.no_grad():
		run(args)
