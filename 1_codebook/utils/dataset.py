from os import listdir
from os.path import join, splitext, exists
import random

from PIL import Image
import torch
import torch.utils.data as data
import torchvision.transforms as transforms

from utils.util import is_image_file, load_img


class DatasetFromFolder(data.Dataset):
    def __init__(self, image_dir):
        super(DatasetFromFolder, self).__init__()
        self.a_path = join(image_dir, "a")
        self.b_path = join(image_dir, "b")
        self.image_filenames = [x for x in listdir(self.a_path) if is_image_file(x)]
        self.random_filenames = self.image_filenames[:]
        random.shuffle(self.random_filenames)
        # I/O fix (2026-09-28, CHANGES.md): pair the sketch by stem, so a photo .jpg finds its sketch .png (FS2K style 2);
        # identical to the original lookup whenever the file names match.
        b_files = {splitext(x)[0]: x for x in listdir(self.b_path) if is_image_file(x)}
        self.b_name = lambda name: name if exists(join(self.b_path, name)) else b_files[splitext(name)[0]]

        transform_list = [transforms.ToTensor(),
                          transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]

        self.transform = transforms.Compose(transform_list)

    # def __getitem__(self, index):
    #     a = Image.open(join(self.a_path, self.image_filenames[index])).convert('RGB')
    #     # index_b = random.randint(0, len(self.b_image_filenames) - 1)
    #     b = Image.open(join(self.b_path, self.image_filenames[index])).convert('RGB')
    #     a = a.resize((286,286), Image.BICUBIC)
    #     b = b.resize((286,286), Image.BICUBIC)

    #     a = transforms.ToTensor()(a)
    #     b = transforms.ToTensor()(b)
    #     w_offset = random.randint(0, max(0, 286 - 256 - 1))
    #     h_offset = random.randint(0, max(0, 286 - 256 - 1))
    
    #     a = a[:, h_offset:h_offset + 256, w_offset:w_offset + 256]
    #     b = b[:, h_offset:h_offset + 256, w_offset:w_offset + 256]
    
    #     a = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(a)
    #     b = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(b)

    #     a0 = Image.open(join(self.a_path, self.random_filenames[index])).convert('RGB')
    #     # index_b = random.randint(0, len(self.b_image_filenames) - 1)
    #     b0 = Image.open(join(self.b_path, self.random_filenames[index])).convert('RGB')
    #     a0 = a0.resize((286,286), Image.BICUBIC)
    #     b0 = b0.resize((286,286), Image.BICUBIC)

    #     a0 = transforms.ToTensor()(a0)
    #     b0 = transforms.ToTensor()(b0)
    #     w_offset = random.randint(0, max(0, 286 - 256 - 1))
    #     h_offset = random.randint(0, max(0, 286 - 256 - 1))
    
    #     a0 = a0[:, h_offset:h_offset + 256, w_offset:w_offset + 256]
    #     b0 = b0[:, h_offset:h_offset + 256, w_offset:w_offset + 256]
    
    #     a0 = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(a0)
    #     b0 = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(b0)

    #     return a, b, a0, b0
    
    def __getitem__(self, index):
        # Load and process the main a and b images
        a = Image.open(join(self.a_path, self.image_filenames[index])).convert('RGB')
        b = Image.open(join(self.b_path, self.b_name(self.image_filenames[index]))).convert('RGB')
        
        a = a.resize((286, 286), Image.BICUBIC)
        b = b.resize((286, 286), Image.BICUBIC)

        a = transforms.ToTensor()(a)
        b = transforms.ToTensor()(b)
        
        w_offset = random.randint(0, max(0, 286 - 256 - 1))
        h_offset = random.randint(0, max(0, 286 - 256 - 1))

        a = a[:, h_offset:h_offset + 256, w_offset:w_offset + 256]
        b = b[:, h_offset:h_offset + 256, w_offset:w_offset + 256]

        a = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(a)
        b = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(b)

        # Load multiple reference a0 and b0 images and stack them as a batch
        num_references = 1  # Number of references to load
        a0_list = []
        b0_list = []

        for _ in range(num_references):
            random_idx = random.choice([i for i in range(len(self.random_filenames)) if i != index])
            
            a0 = Image.open(join(self.a_path, self.image_filenames[random_idx])).convert('RGB')
            b0 = Image.open(join(self.b_path, self.b_name(self.image_filenames[random_idx]))).convert('RGB')
            
            a0 = a0.resize((286, 286), Image.BICUBIC)
            b0 = b0.resize((286, 286), Image.BICUBIC)

            a0 = transforms.ToTensor()(a0)
            b0 = transforms.ToTensor()(b0)
            
            w_offset = random.randint(0, max(0, 286 - 256 - 1))
            h_offset = random.randint(0, max(0, 286 - 256 - 1))

            a0 = a0[:, h_offset:h_offset + 256, w_offset:w_offset + 256]
            b0 = b0[:, h_offset:h_offset + 256, w_offset:w_offset + 256]

            a0 = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(a0)
            b0 = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(b0)

            a0_list.append(a0)
            b0_list.append(b0)

        # Stack the list of tensors along the first dimension to form a batch
        a0_batch = torch.stack(a0_list)  # Shape: (num_references, C, H, W)
        b0_batch = torch.stack(b0_list)  # Shape: (num_references, C, H, W)

        return a, b, a0_batch, b0_batch



    def __len__(self):
        return len(self.image_filenames)

# class DatasetFromFolder(data.Dataset):
#     def __init__(self, image_dir, split="train"):
#         super(DatasetFromFolder, self).__init__()
#         self.split = split
#         self.a_path = join(image_dir, "a")
#         self.b_path = join(image_dir, "b")
#         self.image_filenames = [x for x in listdir(self.a_path) if is_image_file(x)]
#         self.random_filenames = self.image_filenames[:]
#         random.shuffle(self.random_filenames)

#         transform_list = [transforms.ToTensor(),
#                           transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]

#         self.transform = transforms.Compose(transform_list)


#     def __getitem__(self, index):
#         a = Image.open(join(self.a_path, self.image_filenames[index])).convert('RGB')
#         b = Image.open(join(self.b_path, self.image_filenames[index])).convert('RGB')
#         if self.split == "train":
#             a = a.resize((286,286), Image.BICUBIC)
#             b = b.resize((286,286), Image.BICUBIC)
#         elif self.split == "val" or self.split == "test":
#             a = a.resize((256,256), Image.BICUBIC)
#             b = b.resize((256,256), Image.BICUBIC)
#         else:
#             raise ValueError(f"Invalid split name: {self.split}")
        
#         a = transforms.ToTensor()(a)
#         b = transforms.ToTensor()(b)

#         if self.split == "train":
#             w_offset = random.randint(0, max(0, 286 - 256 - 1))
#             h_offset = random.randint(0, max(0, 286 - 256 - 1))
        
#             a = a[:, h_offset:h_offset + 256, w_offset:w_offset + 256]
#             b = b[:, h_offset:h_offset + 256, w_offset:w_offset + 256]
    
#         a = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(a)
#         b = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(b)

#         # reference image
#         a0 = Image.open(join(self.a_path, self.random_filenames[index])).convert('RGB')
#         b0 = Image.open(join(self.b_path, self.random_filenames[index])).convert('RGB')
#         if self.split == "train":
#             a0 = a0.resize((286,286), Image.BICUBIC)
#             b0 = b0.resize((286,286), Image.BICUBIC)
#         elif self.split == "val" or self.split == "test":
#             a0 = a0.resize((256,256), Image.BICUBIC)
#             b0 = b0.resize((256,256), Image.BICUBIC)
#         else:
#             raise ValueError(f"Invalid split name: {self.split}")

#         a0 = transforms.ToTensor()(a0)
#         b0 = transforms.ToTensor()(b0)
#         if self.split == "train":
#             w_offset = random.randint(0, max(0, 286 - 256 - 1))
#             h_offset = random.randint(0, max(0, 286 - 256 - 1))
#             a0 = a0[:, h_offset:h_offset + 256, w_offset:w_offset + 256]
#             b0 = b0[:, h_offset:h_offset + 256, w_offset:w_offset + 256]
    
#         a0 = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(a0)
#         b0 = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(b0)

#         return a, b, a0, b0