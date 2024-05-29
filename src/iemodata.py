import os
import torch
import numpy as np
import h5py
from torch.nn.utils.rnn import pad_sequence
import random
import torch.utils.data as data
from PIL import Image
import torchvision.transforms as transforms
from abc import ABC, abstractmethod


class BaseDataset(data.Dataset, ABC):
    """This class is an abstract base class (ABC) for datasets.

    To create a subclass, you need to implement the following four functions:
    -- <__init__>:                      initialize the class, first call BaseDataset.__init__(self, opt).
    -- <__len__>:                       return the size of dataset.
    -- <__getitem__>:                   get a data point.
    -- <modify_commandline_options>:    (optionally) add dataset-specific options and set default options.
    """

    def __init__(self, opt):
        """Initialize the class; save the options in the class

        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        self.opt = opt
        self.manual_collate_fn = False
        # self.root = opt.dataroot

    @staticmethod
    def modify_commandline_options(parser, is_train):
        """Add new dataset-specific options, and rewrite default values for existing options.

        Parameters:
            parser          -- original option parser
            is_train (bool) -- whether training phase or test phase. You can use this flag to add training-specific or test-specific options.

        Returns:
            the modified parser.
        """
        return parser

    @abstractmethod
    def __len__(self):
        """Return the total number of images in the dataset."""
        return 0

    @abstractmethod
    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index - - a random integer for data indexing

        Returns:
            a dictionary of data with their names. It ususally contains the data itself and its metadata information.
        """
        pass


def get_params(opt, size):
    w, h = size
    new_h = h
    new_w = w
    if opt.preprocess == "resize_and_crop":
        new_h = new_w = opt.load_size
    elif opt.preprocess == "scale_width_and_crop":
        new_w = opt.load_size
        new_h = opt.load_size * h // w

    x = random.randint(0, np.maximum(0, new_w - opt.crop_size))
    y = random.randint(0, np.maximum(0, new_h - opt.crop_size))

    flip = random.random() > 0.5

    return {"crop_pos": (x, y), "flip": flip}


def get_transform(
    opt, params=None, grayscale=False, method=Image.BICUBIC, convert=True
):
    transform_list = []
    if grayscale:
        transform_list.append(transforms.Grayscale(1))
    if "resize" in opt.preprocess:
        osize = [opt.load_size, opt.load_size]
        transform_list.append(transforms.Resize(osize, method))
    elif "scale_width" in opt.preprocess:
        transform_list.append(
            transforms.Lambda(lambda img: __scale_width(img, opt.load_size, method))
        )

    if "crop" in opt.preprocess:
        if params is None:
            transform_list.append(transforms.RandomCrop(opt.crop_size))
        else:
            transform_list.append(
                transforms.Lambda(
                    lambda img: __crop(img, params["crop_pos"], opt.crop_size)
                )
            )

    if opt.preprocess == "none":
        transform_list.append(
            transforms.Lambda(lambda img: __make_power_2(img, base=4, method=method))
        )

    if not opt.no_flip:
        if params is None:
            transform_list.append(transforms.RandomHorizontalFlip())
        elif params["flip"]:
            transform_list.append(
                transforms.Lambda(lambda img: __flip(img, params["flip"]))
            )

    if convert:
        transform_list += [transforms.ToTensor()]
        if grayscale:
            transform_list += [transforms.Normalize((0.5,), (0.5,))]
        else:
            transform_list += [transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    return transforms.Compose(transform_list)


def __make_power_2(img, base, method=Image.BICUBIC):
    ow, oh = img.size
    h = int(round(oh / base) * base)
    w = int(round(ow / base) * base)
    if (h == oh) and (w == ow):
        return img

    __print_size_warning(ow, oh, w, h)
    return img.resize((w, h), method)


def __scale_width(img, target_width, method=Image.BICUBIC):
    ow, oh = img.size
    if ow == target_width:
        return img
    w = target_width
    h = int(target_width * oh / ow)
    return img.resize((w, h), method)


def __crop(img, pos, size):
    ow, oh = img.size
    x1, y1 = pos
    tw = th = size
    if ow > tw or oh > th:
        return img.crop((x1, y1, x1 + tw, y1 + th))
    return img


def __flip(img, flip):
    if flip:
        return img.transpose(Image.FLIP_LEFT_RIGHT)
    return img


def __print_size_warning(ow, oh, w, h):
    """Print warning information about image size(only print once)"""
    if not hasattr(__print_size_warning, "has_printed"):
        print(
            "The image size needs to be a multiple of 4. "
            "The loaded image size was (%d, %d), so it was adjusted to "
            "(%d, %d). This adjustment will be done to all images "
            "whose sizes are not multiples of 4" % (ow, oh, w, h)
        )
        __print_size_warning.has_printed = True


class IEMOData(BaseDataset):
    def __init__(self, opt, data_path, set_name, drop_rate, full_data=False):
        """IEMOCAP dataset reader
        set_name in ['trn', 'val', 'tst']
        """
        super().__init__(opt)

        # record & load basic settings
        cvNo = opt.cvNo
        self.set_name = set_name
        self.drop_rate = drop_rate
        self.full_data = full_data
        config = {
            "target_root": os.path.join(data_path, "target"),
            "feature_root": data_path,
        }
        self.norm_method = opt.norm_method
        self.corpus_name = opt.corpus_name
        # load feature
        self.A_type = opt.A_type
        self.all_A = h5py.File(
            os.path.join(config["feature_root"], "A", f"{self.A_type}.h5"), "r"
        )
        if self.A_type == "comparE":
            self.mean_std = h5py.File(
                os.path.join(config["feature_root"], "A", "comparE_mean_std.h5"), "r"
            )
            self.mean = (
                torch.from_numpy(self.mean_std[str(cvNo)]["mean"][()])
                .unsqueeze(0)
                .float()
            )
            self.std = (
                torch.from_numpy(self.mean_std[str(cvNo)]["std"][()])
                .unsqueeze(0)
                .float()
            )
        elif self.A_type == "comparE_raw":
            self.mean, self.std = self.calc_mean_std()

        self.V_type = opt.V_type
        self.all_V = h5py.File(
            os.path.join(config["feature_root"], "V", f"{self.V_type}.h5"), "r"
        )
        self.L_type = opt.L_type
        self.all_L = h5py.File(
            os.path.join(config["feature_root"], "L", f"{self.L_type}.h5"), "r"
        )

        # load dataset in memory
        if opt.in_mem:
            self.all_A = self.h5_to_dict(self.all_A)
            self.all_V = self.h5_to_dict(self.all_V)
            self.all_L = self.h5_to_dict(self.all_L)

        # load target
        label_path = os.path.join(
            config["target_root"], f"{cvNo}", f"{set_name}_label.npy"
        )
        int2name_path = os.path.join(
            config["target_root"], f"{cvNo}", f"{set_name}_int2name.npy"
        )
        self.label = np.load(label_path)
        if self.corpus_name == "IEMOCAP":
            self.label = np.argmax(self.label, axis=1)
        self.int2name = np.load(int2name_path)
        self.manual_collate_fn = True

    def __getitem__(self, index):
        int2name = self.int2name[index]
        if self.corpus_name == "IEMOCAP":
            int2name = int2name[0].decode()
        label = torch.tensor(self.label[index])
        # process A_feat
        A_feat = torch.from_numpy(self.all_A[int2name][()]).float()
        if self.A_type == "comparE" or self.A_type == "comparE_raw":
            A_feat = (
                self.normalize_on_utt(A_feat)
                if self.norm_method == "utt"
                else self.normalize_on_trn(A_feat)
            )
        # process V_feat
        V_feat = torch.from_numpy(self.all_V[int2name][()]).float()
        # process L_feat
        L_feat = torch.from_numpy(self.all_L[int2name][()]).float()
        X = (L_feat, A_feat, V_feat)
        missing_code = self.get_missing_mode()
        return X, label, missing_code

    def __len__(self):
        return len(self.label)

    def h5_to_dict(self, h5f):
        ret = {}
        for key in h5f.keys():
            ret[key] = h5f[key][()]
        return ret

    def normalize_on_utt(self, features):
        mean_f = torch.mean(features, dim=0).unsqueeze(0).float()
        std_f = torch.std(features, dim=0).unsqueeze(0).float()
        std_f[std_f == 0.0] = 1.0
        features = (features - mean_f) / std_f
        return features

    def normalize_on_trn(self, features):
        features = (features - self.mean) / self.std
        return features

    def get_dim(self):
        return (1024, 130, 342)

    def get_seq_len(self):
        return (22, 350, 50)

    def get_missing_mode(self):
        if self.full_data:
            return 6
        if random.random() < self.drop_rate:
            return random.randint(0, 5)
        else:
            return 6

    def calc_mean_std(self):
        utt_ids = [utt_id for utt_id in self.all_A.keys()]
        feats = np.array([self.all_A[utt_id] for utt_id in utt_ids])
        _feats = feats.reshape(-1, feats.shape[2])
        mean = np.mean(_feats, axis=0)
        std = np.std(_feats, axis=0)
        std[std == 0.0] = 1.0
        return mean, std

    def collate_fn(self, batch):
        max_length = 350
        A = [
            torch.cat(
                [
                    sample[0][1],
                    torch.zeros(
                        (max_length - len(sample[0][1]), sample[0][1].shape[1]),
                        device="cpu",
                    ),
                ]
            )
            for sample in batch
        ]
        V = [sample[0][2] for sample in batch]
        L = [sample[0][0] for sample in batch]
        A = pad_sequence(A, batch_first=True, padding_value=0)
        V = pad_sequence(V, batch_first=True, padding_value=0)
        L = pad_sequence(L, batch_first=True, padding_value=0)
        self.al = A.shape[1]
        self.vl = V.shape[1]
        self.ll = L.shape[1]

        label = torch.tensor([sample[1] for sample in batch])
        missing_code = torch.tensor([sample[2] for sample in batch])

        X = (L, A, V)

        return X, label, missing_code
