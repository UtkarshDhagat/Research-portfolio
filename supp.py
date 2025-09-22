# -*- coding: utf-8 -*-

! pip install imageio[ffmpeg] ffmpeg moviepy asteroid ipywidgets pydub resampy

import warnings
warnings.filterwarnings('ignore')

! pip install h5py

# Dataloader

"""AVE dataset"""
import numpy as np
import torch
import h5py

class AVEDataset(object):

    def __init__(self, video_dir, audio_dir, label_dir, order_dir, batch_size):
        self.video_dir = video_dir
        self.audio_dir = audio_dir
        self.batch_size = batch_size

        with h5py.File(order_dir, 'r') as hf:
            order = hf['order'][:]
        self.lis = order

        with h5py.File(audio_dir, 'r') as hf:
            self.audio_features = hf['avadataset'][:]
        with h5py.File(label_dir, 'r') as hf:
            self.labels = hf['avadataset'][:]
        with h5py.File(video_dir, 'r') as hf:
            self.video_features = hf['avadataset'][:]

        self.video_batch = np.float32(np.zeros([self.batch_size, 10, 7, 7, 512]))
        self.audio_batch = np.float32(np.zeros([self.batch_size, 10, 128]))
        self.label_batch = np.float32(np.zeros([self.batch_size, 10, 29]))

    def __len__(self):
        return len(self.lis)

    def get_batch(self, idx):

        for i in range(self.batch_size):
            id = idx * self.batch_size + i

            self.video_batch[i, :, :, :, :] = self.video_features[self.lis[id], :, :, :, :]
            self.audio_batch[i, :, :] = self.audio_features[self.lis[id], :, :]
            self.label_batch[i, :, :] = self.labels[self.lis[id], :, :]

        return torch.from_numpy(self.audio_batch).float(), torch.from_numpy(self.video_batch).float(), torch.from_numpy(
            self.label_batch).float()

class AVE_weak_Dataset(object):
    def __init__(self, video_dir, video_dir_bg, audio_dir , audio_dir_bg, label_dir, label_dir_bg, label_dir_gt, order_dir, batch_size, status):
        self.video_dir = video_dir
        self.audio_dir = audio_dir
        self.video_dir_bg = video_dir_bg
        self.audio_dir_bg = audio_dir_bg

        self.status = status
        # self.lis_video = os.listdir(video_dir)
        self.batch_size = batch_size
        with h5py.File(order_dir, 'r') as hf:
            train_l = hf['order'][:]
        self.lis = train_l
        with h5py.File(audio_dir, 'r') as hf:
            self.audio_features = hf['avadataset'][:]
        with h5py.File(label_dir, 'r') as hf:
            self.labels = hf['avadataset'][:]
        with h5py.File(video_dir, 'r') as hf:
            self.video_features = hf['avadataset'][:]
        self.audio_features = self.audio_features[train_l, :, :]
        self.video_features = self.video_features[train_l, :, :]
        self.labels = self.labels[train_l, :]

        if status == "train":
            with h5py.File(label_dir_bg, 'r') as hf:
                self.negative_labels = hf['avadataset'][:]

            with h5py.File(audio_dir_bg, 'r') as hf:
                self.negative_audio_features = hf['avadataset'][:]
            with h5py.File(video_dir_bg, 'r') as hf:
                self.negative_video_features = hf['avadataset'][:]

            size = self.audio_features.shape[0] + self.negative_audio_features.shape[0]
            audio_train_new = np.zeros((size, self.audio_features.shape[1], self.audio_features.shape[2]))
            audio_train_new[0:self.audio_features.shape[0], :, :] = self.audio_features
            audio_train_new[self.audio_features.shape[0]:size, :, :] = self.negative_audio_features
            self.audio_features = audio_train_new

            video_train_new = np.zeros((size, 10, 7, 7, 512))
            video_train_new[0:self.video_features.shape[0], :, :] = self.video_features
            video_train_new[self.video_features.shape[0]:size, :, :] = self.negative_video_features
            self.video_features = video_train_new

            y_train_new = np.zeros((size, 29))
            y_train_new[0:self.labels.shape[0], :] = self.labels
            y_train_new[self.labels.shape[0]:size, :] = self.negative_labels
            self.labels = y_train_new
        else:
            with h5py.File(label_dir_gt, 'r') as hf:
                self.labels = hf['avadataset'][:]
                self.labels = self.labels[train_l, :, :]



        self.video_batch = np.float32(np.zeros([self.batch_size, 10, 7, 7, 512]))
        self.audio_batch = np.float32(np.zeros([self.batch_size, 10, 128]))
        if status == "train":
            self.label_batch = np.float32(np.zeros([self.batch_size, 29]))
        else:
            self.label_batch = np.float32(np.zeros([self.batch_size,10, 29]))

    def __len__(self):
        return len(self.labels)

    def get_batch(self, idx):
        for i in range(self.batch_size):
            id = idx * self.batch_size + i

            self.video_batch[i, :, :, :, :] = self.video_features[id, :, :, :, :]
            self.audio_batch[i, :, :] = self.audio_features[id, :, :]
            if self.status == "train":
                self.label_batch[i, :] = self.labels[id, :]
            else:
                self.label_batch[i, :, :] = self.labels[id, :, :]
        return torch.from_numpy(self.audio_batch).float(), torch.from_numpy(self.video_batch).float(), torch.from_numpy(
            self.label_batch).float()

# Models fusion

import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.nn import init

class TBMRF_Net(nn.Module):
    '''
    two-branch/dual muli-modal residual fusion
    '''
    def __init__(self, embedding_dim, hidden_dim, hidden_size, tagset_size, nb_block):
        super(TBMRF_Net, self).__init__()
        self.hidden_dim = hidden_dim
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()
        self.relu = nn.ReLU()

        self.lstm_audio = nn.LSTM(128, hidden_dim, 1, batch_first=True, bidirectional=True)
        self.lstm_video = nn.LSTM(512, hidden_dim, 1, batch_first=True, bidirectional=True)
        self.affine_audio = nn.Linear(128, hidden_size)
        self.affine_video = nn.Linear(512, hidden_size)
        self.affine_v = nn.Linear(hidden_size, 49, bias=False)
        self.affine_g = nn.Linear(hidden_size, 49, bias=False)
        self.affine_h = nn.Linear(49, 1, bias=False)

        # fusion transformation functions
        self.nb_block = nb_block

        self.U_v  = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim * 2),
            nn.Tanh(),
            nn.Linear(hidden_dim * 2, hidden_dim * 2),
        )

        self.U_a = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim * 2),
            nn.Tanh(),
            nn.Linear(hidden_dim * 2, hidden_dim * 2),
        )


        self.L2 = nn.Linear(hidden_dim*2, tagset_size)

        self.init_weights()
        if torch.cuda.is_available():
            self.cuda()
        else:
            self

    def TBMRF_block(self, audio, video, nb_block):

        for i in range(nb_block):
            video_residual = video
            v = self.U_v(video)
            audio_residual = audio
            a = self.U_a(audio)
            merged = torch.mul(v + a, 0.5)

            a_trans = audio_residual
            v_trans = video_residual

            video = self.tanh(a_trans + merged)
            audio = self.tanh(v_trans + merged)

        fusion = torch.mul(video + audio, 0.5)#
        return fusion

    def init_weights(self):
        """Initialize the weights."""
        init.xavier_uniform(self.L2.weight)

    def forward(self, audio, video):

        v_t = video.view(video.size(0) * video.size(1), -1, 512)
        V = v_t

        # Audio-guided visual attention
        v_t = self.relu(self.affine_video(v_t))
        a_t = audio.view(-1, audio.size(-1))
        a_t = self.relu(self.affine_audio(a_t))
        content_v = self.affine_v(v_t) \
                    + self.affine_g(a_t).unsqueeze(2)
        z_t = self.affine_h((F.tanh(content_v))).squeeze(2)
        alpha_t = F.softmax(z_t, dim=-1).view(z_t.size(0), -1, z_t.size(1))  # attention map
        c_t = torch.bmm(alpha_t, V).view(-1, 512)
        video_t = c_t.view(video.size(0), -1, 512) # attended visual features

        # BiLSTM for Temporal modeling
        if torch.cuda.is_available():
            hidden1 = (autograd.Variable(torch.zeros(2, audio.size(0), self.hidden_dim).cuda()),
                       autograd.Variable(torch.zeros(2, audio.size(0), self.hidden_dim).cuda()))
            hidden2 = (autograd.Variable(torch.zeros(2, audio.size(0), self.hidden_dim).cuda()),
                       autograd.Variable(torch.zeros(2, audio.size(0), self.hidden_dim).cuda()))
        else:
            hidden1 = (autograd.Variable(torch.zeros(2, audio.size(0), self.hidden_dim)),
                       autograd.Variable(torch.zeros(2, audio.size(0), self.hidden_dim)))
            hidden2 = (autograd.Variable(torch.zeros(2, audio.size(0), self.hidden_dim)),
                       autograd.Variable(torch.zeros(2, audio.size(0), self.hidden_dim)))
        self.lstm_video.flatten_parameters()
        self.lstm_audio.flatten_parameters()
        lstm_audio, hidden1 = self.lstm_audio(
            audio.view(len(audio), 10, -1), hidden1)
        lstm_video, hidden2 = self.lstm_video(
            video_t.view(len(video), 10, -1), hidden2)

        # Feature fusion and prediction
        output = self.TBMRF_block(lstm_audio, lstm_video, self.nb_block)
        out = self.L2(output)
        out = F.softmax(out, dim=-1)

        return out

import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.nn import init

class att_Net(nn.Module):
    '''
    audio-visual event localization with audio-guided visual attention and audio-visual fusion
    '''
    def __init__(self, embedding_dim, hidden_dim, hidden_size, tagset_size):
        super(att_Net, self).__init__()
        self.hidden_dim = hidden_dim
        self.lstm_audio = nn.LSTM(128, hidden_dim, 1, batch_first=True, bidirectional=True)
        self.lstm_video = nn.LSTM(512, hidden_dim, 1, batch_first=True, bidirectional=True)

        self.relu = nn.ReLU()
        self.affine_audio = nn.Linear(128, hidden_size)
        self.affine_video = nn.Linear(512, hidden_size)
        self.affine_v = nn.Linear(hidden_size, 49, bias=False)
        self.affine_g = nn.Linear(hidden_size, 49, bias=False)
        self.affine_h = nn.Linear(49, 1, bias=False)

        self.L1 = nn.Linear(hidden_dim * 4, 64)
        self.L2 = nn.Linear(64, tagset_size)

        self.init_weights()
        if torch.cuda.is_available():
            self.cuda()
        else:
            self

    def init_weights(self):
        """Initialize the weights."""
        init.xavier_uniform(self.affine_v.weight)
        init.xavier_uniform(self.affine_g.weight)
        init.xavier_uniform(self.affine_h.weight)

        init.xavier_uniform(self.L1.weight)
        init.xavier_uniform(self.L2.weight)
        init.xavier_uniform(self.affine_audio.weight)
        init.xavier_uniform(self.affine_video.weight)

    def forward(self, audio, video):

        v_t = video.view(video.size(0) * video.size(1), -1, 512)
        V = v_t

        # Audio-guided visual attention
        v_t = self.relu(self.affine_video(v_t))
        a_t = audio.view(-1, audio.size(-1))
        a_t = self.relu(self.affine_audio(a_t))
        content_v = self.affine_v(v_t) \
                    + self.affine_g(a_t).unsqueeze(2)

        z_t = self.affine_h((F.tanh(content_v))).squeeze(2)
        alpha_t = F.softmax(z_t, dim=-1).view(z_t.size(0), -1, z_t.size(1)) # attention map
        c_t = torch.bmm(alpha_t, V).view(-1, 512)
        video_t = c_t.view(video.size(0), -1, 512) #attended visual features

        # Bi-LSTM for temporal modeling
        if torch.cuda.is_available():
            hidden1 = (autograd.Variable(torch.zeros(2, audio.size(0), self.hidden_dim).cuda()),
                       autograd.Variable(torch.zeros(2, audio.size(0), self.hidden_dim).cuda()))
            hidden2 = (autograd.Variable(torch.zeros(2, audio.size(0), self.hidden_dim).cuda()),
                       autograd.Variable(torch.zeros(2, audio.size(0), self.hidden_dim).cuda()))
        else:
            hidden1 = (autograd.Variable(torch.zeros(2, audio.size(0), self.hidden_dim)),
                       autograd.Variable(torch.zeros(2, audio.size(0), self.hidden_dim)))
            hidden2 = (autograd.Variable(torch.zeros(2, audio.size(0), self.hidden_dim)),
                       autograd.Variable(torch.zeros(2, audio.size(0), self.hidden_dim)))
        self.lstm_video.flatten_parameters()
        self.lstm_audio.flatten_parameters()
        lstm_audio, hidden1 = self.lstm_audio(
            audio.view(len(audio), 10, -1), hidden1)
        lstm_video, hidden2 = self.lstm_video(
            video_t.view(len(video), 10, -1), hidden2)

        # concatenation and prediction
        output = torch.cat((lstm_audio, lstm_video), -1)
        output = self.relu(output)
        out = self.L1(output)
        out = self.relu(out)
        out = self.L2(out)
        out = F.softmax(out, dim=-1)


        return out

model_name="DMRN"
dir_video="/home/sid/Desktop/Work/Samsung_Project/Research/AVE-ECCV18-master/data/visual_feature.h5"
dir_audio="/home/sid/Desktop/Work/Samsung_Project/Research/AVE-ECCV18-master/data/audio_feature.h5"
dir_labels='/home/sid/Desktop/Work/Samsung_Project/Research/AVE-ECCV18-master/data/labels.h5'
dir_order_train ="/home/sid/Desktop/Work/Samsung_Project/Research/AVE-ECCV18-master/data/train_order.h5"
dir_order_val = "/home/sid/Desktop/Work/Samsung_Project/Research/AVE-ECCV18-master/data/val_order.h5"
dir_order_test ="/home/sid/Desktop/Work/Samsung_Project/Research/AVE-ECCV18-master/data/test_order.h5"
nb_epoch = 300
batch_size = 4
train_flag =True
model_save_path = "/home/sid/Desktop/Work/Samsung_Project/Research/AVE-ECCV18-master"
AVEData = AVEDataset(video_dir=dir_video, audio_dir=dir_audio, label_dir=dir_labels,
                         order_dir=dir_order_train, batch_size=batch_size)
att_model = torch.load('/home/sid/Desktop/Work/Samsung_Project/Research/AVE-ECCV18-master/data/DMRN.pt')

from __future__ import print_function
import os
import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim
from torch.autograd import Variable
from torch.optim.lr_scheduler import StepLR
from sklearn.metrics import accuracy_score, classification_report
import random
random.seed(3344)
import time
import warnings
warnings.filterwarnings("ignore")


model_name="DMRN"
dir_video="/kaggle/input/avedataset-modified/AVE_Dataset/visual_feature.h5"
dir_audio="/kaggle/input/avedataset-modified/AVE_Dataset/audio_feature.h5"
dir_labels='/kaggle/input/avedataset-modified/AVE_Dataset/labels.h5'
dir_order_train ="/kaggle/input/avedataset-modified/AVE_Dataset/train_order.h5"
dir_order_val = "/kaggle/input/avedataset-modified/AVE_Dataset/val_order.h5"
dir_order_test ="/kaggle/input/avedataset-modified/AVE_Dataset/test_order.h5"
nb_epoch = 300
batch_size = 64
train_flag =True
model_save_path = "/kaggle/working/"

# model
if model_name == 'AV_att': # corresponding to A+V-att model in the paper
    net_model = att_Net(128, 128, 512, 29)
elif model_name == 'DMRN': # corresponding to DMRN. The pre-trained DMRN.pt was trained by fine-tuning the AV_att model.
    net_model = TBMRF_Net(128, 128, 512, 29, 1)

net_model

loss_function = nn.MultiLabelSoftMarginLoss()
optimizer = optim.Adam(net_model.parameters(), lr=1e-3)
scheduler = StepLR(optimizer, step_size=15000, gamma=0.1)

def compute_acc(labels, x_labels, nb_batch):
    N = int(nb_batch * 10)
    pre_labels = np.zeros(N)
    real_labels = np.zeros(N)
    c = 0
    for i in range(nb_batch):
        for j in range(x_labels.shape[1]):
            pre_labels[c] = np.argmax(x_labels[i, j, :])
            real_labels[c] = np.argmax(labels[i, j, :])
            c += 1
    target_names = []
    for i in range(29):
        target_names.append("class" + str(i))

    return accuracy_score(real_labels, pre_labels)


def train():
    AVEData = AVEDataset(video_dir=dir_video, audio_dir=dir_audio, label_dir=dir_labels,
                         order_dir=dir_order_train, batch_size=batch_size)
    nb_batch = AVEData.__len__() // batch_size
    epoch_l = []
    best_val_acc = 0
    for epoch in range(nb_epoch):
        epoch_loss = 0
        n = 0
        start = time.time()
        for i in range(nb_batch):
            audio_inputs, video_inputs, labels = AVEData.get_batch(i)

            if torch.cuda.is_available():
                audio_inputs = Variable(audio_inputs.cuda(), requires_grad=False)
                video_inputs = Variable(video_inputs.cuda(), requires_grad=False)
                labels = Variable(labels.cuda(), requires_grad=False)
            else:
                audio_inputs = Variable(audio_inputs, requires_grad=False)
                video_inputs = Variable(video_inputs, requires_grad=False)
                labels = Variable(labels, requires_grad=False)
            net_model.zero_grad()
            scores = net_model(audio_inputs, video_inputs)
            loss = loss_function(scores, labels)
            epoch_loss += loss.cpu().data.numpy()
            loss.backward()
            scheduler.step()
            optimizer.step()
            n = n + 1

        end = time.time()
        epoch_l.append(epoch_loss)
        print("=== Epoch {%s}   Loss: {%.4f}  Running time: {%4f}" % (str(epoch), (epoch_loss) / n, end - start))
        if not os.path.exists(model_save_path):
            os.makedirs(model_save_path)
        if epoch % 5 == 0:
            val_acc = val()

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(net_model, model_save_path + model_name + ".pt")

def val():
    net_model.eval()
    AVEData = AVEDataset(video_dir=dir_video, audio_dir=dir_audio, label_dir=dir_labels,
                         order_dir=dir_order_val, batch_size=402)
    nb_batch = AVEData.__len__()
    audio_inputs, video_inputs, labels = AVEData.get_batch(0)
    if torch.cuda.is_available():
        audio_inputs = Variable(audio_inputs.cuda(), requires_grad=False)
        video_inputs = Variable(video_inputs.cuda(), requires_grad=False)
    else:
        audio_inputs = Variable(audio_inputs, requires_grad=False)
        video_inputs = Variable(video_inputs, requires_grad=False)
    labels = labels.numpy()
    x_labels = net_model(audio_inputs, video_inputs)
    x_labels = x_labels.cpu().data.numpy()

    acc = compute_acc(labels, x_labels, nb_batch)
    print(acc)
    return acc


def test():

    model = torch.load(model_save_path + model_name  + ".pt")
    model.eval()
    AVEData = AVEDataset(video_dir=dir_video, audio_dir=dir_audio, label_dir=dir_labels,
                         order_dir=dir_order_test, batch_size=402)
    nb_batch = AVEData.__len__()
    audio_inputs, video_inputs, labels = AVEData.get_batch(0)
    if torch.cuda.is_available():
        audio_inputs = Variable(audio_inputs.cuda(), requires_grad=False)
        video_inputs = Variable(video_inputs.cuda(), requires_grad=False)
    else:
        audio_inputs = Variable(audio_inputs, requires_grad=False)
        video_inputs = Variable(video_inputs, requires_grad=False)
    labels = labels.numpy()
    x_labels = model(audio_inputs, video_inputs)
    x_labels = x_labels.cpu().data.numpy()
    acc = compute_acc(labels, x_labels, nb_batch)
    print(acc)
    return acc


# training and testing
if train_flag:
    train()
else:
    test()

test()

f = open("/kaggle/input/avedataset-modified/AVE_Dataset/Annotations.txt", 'r')
dataset = f.readlines()
print("The dataset contains %d samples" % (len(dataset)))
f.close()
# print(dataset[0])

labels=[]
for i in dataset[1:]:
    curr_label=i.split('&')[0]
    if curr_label not in labels:
        labels.append(curr_label)
print(len(dataset))
print(len(labels))
labels_set=tuple(labels)
print((labels_set))

testSet=AVEDataset(video_dir=dir_video, audio_dir=dir_audio, label_dir=dir_labels,
                         order_dir=dir_order_test, batch_size=402)
audio_inputs, video_inputs, labels = testSet.get_batch(0)
print(testSet.__len__())

# audio_inputs, video_inputs, labels = AVEData.get_batch(0)

def predict(model, audio_inputs, video_inputs):
    model.eval()
    if torch.cuda.is_available():
        audio_inputs = Variable(audio_inputs.cuda(), requires_grad=False)
        video_inputs = Variable(video_inputs.cuda(), requires_grad=False)
    else:
        audio_inputs = Variable(audio_inputs, requires_grad=False)
        video_inputs = Variable(video_inputs, requires_grad=False)
    with torch.no_grad():
        x_labels = model(audio_inputs, video_inputs)
        x_labels = x_labels.cpu().data.numpy()
    print("New")
    print(len(x_labels))
    return x_labels
predicted_labels=predict(att_model,audio_inputs,video_inputs)

# print(predicted_labels[0][0])
label_idx=np.argmax(predicted_labels[40][0])
if(label_idx<28):
    print(label_idx)
    print(labels_set[label_idx])
else:
    print("Could not find the label for the given video")

predicted_labels.shape

!mkdir /kaggle/working/dmrn/

from __future__ import print_function
import os

import h5py
import torch

# os.environ["CUDA_VISIBLE_DEVICES"] = "0" # set GPU ID
import torch.nn as nn
from torch.autograd import Variable
import matplotlib.pyplot as plt
import random
import time
from sklearn.metrics import accuracy_score
import torch.nn.functional as F
import imageio
import cv2
from sklearn.preprocessing import MinMaxScaler
import warnings

# model_name="AV_att"
dir_video="/kaggle/input/avedataset-modified/AVE_Dataset/visual_feature.h5"
dir_audio="/kaggle/input/avedataset-modified/AVE_Dataset/audio_feature.h5"
dir_labels='/kaggle/input/avedataset-modified/AVE_Dataset/labels.h5'
dir_order_train ="/kaggle/input/avedataset-modified/AVE_Dataset/train_order.h5"
dir_order_val = "/kaggle/input/avedataset-modified/AVE_Dataset/val_order.h5"
dir_order_test ="/kaggle/input/avedataset-modified/AVE_Dataset/test_order.h5"
batch_size = 64
model_save_path = "/kaggle/working/"

warnings.filterwarnings("ignore")

scaler = MinMaxScaler(copy=True, feature_range=(0, 1))
def video_frame_sample(frame_interval, video_length, sample_num):

    num = []
    for l in range(video_length):

        for i in range(sample_num):
            num.append(int(l * frame_interval + (i * 1.0 / sample_num) * frame_interval))

    return num

def normlize(x, min = 0, max = 255):

    num, row, col = x.shape
    for i in range(num):
        xi = x[i, :, :]
        xi = max *(xi - np.min(xi))/(np.max(xi) - np.min(xi))
        x[i, :, :] = xi
    return x

def create_heatmap(im_map, im_cloud, kernel_size=(5,5),colormap=cv2.COLORMAP_JET,a1=0.5,a2=0.3):
    print(np.max(im_cloud))

    im_cloud[:, :, 1] = 0
    im_cloud[:, :, 2] = 0
    return (a1*im_map + a2*im_cloud).astype(np.uint8)


# access to original videos for extracting video frames
raw_video_dir = "/kaggle/input/avedataset-modified/AVE_Dataset/AVE/" # videos in AVE dataset
lis = os.listdir(raw_video_dir)
f = open("/kaggle/input/avedataset-modified/AVE_Dataset/Annotations.txt", 'r')
dataset = f.readlines()
print("The dataset contains %d samples" % (len(dataset)))
f.close()
f = open("/kaggle/input/avedataset-modified/AVE_Dataset/testSet.txt", 'r')
testsetfiles = f.readlines()
print("The dataset contains %d samples" % (len(testsetfiles)))
f.close()
print(len(testsetfiles))
with h5py.File(dir_order_test, 'r') as hf:
    test_order = hf['order'][:]
# print(len(test_order))

# # pre-trained models
# att_model = torch.load('model/AV_att.pt')
att_model = torch.load('/kaggle/working/DMRN.pt')
att_layer = att_model._modules.get('affine_h') # extract attention maps from the layer


# load testing set
AVEData = AVEDataset(video_dir=dir_video, audio_dir=dir_audio, label_dir=dir_labels,
                     order_dir=dir_order_test, batch_size=402)
nb_batch = AVEData.__len__()
print(nb_batch)
audio_inputs, video_inputs, labels = AVEData.get_batch(0)
if torch.cuda.is_available():
    audio_inputs = Variable(audio_inputs.cuda(), requires_grad=False)
    video_inputs = Variable(video_inputs.cuda(), requires_grad=False)
else:
    audio_inputs = Variable(audio_inputs, requires_grad=False)
    video_inputs = Variable(video_inputs, requires_grad=False)
labels = labels.numpy()

# generate attention maps
att_map = torch.zeros((4020, 49, 1))
def fun(m, i, o): att_map.copy_(o.data)
map = att_layer.register_forward_hook(fun)
h_x = att_model(audio_inputs, video_inputs)
map.remove()
z_t = Variable(att_map.squeeze( 2 ))
alpha_t = F.softmax( z_t, dim = -1 ).view( z_t.size( 0 ), -1, z_t.size( 1 ) )
att_weight = alpha_t.view(402, 10, 7, 7).cpu().data.numpy() # attention maps of all testing samples

c = 0
t = 10
sample_num = 24 # 24 frames for 1-sec video segment
extract_frames = np.zeros((240, 224, 224, 3)) # 240 224x224x3 frames for a 10-sec video
save_dir = "/kaggle/working/dmrn/visual_att/attention_maps/" # store attention maps
original_dir = "/kaggle/working/dmrn/visual_att/original/"   # store video frames

for num in range(len(test_order)):
    print(num)
    data = testsetfiles[test_order[num]]
    x = data.split("&")

    # extract video frames
    video_index = os.path.join(raw_video_dir, x[1] + '.mp4')
    vid = imageio.get_reader(video_index, 'ffmpeg')
    #vid_len = len(vid)
    vid_len = vid.count_frames()
    frame_interval = int(vid_len / t)

    frame_num = video_frame_sample(frame_interval, t, sample_num)
    imgs = []
    for i, im in enumerate(vid):

        x_im = cv2.resize(im, (224, 224))
        imgs.append(x_im)
    vid.close()
    cc = 0
    for n in frame_num:
        extract_frames[cc, :, :, :] = imgs[n]
        cc += 1

    # process generated attention maps
    att = att_weight[num, :, :, :]
    att = normlize(att, 0, 255)
    att_scaled = np.zeros((10, 224, 224))
    for k in range(att.shape[0]):
        att_scaled[k, :, :] = cv2.resize(att[k, :, :], (224, 224)) # scaling attention maps

    att_t = np.repeat(att_scaled, 24, axis = 0) # 1-sec segment only has 1 attention map. Here, repeat 16 times to generate 16 maps for a 1-sec video
    heat_maps = np.repeat(att_t.reshape(240, 224, 224, 1), 3, axis = -1)
    c += 1

    att_dir = save_dir + x[1]
    ori_dir =  original_dir + x[1]
    out_video = cv2.VideoWriter(x[1]+".mp4",cv2.VideoWriter_fourcc(*'mp4v'),24,(224, 224))
    if not os.path.exists(att_dir):
      os.makedirs(att_dir)
    if not os.path.exists(ori_dir):
      os.makedirs(ori_dir)
    for idx in range(240):
        heat_map = heat_maps[idx, :, :, 0]
        im = extract_frames[idx, :, :, :]
        im = im[:, :, (2, 1, 0)]
        heatmap = cv2.applyColorMap(np.uint8(heat_map), cv2.COLORMAP_JET)

        att_frame = heatmap * 0.2 + np.uint8(im) * 0.6
        n = "%04d" % idx
        vid_index = os.path.join(att_dir, 'pic' + n + '.jpg')
        cv2.imwrite(vid_index, att_frame)
        ori_frame = np.uint8(im)
        ori_index = os.path.join(ori_dir, 'ori' + n + '.jpg')
        cv2.imwrite(ori_index, ori_frame)
        out_video.write((att_frame).astype(np.uint8))
    out_video.release()

from matplotlib.pyplot import figure, imshow, axis
from matplotlib.image import imread
from matplotlib import pyplot as plt
from PIL import Image

def showImagesHorizontally(list_of_files,startidx,endidx):
    fig,axs = plt.subplots(ncols=5,nrows=1,figsize=(8,8))
    axs.flatten()
    number_of_files = len(list_of_files)
    c=0
    for i in range(startidx,endidx):
        img=Image.open(list_of_files[i])
#         img=img.resize((500,500))
        axs[c].imshow(img)
        axs[c].axis('off')
        c+=1
    plt.show()


import os
from os import listdir
'''1Bov3GgLoY4
4yx7AdsRAOs
---1_cCGK4M

'''
# get the path/directory
heat_map_path = "/kaggle/working/dmrn/visual_att/attention_maps/4yx7AdsRAOs/"
orig_img_path = "/kaggle/working/dmrn/visual_att/original/4yx7AdsRAOs/"
heat_map_list=[]
orig_img_list=[]
# print(os.listdir(heat_map_path))
for images in os.listdir(heat_map_path):

    # check if the image ends with png
    if (images.endswith(".jpg")):
        heat_map_list.append(heat_map_path+images)

for images in os.listdir(orig_img_path):

    # check if the image ends with png
    if (images.endswith(".jpg")):
        orig_img_list.append(orig_img_path+images)
print(len(heat_map_list))
orig_img_list.sort()
heat_map_list.sort()

showImagesHorizontally(heat_map_list,50,55)
# print(heat_map_list[0])
showImagesHorizontally(orig_img_list,50,55)

from moviepy.editor import *
audio = VideoFileClip("/kaggle/input/avedataset-modified/AVE_Dataset/AVE/4yx7AdsRAOs.mp4")
print(type(audio))
audio = audio.audio
print(type(audio))
audio.write_audiofile("audio_file.mp3")

pip install llvmlite==0.31.0

from matplotlib.pyplot import figure, imshow, axis
from matplotlib.image import imread
from matplotlib import pyplot as plt
def showImagesHorizontally(list_of_files,startidx,endidx):
    fig,axs = plt.subplots(ncols=5,nrows=1,figsize=(8,8))
    axs.flatten()
    number_of_files = len(list_of_files)
    c=0
    for i in range(startidx,endidx):
        img=Image.open(list_of_files[i])
#         img=img.resize((500,500))
        axs[c].imshow(img)
        axs[c].axis('off')
        c+=1
    plt.show()


import os
from os import listdir
'''1Bov3GgLoY4
4yx7AdsRAOs
cjhDu8BokA0

'''
# get the path/directory
heat_map_path = "/kaggle/working/dmrn/visual_att/attention_maps/1Bov3GgLoY4/"
orig_img_path = "/kaggle/working/dmrn/visual_att/original/1Bov3GgLoY4/"
heat_map_list=[]
orig_img_list=[]
# print(os.listdir(heat_map_path))
for images in os.listdir(heat_map_path):

    # check if the image ends with png
    if (images.endswith(".jpg")):
        heat_map_list.append(heat_map_path+images)

for images in os.listdir(orig_img_path):

    # check if the image ends with png
    if (images.endswith(".jpg")):
        orig_img_list.append(orig_img_path+images)
print(len(heat_map_list))
orig_img_list.sort()
heat_map_list.sort()

showImagesHorizontally(heat_map_list,40,45)
# print(heat_map_list[0])
showImagesHorizontally(orig_img_list,40,45)

from matplotlib.pyplot import figure, imshow, axis
from matplotlib.image import imread
from matplotlib import pyplot as plt
from PIL import Image
def showImagesHorizontally(list_of_files,startidx,endidx):
    fig,axs = plt.subplots(ncols=5,nrows=1,figsize=(8,8))
    axs.flatten()
    number_of_files = len(list_of_files)
    c=0
    for i in range(startidx,endidx):
        img=Image.open(list_of_files[i])
#         img=img.resize((500,500))
        axs[c].imshow(img)
        axs[c].axis('off')
        c+=1
    plt.show()


import os
from os import listdir
'''1Bov3GgLoY4
4yx7AdsRAOs

'''
# get the path/directory
heat_map_path = "/kaggle/working/dmrn/visual_att/attention_maps/cjhDu8BokA0/"
orig_img_path = "/kaggle/working/dmrn/visual_att/original/cjhDu8BokA0/"
heat_map_list=[]
orig_img_list=[]
# print(os.listdir(heat_map_path))
for images in os.listdir(heat_map_path):

    # check if the image ends with png
    if (images.endswith(".jpg")):
        heat_map_list.append(heat_map_path+images)

for images in os.listdir(orig_img_path):

    # check if the image ends with png
    if (images.endswith(".jpg")):
        orig_img_list.append(orig_img_path+images)
print(len(heat_map_list))
orig_img_list.sort()
heat_map_list.sort()

showImagesHorizontally(heat_map_list,25,30)
# print(heat_map_list[0])
showImagesHorizontally(orig_img_list,25,30)

import os
from IPython.display import Image
import cv2
# img = cv2.imread('/kaggle/working/visual_att/original/--FJhcfXeow/ori0050.jpg')
# dimensions = img.shape
# print(dimensions)
Image(filename="/kaggle/working/visual_att/original/0RoAsnTIm5Q/ori0050.jpg")

from moviepy.editor import *
audio = VideoFileClip("/kaggle/input/avedataset-modified/AVE_Dataset/AVE/1Bov3GgLoY4.mp4")
print(type(audio))
audio = audio.audio
print(type(audio))
audio.write_audiofile("audio_file.mp3")

from asteroid.models import BaseModel
import soundfile as sf

AudioSuppressionModel = BaseModel.from_pretrained("groadabike/ConvTasNet_DAMP-VSEP_enhboth")
AudioSuppressionModel.separate('/kaggle/working/audio_file.mp3',resample=True)
overwrite=True

from IPython.display import display, Audio
display(Audio("audio_file.mp3"))
display(Audio("audio_file_est1.wav"))
display(Audio("audio_file_est2.wav"))

import librosa
x, sr = librosa.load("/kaggle/working/audio_file.mp3", sr=44100)

print(type(x), type(sr))
print(x.shape, sr)

plt.figure(figsize=(14, 5))
librosa.display.waveshow(x, sr=sr)

import librosa
x, sr = librosa.load("/kaggle/working/audio_file_est1.wav", sr=44100)

print(type(x), type(sr))
print(x.shape, sr)

plt.figure(figsize=(14, 5))
librosa.display.waveshow(x, sr=sr)

import librosa
x, sr = librosa.load("/kaggle/working/audio_file_est2.wav", sr=44100)

print(type(x), type(sr))
print(x.shape, sr)

plt.figure(figsize=(14, 5))
librosa.display.waveshow(x, sr=sr)

# display(Audio("./preview.wav"))
display(Audio("/kaggle/working/audio_file_est1.wav")) # Subject
display(Audio("/kaggle/working/audio_file_est2.wav")) #  noise

!cp /kaggle/input/audioseparation/female-female-mixture.wav /kaggle/working/female.wav

SpeakerSeparationModel = BaseModel.from_pretrained("mpariente/DPRNNTasNet-ks2_WHAM_sepclean")
SpeakerSeparationModel.separate("/kaggle/working/female.wav",resample=True)
overwrite=True# more accurate

display(Audio("/kaggle/working/female.wav"))
display(Audio("/kaggle/working/female_est1.wav"))
display(Audio("/kaggle/working/female_est2.wav"))

SpeakerSeparationModel = BaseModel.from_pretrained("mpariente/DPRNNTasNet-ks2_WHAM_sepclean")
SpeakerSeparationModel.separate("/kaggle/working/audio_file_est1.wav",resample=True)
overwrite=True# more accurate

display(Audio("/kaggle/working/audio_file_est1.wav"))
display(Audio("/kaggle/working/audio_file_est1_est1.wav"))
display(Audio("/kaggle/working/audio_file_est1_est2.wav"))

# for data transformation
import numpy as np
# for visualizing the data
import matplotlib.pyplot as plt
# for opening the media file
import scipy.io.wavfile as wavfile

import librosa
x, sr = librosa.load("/kaggle/working/female.wav", sr=44100)

print(type(x), type(sr))
print(x.shape, sr)

plt.figure(figsize=(14, 5))
librosa.display.waveshow(x, sr=sr)

import librosa
x, sr = librosa.load("/kaggle/working/female_est1.wav", sr=44100)

print(type(x), type(sr))
print(x.shape, sr)

plt.figure(figsize=(14, 5))
librosa.display.waveshow(x, sr=sr)

import librosa
x, sr = librosa.load("/kaggle/working/female_est2.wav", sr=44100)

print(type(x), type(sr))
print(x.shape, sr)

plt.figure(figsize=(14, 5))
librosa.display.waveshow(x, sr=sr)

person1vol=input("enter a number in range of -5 to 5")
print(person1vol)
person2vol=input("enter a number in range of -5 to 5")
print(person2vol)
noisevol=input("enter a number in range of -5 to 5")
print(noisevol)

print(person1vol)
print(person2vol)
print(noisevol)

"""Audio mixing"""

# import ipywidgets
# from IPython.display import display

# def responsive_slider(Person1):
#     return 2 * Person1

# person1slider = ipywidgets.FloatSlider(min=0, max=8, step=0.1, value=3)
# display(widgets.interactive(responsive_slider, Person1=person1slider))

# def responsive_slider(Person2):
#     return 2 * Person2

# person2slider = ipywidgets.FloatSlider(min=0, max=8, step=0.1, value=3)
# display(widgets.interactive(responsive_slider, Person2=person2slider))

# def responsive_slider(noise):
#     return 2 * noise

# noiseSlider = ipywidgets.FloatSlider(min=0, max=8, step=0.1, value=3)
# display(widgets.interactive(responsive_slider, noise=noiseSlider))

from pydub import AudioSegment
audio1=AudioSegment.from_wav("/kaggle/working/female_est2.wav")
# audio2=AudioSegment.from_wav("./female-female-mixture_est2.wav")
audio1=audio1 - 12
# audio1=audio1.overlay(audio2)
audio1.export("./audio1.wav",format="wav")

from IPython.display import display, Audio
display(Audio("./audio1.wav"))

from pydub import AudioSegment
audio1=AudioSegment.from_wav("/kaggle/working/audio_file_est1_est1.wav")
audio2=AudioSegment.from_wav("/kaggle/working/audio_file_est1_est2.wav")
noise=AudioSegment.from_wav("audio_file_est2.wav")
audio1=audio1 +person1vol
audio2=audio2+person2vol
noise=noise+noisevol
audio1=audio1.overlay(audio2)
audio1=audio1.overlay(noise)
audio1.export("./output.mp3",format="mp3")

from IPython.display import display, Audio
display(Audio("./output.mp3"))

from pydub import AudioSegment
import subprocess
videofile = "/kaggle/working/4yx7AdsRAOs.mp4"
audiofile = "/kaggle/working/audio_file.mp3"
outputfile = "/kaggle/working/4yx7AdsRAOs_joined.mp4"
codec = "copy"
subprocess.run(f"ffmpeg -i {videofile} -i {audiofile} -c {codec} {outputfile} -y",shell=True)

!rm /kaggle/working/joinedVideo.mp4





