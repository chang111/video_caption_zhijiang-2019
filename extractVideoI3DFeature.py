import argparse
import os
import glob
import shutil
import subprocess
import numpy as np
import torch
from scipy import misc
from tqdm import tqdm
from src.i3dpt import I3D
from matplotlib.pyplot import imread

FPS = 25
MAX_INTERVAL = 400
OVERLAP = 25
rgb_pt_checkpoint = 'save_model/model_rgb.pth'
#os.environ["CUDA_VISIBLE_DEVICES"] = "1"


def get_features(sample, model):
    sample = sample.transpose(0, 4, 1, 2, 3)
    sample_var = torch.autograd.Variable(torch.from_numpy(sample).cuda())
    out_var = model.extract(sample_var)
    out_tensor = out_var.data.cpu()
    return out_tensor.numpy()


def read_video(video_dir):
    frames = video_dir
    data = []
    for i, frame in enumerate(frames):
        I = imread(frame)
        if len(I.shape) == 2:
            I = I[:, :, np.newaxis]
            I = np.concatenate((I, I, I), axis=2)
        I = (I.astype('float32') / 255.0 - 0.5) * 2
        data.append(I)
    if len(data) <= 0:
        return None
    res = np.asarray(data)[np.newaxis, :, :, :, :]
    return res


def extract_frames(video, dst):
    with open(os.devnull, "w") as ffmpeg_log:
        if os.path.exists(dst):
            shutil.rmtree(dst)
        os.makedirs(dst)
        video_to_frames_command = ["ffmpeg",
                                   # (optional) overwrite output file if it exists
                                   '-y',
                                   '-i', video,  # input file
                                   '-vf', "scale=256:256",  # input file
                                   '-qscale:v', "2",  # quality for JPEG
                                   '-vframes', '160',
                                   '{0}/%06d.jpg'.format(dst)]
        subprocess.call(video_to_frames_command,
                        stdout=ffmpeg_log, stderr=ffmpeg_log)


def run(args):
    # Run RGB model
    i3d_rgb = I3D(num_classes=400, modality='rgb')
    i3d_rgb.eval()
    i3d_rgb.load_state_dict(torch.load(args.rgb_weights_path))
    i3d_rgb.cuda()

    # read the video list which records the readable video
    video_list = glob.glob(args.input_path + '*')

    for vid in tqdm(video_list):
        extract_frames(vid, 'data/tmp_image')
        image_list = sorted(glob.glob(os.path.join('data/tmp_image', '*.jpg')))
        samples = np.round(np.linspace(
            0, len(image_list) - 1, 160))
        image_list = [image_list[int(sample)] for sample in samples]
        clip = read_video(image_list)
        if clip is None:
            print(vid)
            continue
        clip_len = clip.shape[1]
        if clip_len <= MAX_INTERVAL:
            features = get_features(clip, i3d_rgb)
        else:
            tmp_1 = 0
            features = []
            while True:
                tmp_2 = tmp_1 + MAX_INTERVAL
                tmp_2 = min(tmp_2, clip_len)
                feat = get_features(clip[:, tmp_1:tmp_2], i3d_rgb)
                features.append(feat)
                if tmp_2 == clip_len:
                    break
                tmp_1 = max(0, tmp_2 - OVERLAP)
            features = np.concatenate(features, axis=1)
        name = vid.split('/')[-1]
        name = name[:-4]
        name = name + '.npy'
        if not os.path.exists(args.save_path):
            os.makedirs(args.save_path)
        np.save(args.save_path+name, features)
        


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # RGB arguments
    parser.add_argument(
        '--rgb', action='store_true', help='Evaluate RGB pretrained network')
    parser.add_argument(
        '--rgb_weights_path',
        type=str,
        default='save_model/model_rgb.pth',
        help='Path to rgb model state_dict')

    parser.add_argument('--input_path', type=str, default='data/tianchi_test/')
    parser.add_argument('--save_path', type=str, default="data/tianchi_test_i3d_features/")
    args = parser.parse_args()
    run(args)
