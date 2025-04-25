import os.path
import torch
import os
import numpy as np
import time
from talkingface.run_utils import smooth_array, video_pts_process
from talkingface.run_utils import mouth_replace, prepare_video_data
from talkingface.utils import generate_face_mask, INDEX_LIPS_OUTER
from talkingface.data.few_shot_dataset import select_ref_index,get_ref_images_fromVideo,generate_input, generate_input_pixels
device = "cuda" if torch.cuda.is_available() else "cpu"
import pickle
import cv2

face_mask = generate_face_mask()


miss_list = ['source_in_conv.0.conv.weight', 'source_in_conv.0.conv.bias', 'source_in_conv.0.norm.weight', 'source_in_conv.0.norm.bias', 'source_in_conv.0.norm.running_mean', 'source_in_conv.0.norm.running_var', 'source_in_conv.0.norm.num_batches_tracked', 'source_in_conv.1.conv.weight', 'source_in_conv.1.conv.bias', 'source_in_conv.1.norm.weight', 'source_in_conv.1.norm.bias', 'source_in_conv.1.norm.running_mean', 'source_in_conv.1.norm.running_var', 'source_in_conv.1.norm.num_batches_tracked', 'source_in_conv.2.conv.weight', 'source_in_conv.2.conv.bias', 'source_in_conv.2.norm.weight', 'source_in_conv.2.norm.bias', 'source_in_conv.2.norm.running_mean', 'source_in_conv.2.norm.running_var', 'source_in_conv.2.norm.num_batches_tracked', 'ref_in_conv.0.conv.weight', 'ref_in_conv.0.conv.bias', 'ref_in_conv.0.norm.weight', 'ref_in_conv.0.norm.bias', 'ref_in_conv.0.norm.running_mean', 'ref_in_conv.0.norm.running_var', 'ref_in_conv.0.norm.num_batches_tracked', 'ref_in_conv.1.conv.weight', 'ref_in_conv.1.conv.bias', 'ref_in_conv.1.norm.weight', 'ref_in_conv.1.norm.bias', 'ref_in_conv.1.norm.running_mean', 'ref_in_conv.1.norm.running_var', 'ref_in_conv.1.norm.num_batches_tracked', 'ref_in_conv.2.conv.weight', 'ref_in_conv.2.conv.bias', 'ref_in_conv.2.norm.weight', 'ref_in_conv.2.norm.bias', 'ref_in_conv.2.norm.running_mean', 'ref_in_conv.2.norm.running_var', 'ref_in_conv.2.norm.num_batches_tracked', 'trans_conv.0.conv.weight', 'trans_conv.0.conv.bias', 'trans_conv.0.norm.weight', 'trans_conv.0.norm.bias', 'trans_conv.0.norm.running_mean', 'trans_conv.0.norm.running_var', 'trans_conv.0.norm.num_batches_tracked', 'trans_conv.1.conv.weight', 'trans_conv.1.conv.bias', 'trans_conv.1.norm.weight', 'trans_conv.1.norm.bias', 'trans_conv.1.norm.running_mean', 'trans_conv.1.norm.running_var', 'trans_conv.1.norm.num_batches_tracked', 'trans_conv.2.conv.weight', 'trans_conv.2.conv.bias', 'trans_conv.2.norm.weight', 'trans_conv.2.norm.bias', 'trans_conv.2.norm.running_mean', 'trans_conv.2.norm.running_var', 'trans_conv.2.norm.num_batches_tracked', 'trans_conv.3.conv.weight', 'trans_conv.3.conv.bias', 'trans_conv.3.norm.weight', 'trans_conv.3.norm.bias', 'trans_conv.3.norm.running_mean', 'trans_conv.3.norm.running_var', 'trans_conv.3.norm.num_batches_tracked', 'trans_conv.4.conv.weight', 'trans_conv.4.conv.bias', 'trans_conv.4.norm.weight', 'trans_conv.4.norm.bias', 'trans_conv.4.norm.running_mean', 'trans_conv.4.norm.running_var', 'trans_conv.4.norm.num_batches_tracked', 'trans_conv.5.conv.weight', 'trans_conv.5.conv.bias', 'trans_conv.5.norm.weight', 'trans_conv.5.norm.bias', 'trans_conv.5.norm.running_mean', 'trans_conv.5.norm.running_var', 'trans_conv.5.norm.num_batches_tracked', 'trans_conv.6.conv.weight', 'trans_conv.6.conv.bias', 'trans_conv.6.norm.weight', 'trans_conv.6.norm.bias', 'trans_conv.6.norm.running_mean', 'trans_conv.6.norm.running_var', 'trans_conv.6.norm.num_batches_tracked', 'trans_conv.7.conv.weight', 'trans_conv.7.conv.bias', 'trans_conv.7.norm.weight', 'trans_conv.7.norm.bias', 'trans_conv.7.norm.running_mean', 'trans_conv.7.norm.running_var', 'trans_conv.7.norm.num_batches_tracked', 'trans_conv.8.conv.weight', 'trans_conv.8.conv.bias', 'trans_conv.8.norm.weight', 'trans_conv.8.norm.bias', 'trans_conv.8.norm.running_mean', 'trans_conv.8.norm.running_var', 'trans_conv.8.norm.num_batches_tracked', 'appearance_conv_list.0.0.conv1.weight', 'appearance_conv_list.0.0.conv1.bias', 'appearance_conv_list.0.0.conv2.weight', 'appearance_conv_list.0.0.conv2.bias', 'appearance_conv_list.0.0.norm1.weight', 'appearance_conv_list.0.0.norm1.bias', 'appearance_conv_list.0.0.norm1.running_mean', 'appearance_conv_list.0.0.norm1.running_var', 'appearance_conv_list.0.0.norm1.num_batches_tracked', 'appearance_conv_list.0.0.norm2.weight', 'appearance_conv_list.0.0.norm2.bias', 'appearance_conv_list.0.0.norm2.running_mean', 'appearance_conv_list.0.0.norm2.running_var', 'appearance_conv_list.0.0.norm2.num_batches_tracked', 'appearance_conv_list.0.1.conv1.weight', 'appearance_conv_list.0.1.conv1.bias', 'appearance_conv_list.0.1.conv2.weight', 'appearance_conv_list.0.1.conv2.bias', 'appearance_conv_list.0.1.norm1.weight', 'appearance_conv_list.0.1.norm1.bias', 'appearance_conv_list.0.1.norm1.running_mean', 'appearance_conv_list.0.1.norm1.running_var', 'appearance_conv_list.0.1.norm1.num_batches_tracked', 'appearance_conv_list.0.1.norm2.weight', 'appearance_conv_list.0.1.norm2.bias', 'appearance_conv_list.0.1.norm2.running_mean', 'appearance_conv_list.0.1.norm2.running_var', 'appearance_conv_list.0.1.norm2.num_batches_tracked', 'appearance_conv_list.1.0.conv1.weight', 'appearance_conv_list.1.0.conv1.bias', 'appearance_conv_list.1.0.conv2.weight', 'appearance_conv_list.1.0.conv2.bias', 'appearance_conv_list.1.0.norm1.weight', 'appearance_conv_list.1.0.norm1.bias', 'appearance_conv_list.1.0.norm1.running_mean', 'appearance_conv_list.1.0.norm1.running_var', 'appearance_conv_list.1.0.norm1.num_batches_tracked', 'appearance_conv_list.1.0.norm2.weight', 'appearance_conv_list.1.0.norm2.bias', 'appearance_conv_list.1.0.norm2.running_mean', 'appearance_conv_list.1.0.norm2.running_var', 'appearance_conv_list.1.0.norm2.num_batches_tracked', 'appearance_conv_list.1.1.conv1.weight', 'appearance_conv_list.1.1.conv1.bias', 'appearance_conv_list.1.1.conv2.weight', 'appearance_conv_list.1.1.conv2.bias', 'appearance_conv_list.1.1.norm1.weight', 'appearance_conv_list.1.1.norm1.bias', 'appearance_conv_list.1.1.norm1.running_mean', 'appearance_conv_list.1.1.norm1.running_var', 'appearance_conv_list.1.1.norm1.num_batches_tracked', 'appearance_conv_list.1.1.norm2.weight', 'appearance_conv_list.1.1.norm2.bias', 'appearance_conv_list.1.1.norm2.running_mean', 'appearance_conv_list.1.1.norm2.running_var', 'appearance_conv_list.1.1.norm2.num_batches_tracked', 'adaAT.commn_linear.0.weight', 'adaAT.commn_linear.0.bias', 'adaAT.scale.0.weight', 'adaAT.scale.0.bias', 'adaAT.rotation.0.weight', 'adaAT.rotation.0.bias', 'adaAT.translation.0.weight', 'adaAT.translation.0.bias', 'out_conv.0.conv.weight', 'out_conv.0.conv.bias', 'out_conv.0.norm.weight', 'out_conv.0.norm.bias', 'out_conv.0.norm.running_mean', 'out_conv.0.norm.running_var', 'out_conv.0.norm.num_batches_tracked', 'out_conv.1.conv.weight', 'out_conv.1.conv.bias', 'out_conv.1.norm.weight', 'out_conv.1.norm.bias', 'out_conv.1.norm.running_mean', 'out_conv.1.norm.running_var', 'out_conv.1.norm.num_batches_tracked', 'out_conv.2.conv1.weight', 'out_conv.2.conv1.bias', 'out_conv.2.conv2.weight', 'out_conv.2.conv2.bias', 'out_conv.2.norm1.weight', 'out_conv.2.norm1.bias', 'out_conv.2.norm1.running_mean', 'out_conv.2.norm1.running_var', 'out_conv.2.norm1.num_batches_tracked', 'out_conv.2.norm2.weight', 'out_conv.2.norm2.bias', 'out_conv.2.norm2.running_mean', 'out_conv.2.norm2.running_var', 'out_conv.2.norm2.num_batches_tracked', 'out_conv.3.conv.weight', 'out_conv.3.conv.bias', 'out_conv.3.norm.weight', 'out_conv.3.norm.bias', 'out_conv.3.norm.running_mean', 'out_conv.3.norm.running_var', 'out_conv.3.norm.num_batches_tracked', 'out_conv.4.weight', 'out_conv.4.bias']


class RenderModel:
    def __init__(self):
        self.__net = None

        self.__pts_driven = None
        self.__mat_list = None
        self.__pts_normalized_list = None
        self.__face_mask_pts = None
        self.__ref_img = None
        self.__cap_input = None
        self.frame_index = 0
        self.__mouth_coords_array = None

    def get_net(self):
        return self.__net

    def loadModel(self, ckpt_path):
        from talkingface.models.DINet import DINet_five_Ref as DINet
        n_ref = 5
        source_channel = 6
        ref_channel = n_ref * 6
        self.__net = DINet(source_channel, ref_channel).cuda()
        checkpoint = torch.load(ckpt_path)

        
        try:
            self.__net.load_state_dict(checkpoint)
        except Exception as e:
            print(str(e))

            checkpoint_old = torch.load("checkpoint/render_backup.pth")
            for x in miss_list:
                checkpoint_old[x] = checkpoint['state_dict']["net_g"][x]

            self.__net.load_state_dict(checkpoint_old)
            



        self.__net.eval()

    def reset_charactor(self, video_path, Path_pkl, ref_img_index_list = None):
        if self.__cap_input is not None:
            self.__cap_input.release()

        self.__pts_driven, self.__mat_list,self.__pts_normalized_list, self.__face_mask_pts, self.__ref_img, self.__cap_input = \
            prepare_video_data(video_path, Path_pkl, ref_img_index_list)

        ref_tensor = torch.from_numpy(self.__ref_img / 255.).float().permute(2, 0, 1).unsqueeze(0).cuda()
        self.__net.ref_input(ref_tensor)

        x_min, x_max = np.min(self.__pts_normalized_list[:, INDEX_LIPS_OUTER, 0]), np.max(self.__pts_normalized_list[:, INDEX_LIPS_OUTER, 0])
        y_min, y_max = np.min(self.__pts_normalized_list[:, INDEX_LIPS_OUTER, 1]), np.max(self.__pts_normalized_list[:, INDEX_LIPS_OUTER, 1])
        z_min, z_max = np.min(self.__pts_normalized_list[:, INDEX_LIPS_OUTER, 2]), np.max(self.__pts_normalized_list[:, INDEX_LIPS_OUTER, 2])

        x_mid,y_mid,z_mid = (x_min + x_max)/2, (y_min + y_max)/2, (z_min + z_max)/2
        x_len, y_len, z_len = (x_max - x_min)/2, (y_max - y_min)/2, (z_max - z_min)/2
        x_min, x_max = x_mid - x_len*0.9, x_mid + x_len*0.9
        y_min, y_max = y_mid - y_len*0.9, y_mid + y_len*0.9
        z_min, z_max = z_mid - z_len*0.9, z_mid + z_len*0.9

        # print(face_personal.shape, x_min, x_max, y_min, y_max, z_min, z_max)
        coords_array = np.zeros([100, 150, 4])
        for i in range(100):
            for j in range(150):
                coords_array[i, j, 0] = j/149
                coords_array[i, j, 1] = i/100
                # coords_array[i, j, 2] = int((-75 + abs(j - 75))*(2./3))
                coords_array[i, j, 2] = ((j - 75)/ 75) ** 2
                coords_array[i, j, 3] = 1

        coords_array = coords_array*np.array([x_max - x_min, y_max - y_min, z_max - z_min, 1]) + np.array([x_min, y_min, z_min, 0])
        self.__mouth_coords_array = coords_array.reshape(-1, 4).transpose(1, 0)



    def interface(self, mouth_frame):
        vid_frame_count = self.__cap_input.get(cv2.CAP_PROP_FRAME_COUNT)
        if self.frame_index % vid_frame_count == 0:
            self.__cap_input.set(cv2.CAP_PROP_POS_FRAMES, 0)  # 设置要获取的帧号
        ret, frame = self.__cap_input.read()  # 按帧读取视频

        epoch = self.frame_index // len(self.__mat_list)
        if epoch % 2 == 0:
            new_index = self.frame_index % len(self.__mat_list)
        else:
            new_index = -1 - self.frame_index % len(self.__mat_list)

        # print(self.__face_mask_pts.shape, "ssssssss")
        source_img, target_img, crop_coords = generate_input_pixels(frame, self.__pts_driven[new_index], self.__mat_list[new_index],
                                                                    mouth_frame, self.__face_mask_pts[new_index],
                                                                    self.__mouth_coords_array)

        # tensor
        source_tensor = torch.from_numpy(source_img / 255.).float().permute(2, 0, 1).unsqueeze(0).cuda()
        target_tensor = torch.from_numpy(target_img / 255.).float().permute(2, 0, 1).unsqueeze(0).cuda()

        source_tensor, source_prompt_tensor = source_tensor[:, :3], source_tensor[:, 3:]
        fake_out = self.__net.interface(source_tensor, source_prompt_tensor)

        image_numpy = fake_out.detach().squeeze(0).cpu().float().numpy()
        image_numpy = np.transpose(image_numpy, (1, 2, 0)) * 255.0
        image_numpy = image_numpy.clip(0, 255)
        image_numpy = image_numpy.astype(np.uint8)

        image_numpy = target_img * face_mask + image_numpy * (1 - face_mask)

        img_bg = frame
        x_min, y_min, x_max, y_max = crop_coords

        img_face = cv2.resize(image_numpy, (x_max - x_min, y_max - y_min))
        img_bg[y_min:y_max, x_min:x_max] = img_face
        self.frame_index += 1
        return img_bg

    def save(self, path):
        torch.save(self.__net.state_dict(), path)