import json
import os, io, csv, math, random
import numpy as np
import torchvision
from einops import rearrange
from decord import VideoReader
from os.path import join as opj
import imageio
import torch
import torchvision.transforms as transforms
from torch.utils.data.dataset import Dataset
from tqdm import tqdm
from PIL import Image
from collections import deque
from opensora.utils.dataset_utils import DecordInit
from opensora.utils.utils import text_preprocessing
from concurrent.futures import ThreadPoolExecutor
import concurrent.futures
import multiprocessing
import cv2
import copy

from s3torchconnector import S3MapDataset, S3IterableDataset
import gc
# from memory_profiler import profile
import tempfile

def random_video_noise(t, c, h, w):
    vid = torch.rand(t, c, h, w) * 255.0
    vid = vid.to(torch.uint8)
    return vid



class S3MP4Dataset():
    def __init__(self, urls, region):
        self.dataset = []
        self.lookup_dict = {}
        for url in urls:
            raw_dataset = S3MapDataset.from_prefix(url, region=region)
            dataset = self.filter_mp4_files(raw_dataset)
            self.dataset.extend(dataset)
            num_non_mp4_files = len(raw_dataset) - len(dataset)
            print(f"Number of non-mp4 files filtered for {url}: {num_non_mp4_files}")
            print(f"Number of mp4 files for {url}: {len(dataset)}")
        print(f"Number of total mp4 files: {len(self.dataset)}")

        for idx, obj in enumerate(self.dataset):
            parts = obj.key.split('/')
            class_name = parts[-2]
            only_filename = parts[-1]
            self.lookup_dict[(class_name, only_filename)] = idx

    def filter_mp4_files(self, raw_dataset):
        """Yield only items that are .mp4 files."""
        processed = []
        for item in raw_dataset:
            if item.key.endswith('.mp4'):
                processed.append(item)
        return processed
                
def tv_read(path, frame_idx, num_frames):
    vframes, aframes, info = torchvision.io.read_video(filename=path, pts_unit='sec', output_format='TCHW')
    start_frame_ind, end_frame_ind = frame_idx.split(':')
    start_frame_ind, end_frame_ind = int(start_frame_ind), int(end_frame_ind)
    # assert end_frame_ind - start_frame_ind >= self.num_frames
    frame_indice = np.linspace(start_frame_ind, end_frame_ind - 1, num_frames, dtype=int)
    # frame_indice = np.linspace(0, 63, self.num_frames, dtype=int)

    video = vframes[frame_indice]  # (T, C, H, W)

    return video

def cv_read(video_path, frame_idx, num_frames):
    # frame index processsing, same as others
    start_frame_ind, end_frame_ind = frame_idx.split(':')
    start_frame_ind, end_frame_ind = int(start_frame_ind), int(end_frame_ind)
    # assert end_frame_ind - start_frame_ind >= self.num_frames
    frame_numbers = np.linspace(start_frame_ind, end_frame_ind - 1, num_frames, dtype=int)
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error opening video file: {video_path}")
        return
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    valid_frame_numbers = [fn for fn in frame_numbers if 0 <= fn < total_frames]
    if len(valid_frame_numbers) != len(frame_numbers):
        print(f"Some frame numbers are out of range in {video_path}")
    frame_list = []
    for frame_number in valid_frame_numbers:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        ret, frame = cap.read()
        if not ret:
            print(f"Error reading frame number {frame_number} from {video_path}")
            continue
        # Convert BGR (cv2) to RGB (standard)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_list.append(rgb_frame)
    cap.release()
    # cv2.destroyAllWindows()
    # print(f"Released video file: {video_path}")
    raw_video = np.array(frame_list)
    raw_video = raw_video.transpose(0, 3, 1, 2) # T H W C -> T C H W
    return torch.from_numpy(raw_video)

class S3_T2V_dataset(Dataset):
    def __init__(self, args, transform, temporal_sample, tokenizer, rank=0, video_decoder='decord'):
        self.image_data = args.image_data
        self.video_data = args.video_data
        self.num_frames = args.num_frames
        self.use_image_num = args.use_image_num
        self.use_img_from_vid = args.use_img_from_vid
        self.transform = transform
        self.temporal_sample = temporal_sample
        self.tokenizer = tokenizer
        self.model_max_length = args.model_max_length
        self.video_decoder = video_decoder
        self.valid_samples = deque(maxlen=4)  # a fallback data queue
        URIs = [
            "s3://prj-zihan/opensora-dataset/v1.1/mixkit",
            "s3://prj-zihan/opensora-dataset/v1.1/pexels",
            "s3://prj-zihan/opensora-dataset/v1.1/pixabay_v2",
        ]
        REGION = "us-west-2" # check aws s3 bucket region
        self.s3dataset = S3MP4Dataset(URIs, REGION)
        if video_decoder == 'decord':
            self.v_decoder = DecordInit(num_threads=1, device_id=rank, device_type='gpu')

        if self.num_frames != 1:
            self.vid_cap_list = self.get_vid_cap_list()
            if self.use_image_num != 0 and not self.use_img_from_vid:
                self.img_cap_list = self.get_img_cap_list()
        else:
            self.img_cap_list = self.get_img_cap_list()   

        self._filter_nonexist_files()
        print(f"Number of videos: {len(self.vid_cap_list)}")

    # def _filter_nonexist_files(self,):
    #     print("Filtering non-existing videos...")
    #     # Extract key values from dataset
    #     dataset = self.s3dataset.dataset
    #     key_values = set()
    #     for item in dataset:
    #         parts = item.key.split('/')
    #         key_value = '/'.join(parts[-2:])  # Join the second to last and last part
    #         key_values.add(key_value)

    #     # Filter vid_cap_list by checking if any key_value exists in the path of each entry
    #     filtered_vid_cap_list = [entry for entry in self.vid_cap_list if any(key_value in entry['path'] for key_value in key_values)]
    #     print("Number of videos after filtering:", len(filtered_vid_cap_list))
    #     print("Number of non-existing videos filtered:", len(self.vid_cap_list) - len(filtered_vid_cap_list))
        # self.vid_cap_list = filtered_vid_cap_list


    def _filter_nonexist_files(self,):
        """ A multithread fast version"""
        print("Filtering non-existing videos...")
        dataset = self.s3dataset.dataset
        # Extract key values from dataset
        key_values = set()
        for item in dataset:
            parts = item.key.split('/')
            key_value = '/'.join(parts[-2:])  # Join the second to last and last part
            key_values.add(key_value)

        def filter_entry(entry):
            """ Filter individual entries, checking against the global key_values. """
            return '/'.join(entry['path'].split('/')[-2:]) in key_values

        # Use ThreadPoolExecutor to filter vid_cap_list in parallel
        with ThreadPoolExecutor() as executor:
            results = list(executor.map(filter_entry, self.vid_cap_list))

        # Combine entries with their filter results
        filtered_vid_cap_list = [entry for entry, include in zip(self.vid_cap_list, results) if include]
        print("Number of videos before filtering:", len(self.vid_cap_list))
        print("Number of videos after filtering:", len(filtered_vid_cap_list))
        print("Number of non-existing videos filtered:", len(self.vid_cap_list) - len(filtered_vid_cap_list))
        self.vid_cap_list = filtered_vid_cap_list

    def get_fallback_data(self):
        # Return a random sample from the queue if available
        # assert len(self.valid_samples) > 0, "Fallback data requested but no valid samples are available."
        if len(self.valid_samples) > 0:
            return random.choice(self.valid_samples)
        else:
            print("Fallback data requested but no valid samples are available.")
            return None

    def __len__(self):
        if self.num_frames != 1:
            return len(self.vid_cap_list)
        else:
            return len(self.img_cap_list)
        
    def __getitem__(self, idx):
        try:
            video_data, image_data = {}, {}
            if self.num_frames != 1:
                video_data = self.get_video(idx)
                if self.use_image_num != 0:
                    if self.use_img_from_vid:
                        image_data = self.get_image_from_video(video_data)
                    else:
                        image_data = self.get_image(idx)
            else:
                image_data = self.get_image(idx)  # 1 frame video as image
            return dict(video_data=video_data, image_data=image_data)
        except Exception as e:
            print(f'Error with {e}')
            return self.__getitem__(random.randint(0, self.__len__() - 1))


    # def __getitem__(self, idx):
    #     max_attempts = 3
    #     for _ in range(max_attempts):
    #         try:
    #             video_data, image_data = {}, {}
    #             if self.num_frames != 1:
    #                 video_data = self.get_video(idx)
    #                 if video_data is None:
    #                     idx = random.randint(0, self.__len__() - 1)
    #                     continue
    #                 if self.use_image_num != 0:
    #                     if self.use_img_from_vid:
    #                         image_data = self.get_image_from_video(video_data)
    #                     else:
    #                         image_data = self.get_image(idx)
    #             else:
    #                 image_data = self.get_image(idx)  # 1 frame video as image
    #             valid_data = dict(video_data=video_data, image_data=image_data)
    #             self.valid_samples.append(valid_data)
    #             return valid_data
    #         except Exception as e:
    #             print(f'Error with {e}')
    #             idx = random.randint(0, self.__len__() - 1)
    #     return self.get_fallback_data()

    # def __getitem__(self, idx):
    #     # try:
    #     video_data, image_data = {}, {}
    #     if self.num_frames != 1:
    #         video_data = self.get_video(idx)
    #         if video_data is None:
    #             return self.get_fallback_data()  # TODO need to fix
    #         if self.use_image_num != 0:
    #             if self.use_img_from_vid:
    #                 image_data = self.get_image_from_video(video_data)
    #             else:
    #                 image_data = self.get_image(idx)
    #     else:
    #         image_data = self.get_image(idx)  # 1 frame video as image
    #     valid_data = dict(video_data=video_data, image_data=image_data)
    #     del video_data
    #     # self.valid_samples.append(valid_data)
    #     return valid_data
    #     # except Exception as e:
    #     #     print(f'Error with {e}')
    #     # return self.get_fallback_data()

    def get_path(self, obj):
        """ mmap and chunk of tempfile does not help with memory leakage """
        obj2 = copy.deepcopy(obj)  # this is essential to avoid memory leakage; copy.copy() is not enough
        obj2.seek(0)
        video_data = obj2.read()
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmpfile:
            tmpfile.write(video_data)
            tmpfile.flush()  # Ensure data is written to disk
            tmpfile_path = tmpfile.name
        del video_data
        return tmpfile_path

    # @profile
    def get_video(self, idx):
        # video = random.choice([random_video_noise(65, 3, 720, 360) * 255, random_video_noise(65, 3, 1024, 1024), random_video_noise(65, 3, 360, 720)])
        # # print('random shape', video.shape)
        # input_ids = torch.ones(1, 120).to(torch.long).squeeze(0)
        # cond_mask = torch.cat([torch.ones(1, 60).to(torch.long), torch.ones(1, 60).to(torch.long)], dim=1).squeeze(0)
        
        video_path = self.vid_cap_list[idx]['path']
        parts = video_path.split('/')
        class_name = parts[-2]
        only_filename = parts[-1]
        video_idx = self.s3dataset.lookup_dict.get((class_name, only_filename), None)
        if video_idx is None:
            print(f"KeyError: ({class_name}, {only_filename}) not found in the dataset.")
            return None
        obj = self.s3dataset.dataset[video_idx]
        frame_idx = self.vid_cap_list[idx]['frame_idx']
        obj_video_path = self.get_path(obj)

        if self.video_decoder == 'decord':
            video = self.decord_read(obj_video_path, frame_idx)  # does not work with mp.spawn
        elif self.video_decoder == 'cv2':
            video = cv_read(obj_video_path, frame_idx, self.num_frames)
        else:
            # video = self.tv_read(video_path, frame_idx)

            # https://stackoverflow.com/questions/15455048/releasing-memory-in-python
            # Use a ProcessPoolExecutor to run the tv_read function in a separate process
            with concurrent.futures.ProcessPoolExecutor(max_workers=1) as executor:
                # future = executor.submit(self.tv_read, obj_video_path, frame_idx)  # error: cannot spawn s3client...
                future = executor.submit(tv_read, obj_video_path, frame_idx, self.num_frames)
                video = future.result()  # Wait for the child process to complete and get the result

            # with multiprocessing.Pool(1) as pool:
            #     # Submit the read function to the pool
            #     result = pool.apply_async(read, (obj_video_path, frame_idx, 65))
            #     # Retrieve the result of the function call
            #     video = result.get()  # This will block until the function completes

        # Delete the temporary file
        os.unlink(obj_video_path)
        gc.collect()

        video = self.transform(video)  # T C H W -> T C H W

        video = video.transpose(0, 1)  # T C H W -> C T H W
        text = self.vid_cap_list[idx]['cap']

        text = text_preprocessing(text)
        text_tokens_and_mask = self.tokenizer(
            text,
            max_length=self.model_max_length,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            add_special_tokens=True,
            return_tensors='pt'
        )
        input_ids = text_tokens_and_mask['input_ids']
        cond_mask = text_tokens_and_mask['attention_mask']
        # return dict(video=video, input_ids=input_ids, cond_mask=cond_mask)
        output = dict(video=video, input_ids=input_ids, cond_mask=cond_mask, text=text)
        del video
        return output

    def get_image_from_video(self, video_data):
        select_image_idx = np.linspace(0, self.num_frames-1, self.use_image_num, dtype=int)
        assert self.num_frames >= self.use_image_num
        image = [video_data['video'][:, i:i+1] for i in select_image_idx]  # num_img [c, 1, h, w]
        input_ids = video_data['input_ids'].repeat(self.use_image_num, 1)  # self.use_image_num, l
        cond_mask = video_data['cond_mask'].repeat(self.use_image_num, 1)  # self.use_image_num, l
        return dict(image=image, input_ids=input_ids, cond_mask=cond_mask)

    def get_image(self, idx):
        idx = idx % len(self.img_cap_list)  # out of range
        image_data = self.img_cap_list[idx]  # [{'path': path, 'cap': cap}, ...]
        
        image = [Image.open(i['path']).convert('RGB') for i in image_data] # num_img [h, w, c]
        image = [torch.from_numpy(np.array(i)) for i in image] # num_img [h, w, c]
        image = [rearrange(i, 'h w c -> c h w').unsqueeze(0) for i in image] # num_img [1 c h w]
        image = [self.transform(i) for i in image]  # num_img [1 C H W] -> num_img [1 C H W]
        # image = [torch.rand(1, 3, 512, 512) for i in image_data]
        image = [i.transpose(0, 1) for i in image]  # num_img [1 C H W] -> num_img [C 1 H W]

        caps = [i['cap'] for i in image_data]
        text = [text_preprocessing(cap) for cap in caps]
        input_ids, cond_mask = [], []
        for t in text:
            text_tokens_and_mask = self.tokenizer(
                t,
                max_length=self.model_max_length,
                padding='max_length',
                truncation=True,
                return_attention_mask=True,
                add_special_tokens=True,
                return_tensors='pt'
            )
            input_ids.append(text_tokens_and_mask['input_ids'])
            cond_mask.append(text_tokens_and_mask['attention_mask'])
        input_ids = torch.cat(input_ids)  # self.use_image_num, l
        cond_mask = torch.cat(cond_mask)  # self.use_image_num, l
        return dict(image=image, input_ids=input_ids, cond_mask=cond_mask)

    def tv_read(self, path, frame_idx=None):
        vframes, aframes, info = torchvision.io.read_video(filename=path, pts_unit='sec', output_format='TCHW')
        total_frames = len(vframes)
        if frame_idx is None:
            start_frame_ind, end_frame_ind = self.temporal_sample(total_frames)
        else:
            start_frame_ind, end_frame_ind = frame_idx.split(':')
            start_frame_ind, end_frame_ind = int(start_frame_ind), int(end_frame_ind)
        # assert end_frame_ind - start_frame_ind >= self.num_frames
        frame_indice = np.linspace(start_frame_ind, end_frame_ind - 1, self.num_frames, dtype=int)
        # frame_indice = np.linspace(0, 63, self.num_frames, dtype=int)

        video = vframes[frame_indice]  # (T, C, H, W)

        return video
    
    def decord_read(self, path, frame_idx=None):
        decord_vr = self.v_decoder(path)
        total_frames = len(decord_vr)
        # Sampling video frames
        if frame_idx is None:
            start_frame_ind, end_frame_ind = self.temporal_sample(total_frames)
        else:
            start_frame_ind, end_frame_ind = frame_idx.split(':')
            # start_frame_ind, end_frame_ind = int(start_frame_ind), int(end_frame_ind)
            start_frame_ind, end_frame_ind = int(start_frame_ind), int(start_frame_ind) + self.num_frames
        # assert end_frame_ind - start_frame_ind >= self.num_frames
        frame_indice = np.linspace(start_frame_ind, end_frame_ind - 1, self.num_frames, dtype=int)
        # frame_indice = np.linspace(0, 63, self.num_frames, dtype=int)

        video_data = decord_vr.get_batch(frame_indice).asnumpy()
        video_data = torch.from_numpy(video_data)
        video_data = video_data.permute(0, 3, 1, 2)  # (T, H, W, C) -> (T C H W)
        return video_data


    def get_vid_cap_list(self):
        vid_cap_lists = []
        with open(self.video_data, 'r') as f:
            folder_anno = [i.strip().split(',') for i in f.readlines() if len(i.strip()) > 0]
            # print(folder_anno)
        for folder, anno in folder_anno:
            with open(anno, 'r') as f:
                vid_cap_list = json.load(f)
            print(f'Building {anno}...')
            for i in tqdm(range(len(vid_cap_list))):
                path = opj(folder, vid_cap_list[i]['path'])
                if os.path.exists(path.replace('.mp4', '_resize_1080p.mp4')):
                    path = path.replace('.mp4', '_resize_1080p.mp4')
                vid_cap_list[i]['path'] = path
            vid_cap_lists += vid_cap_list
        return vid_cap_lists

    def get_img_cap_list(self):
        use_image_num = self.use_image_num if self.use_image_num != 0 else 1
        img_cap_lists = []
        with open(self.image_data, 'r') as f:
            folder_anno = [i.strip().split(',') for i in f.readlines() if len(i.strip()) > 0]
        for folder, anno in folder_anno:
            with open(anno, 'r') as f:
                img_cap_list = json.load(f)
            print(f'Building {anno}...')
            for i in tqdm(range(len(img_cap_list))):
                img_cap_list[i]['path'] = opj(folder, img_cap_list[i]['path'])
            img_cap_lists += img_cap_list
        img_cap_lists = [img_cap_lists[i: i+use_image_num] for i in range(0, len(img_cap_lists), use_image_num)]
        return img_cap_lists[:-1]  # drop last to avoid error length