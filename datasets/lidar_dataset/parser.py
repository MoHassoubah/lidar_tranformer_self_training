import os
import os.path as osp
import numpy as np
import torch
from torch.utils.data import Dataset
from datasets.lidar_dataset.laserscan import LaserScan, SemLaserScan

from nuscenes.nuscenes import NuScenes

# from nuscenes.utils.data_classes import LidarPointCloud

EXTENSIONS_SCAN = ['.bin']
EXTENSIONS_LABEL = ['.label']
NUSCENES_TRAIN_SIZE = 700


def is_scan(filename):
  return any(filename.endswith(ext) for ext in EXTENSIONS_SCAN)


def is_label(filename):
  return any(filename.endswith(ext) for ext in EXTENSIONS_LABEL)


class SemanticKitti(Dataset):

  def __init__(self, root,    # directory where data is
               sequences,     # sequences for this data (e.g. [1,3,4,6])
               labels,        # label dict: (e.g 10: "car")
               color_map,     # colors dict bgr (e.g 10: [255, 0, 0])
               learning_map,  # classes to learn (0 to N-1 for xentropy)
               learning_map_inv,    # inverse of previous (recover labels)
               sensor,              # sensor to parse scans from
               max_iters=None,# maximum number of training itirations
               max_points=150000,   # max number of points present in dataset
               gt=True,            # send ground truth?
               nuscenes_dataset=False,         # nuscebes dataset?
               pretrain=False,                  # use pretraining flag
               evaluate=False):                  # evaluate flag for noise and model uncertainity calculations  
    # save deats
    self.root = root #os.path.join(root, "sequences") #root
    self.pretrain = pretrain
    self.evaluate = evaluate
    self.drop_percentage = 0.2
    self.sequences = sequences
    self.labels = labels
    self.color_map = color_map
    self.learning_map = learning_map
    self.learning_map_inv = learning_map_inv
    self.sensor = sensor
    self.sensor_img_H = sensor["img_prop"]["height"]
    self.sensor_img_W = sensor["img_prop"]["width"]
    self.sensor_img_means = torch.tensor(sensor["img_means"],
                                         dtype=torch.float)
    self.sensor_img_stds = torch.tensor(sensor["img_stds"],
                                        dtype=torch.float)
    self.sensor_fov_up = sensor["fov_up"]
    self.sensor_fov_down = sensor["fov_down"]
    self.max_points = max_points
    self.gt = gt
    self.nuscenes_dataset = nuscenes_dataset

    # get number of classes (can't be len(self.learning_map) because there
    # are multiple repeated entries, so the number that matters is how many
    # there are for the xentropy)
    self.nclasses = len(self.learning_map_inv)

    # sanity checks

    # make sure directory exists
    if os.path.isdir(self.root):
      print("Sequences folder exists! Using sequences from %s" % self.root)
    else:
      raise ValueError("Sequences folder doesn't exist! Exiting...")

    # make sure labels is a dict
    assert(isinstance(self.labels, dict))

    # make sure color_map is a dict
    assert(isinstance(self.color_map, dict))

    # make sure learning_map is a dict
    assert(isinstance(self.learning_map, dict))

    if(self.nuscenes_dataset == False):
        # make sure sequences is a list
        assert(isinstance(self.sequences, list))

    # placeholder for filenames
    self.files = []

    scan_files_accum = []
    label_files_accum = []
    if(self.nuscenes_dataset == False):
        # fill in with names, checking that all sequences are complete
        for seq in self.sequences:
          # to string
          seq = '{0:02d}'.format(int(seq))

          print("parsing seq {}".format(seq))

          # get paths for each
          # scan_path = os.path.join(self.root, seq, "velodyne")
          # label_path = os.path.join(self.root, seq, "labels")
          
          scan_path = os.path.join(self.root, "volodyne_points", "data_odometry_velodyne", "dataset", "sequences", seq, "velodyne")

          # get files
          scan_files = [os.path.join(dp, f) for dp, dn, fn in os.walk(
              os.path.expanduser(scan_path)) for f in fn if is_scan(f)]
          
          scan_files_accum.extend(scan_files)
          
          # check all scans have labels
          if self.gt:
              label_path = os.path.join(self.root, "data_odometry_labels", "dataset", "sequences", seq, "labels")
              label_files = [os.path.join(dp, f) for dp, dn, fn in os.walk(
                  os.path.expanduser(label_path)) for f in fn if is_label(f)]

              assert(len(scan_files) == len(label_files))

              # extend list
              label_files_accum.extend(label_files)

        # sort for correspondance
        scan_files_accum.sort()
        if self.gt:
            label_files_accum.sort()
        
    else:
        nusc = NuScenes(version='v1.0-trainval', dataroot=self.root, verbose=True)
        for idx in range(self.sequences[0],self.sequences[1]):
            my_scene = nusc.scene[idx]

            first_sample_token = my_scene['first_sample_token']

            my_sample = nusc.get('sample', first_sample_token)
            
            while(True):
                lidar_data = nusc.get('sample_data', my_sample['data']['LIDAR_TOP'])
                
                scan_files_accum.append(osp.join(self.root, lidar_data["filename"]))
                
                if self.gt:
                    lidar_seg = nusc.get('lidarseg', my_sample['data']['LIDAR_TOP']) #returns data as # # print(nusc.lidarseg[index])
                    
                    label_files_accum.append(osp.join(self.root, lidar_seg["filename"]))
                
                if(my_sample['next'] == ''):
                    break
                
                my_sample = nusc.get('sample', my_sample['next'])
    
    for i in range(len(scan_files_accum)):
        
        if self.gt:
            self.files.append({
                "scan": scan_files_accum[i],
                "label": label_files_accum[i]
            })
        else:
            self.files.append({
                "scan": scan_files_accum[i]
            })
        
    

    print("Using {} scans from sequences {}".format(len(self.files),
                                                    self.sequences))
                                                    
    if not max_iters==None:
            self.files = self.files * int(np.ceil(float(max_iters) / len(self.files)))

  def __getitem__(self, index):
    # get item in tensor shape
    scan_file = self.files[index]["scan"]
    if self.gt:
      label_file = self.files[index]["label"]

    # open a semantic laserscan
    if self.gt:
      scan = SemLaserScan(self.color_map,
                          project=True,
                          H=self.sensor_img_H,
                          W=self.sensor_img_W,
                          fov_up=self.sensor_fov_up,
                          fov_down=self.sensor_fov_down,
                          nuscenes_dataset=self.nuscenes_dataset,
                          pretrain= self.pretrain,
                          evaluate=self.evaluate,
                          drop_percentage=self.drop_percentage)
    else:
      scan = LaserScan(project=True,
                       H=self.sensor_img_H,
                       W=self.sensor_img_W,
                       fov_up=self.sensor_fov_up,
                       fov_down=self.sensor_fov_down,
                       nuscenes_dataset=self.nuscenes_dataset,
                       pretrain= self.pretrain,
                       evaluate=self.evaluate,
                       drop_percentage=self.drop_percentage)
                       
    # open and obtain scan
    scan.open_scan(scan_file)
    
    if self.gt:
      scan.open_label(label_file)
      # map unused classes to used classes (also for projection)
      scan.proj_sem_label = self.map(scan.proj_sem_label, self.learning_map)

    
    if not self.pretrain:
        if self.gt:
          # map unused classes to used classes (also for projection)
          scan.sem_label = self.map(scan.sem_label, self.learning_map)
          
    if not (self.pretrain or self.evaluate):
        # make a tensor of the uncompressed data (with the max num points)
        unproj_n_points = scan.points.shape[0]
        unproj_xyz = torch.full((self.max_points, 3), -1.0, dtype=torch.float)
        unproj_xyz[:unproj_n_points] = torch.from_numpy(scan.points)
        unproj_range = torch.full([self.max_points], -1.0, dtype=torch.float)
        unproj_range[:unproj_n_points] = torch.from_numpy(scan.unproj_range)
        unproj_remissions = torch.full([self.max_points], -1.0, dtype=torch.float)
        unproj_remissions[:unproj_n_points] = torch.from_numpy(scan.remissions)
        if self.gt:
          unproj_labels = torch.full([self.max_points], -1.0, dtype=torch.int32)
          unproj_labels[:unproj_n_points] = torch.from_numpy(scan.sem_label)
        else:
          unproj_labels = []
        
        proj_x = torch.full([self.max_points], -1, dtype=torch.long)
        proj_x[:unproj_n_points] = torch.from_numpy(scan.proj_x)
        proj_y = torch.full([self.max_points], -1, dtype=torch.long)
        proj_y[:unproj_n_points] = torch.from_numpy(scan.proj_y)

    # get points and labels
    proj_range = torch.from_numpy(scan.proj_range).clone()
    proj_xyz = torch.from_numpy(scan.proj_xyz).clone()
    proj_remission = torch.from_numpy(scan.proj_remission).clone()
    proj_mask = torch.from_numpy(scan.proj_mask)
    if self.gt:
      proj_labels = torch.from_numpy(scan.proj_sem_label).clone()
      proj_labels = proj_labels * proj_mask
    else:
      proj_labels = []
      
        
    proj = torch.cat([proj_range.unsqueeze(0).clone(),
                      proj_xyz.clone().permute(2, 0, 1),
                      proj_remission.unsqueeze(0).clone()]) # TO create 4 channels image each channel carry sort of data (range,x,y,z,remission)
    proj = (proj - self.sensor_img_means[:, None, None]
            ) / self.sensor_img_stds[:, None, None]
    proj = proj * proj_mask.float()
    
    if self.pretrain or self.evaluate:
        # get points and labels
        reduced_proj_range = torch.from_numpy(scan.reduced_proj_range).clone()
        reduced_proj_xyz = torch.from_numpy(scan.reduced_proj_xyz).clone()
        reduced_proj_remission = torch.from_numpy(scan.reduced_proj_remission).clone()
        reduced_proj_mask = torch.from_numpy(scan.reduced_proj_mask)
          
            
        reduced_proj = torch.cat([reduced_proj_range.unsqueeze(0).clone(),
                          reduced_proj_xyz.clone().permute(2, 0, 1),
                          reduced_proj_remission.unsqueeze(0).clone()]) # TO create 4 channels image each channel carry sort of data (range,x,y,z,remission)
        reduced_proj = (reduced_proj - self.sensor_img_means[:, None, None]
                ) / self.sensor_img_stds[:, None, None]
        reduced_proj = reduced_proj * reduced_proj_mask.float()
        
        
    # get name and sequence
    path_norm = os.path.normpath(scan_file)
    path_split = path_norm.split(os.sep)
    path_seq = path_split[-3]
    path_name = path_split[-1].replace(".bin", ".label")
    # print("path_norm: ", path_norm)
    # print("path_seq", path_seq)
    # print("path_name", path_name)

    # return
    if self.evaluate:
        return proj, proj_mask, proj_labels, reduced_proj, reduced_proj_mask, path_seq, path_name
    elif self.pretrain:
        return proj, proj_mask, reduced_proj, reduced_proj_mask, scan.rot_ang_around_z_axis, path_seq, path_name
    else:
        return proj, proj_mask, proj_labels, unproj_labels, path_seq, path_name, proj_x, proj_y, proj_range, unproj_range, proj_xyz, unproj_xyz, proj_remission, unproj_remissions, unproj_n_points
        
        
  def __len__(self):
    return len(self.files)

  @staticmethod
  def map(label, mapdict):
    # put label from original values to xentropy
    # or vice-versa, depending on dictionary values
    # make learning map a lookup table
    maxkey = 0
    for key, data in mapdict.items():
      if isinstance(data, list):
        nel = len(data)
      else:
        nel = 1
      if key > maxkey:
        maxkey = key
    # +100 hack making lut bigger just in case there are unknown labels
    if nel > 1:
      lut = np.zeros((maxkey + 100, nel), dtype=np.int32)
    else:
      lut = np.zeros((maxkey + 100), dtype=np.int32)
    for key, data in mapdict.items():
      try:
        lut[key] = data
      except IndexError:
        print("Wrong key ", key)
    # do the mapping
    return lut[label]
    
  def set_eval_drop_percentage_dataset(self, drp_percentage):
    self.drop_percentage=drp_percentage


class Parser():
  # standard conv, BN, relu
  def __init__(self,
               root,                            # directory for data
               train_sequences,                 # sequences to train
               valid_sequences,                 # sequences to validate.
               test_sequences,                  # sequences to test (if none, don't get)
               labels,                          # labels in data
               color_map,                       # color for each label
               learning_map,                    # mapping for training labels
               learning_map_inv,                # recover labels from xentropy
               sensor,                          # sensor to use
               max_points,                      # max points in each scan in entire dataset
               batch_size,                      # batch size for train and val
               workers,                         # threads to load data
               max_iters=None,                  # maximum number of training itirations
               gt=True,                         # get gt?
               shuffle_train=True,              # shuffle training set?
               nuscenes_dataset=False,          # nuscebes dataset?
               pretrain=False,                  # use pretraining flag
               evaluate=False):                 # evaluate flag for noise and model uncertainity calculations  
               
    super(Parser, self).__init__()

    # if I am training, get the dataset
    self.root = root
    self.max_iters = max_iters
    self.train_sequences = train_sequences
    self.valid_sequences = valid_sequences
    self.test_sequences = test_sequences
    self.labels = labels
    self.color_map = color_map
    self.learning_map = learning_map
    self.learning_map_inv = learning_map_inv
    self.sensor = sensor
    self.max_points = max_points
    self.batch_size = batch_size
    self.workers = workers
    self.pretrain = pretrain
    self.evaluate = evaluate
    self.gt = gt
    self.shuffle_train = shuffle_train
    self.nuscenes_dataset = nuscenes_dataset

    # number of classes that matters is the one for xentropy
    self.nclasses = len(self.learning_map_inv)

    if self.train_sequences:
        # Data loading code
        self.train_dataset = SemanticKitti(root=self.root,
                                           sequences=self.train_sequences,
                                           labels=self.labels,
                                           color_map=self.color_map,
                                           learning_map=self.learning_map,
                                           learning_map_inv=self.learning_map_inv,
                                           sensor=self.sensor,
                                           max_iters=self.max_iters,
                                           max_points=max_points,
                                           gt=self.gt,
                                           nuscenes_dataset=self.nuscenes_dataset,
                                           pretrain=self.pretrain)

        self.trainloader = torch.utils.data.DataLoader(self.train_dataset,
                                                       batch_size=self.batch_size,
                                                       shuffle=self.shuffle_train,
                                                       num_workers=self.workers,
                                                       pin_memory=True,
                                                       drop_last=True)
        assert len(self.trainloader) > 0
        self.trainiter = iter(self.trainloader)

        
    if self.valid_sequences:
      self.valid_dataset = SemanticKitti(root=self.root,
                                       sequences=self.valid_sequences,
                                       labels=self.labels,
                                       color_map=self.color_map,
                                       learning_map=self.learning_map,
                                       learning_map_inv=self.learning_map_inv,
                                       sensor=self.sensor,
                                       max_points=max_points,
                                       gt=self.gt,
                                       nuscenes_dataset=self.nuscenes_dataset,
                                       pretrain=self.pretrain,
                                       evaluate=self.evaluate)

      self.validloader = torch.utils.data.DataLoader(self.valid_dataset,
                                                   batch_size=self.batch_size,
                                                   shuffle=False,
                                                   num_workers=self.workers,
                                                   pin_memory=True,
                                                   drop_last=True)
      assert len(self.validloader) > 0
      self.validiter = iter(self.validloader)

    if self.test_sequences:
      self.test_dataset = SemanticKitti(root=self.root,
                                        sequences=self.test_sequences,
                                        labels=self.labels,
                                        color_map=self.color_map,
                                        learning_map=self.learning_map,
                                        learning_map_inv=self.learning_map_inv,
                                        sensor=self.sensor,
                                        max_points=max_points,
                                        gt=False,
                                        nuscenes_dataset=self.nuscenes_dataset)

      self.testloader = torch.utils.data.DataLoader(self.test_dataset,
                                                    batch_size=self.batch_size,
                                                    shuffle=False,
                                                    num_workers=self.workers,
                                                    pin_memory=True,
                                                    drop_last=True)
      assert len(self.testloader) > 0
      self.testiter = iter(self.testloader)

  def set_eval_drop_percentage(self, drp_percentage):
    if self.valid_sequences:
        self.valid_dataset.set_eval_drop_percentage_dataset(drp_percentage)

  def get_train_batch(self):
    scans = self.trainiter.next()
    return scans

  def get_train_set(self):
    return self.trainloader

  def get_valid_batch(self):
    scans = self.validiter.next()
    return scans

  def get_valid_set(self):
    return self.validloader

  def get_test_batch(self):
    scans = self.testiter.next()
    return scans

  def get_test_set(self):
    return self.testloader

  def get_train_size(self):
    return len(self.trainloader)

  def get_valid_size(self):
    return len(self.validloader)

  def get_test_size(self):
    return len(self.testloader)

  def get_n_classes(self):
    return self.nclasses

  def get_original_class_string(self, idx):
    return self.labels[idx]

  def get_xentropy_class_string(self, idx):
    return self.labels[self.learning_map_inv[idx]]

  def to_original(self, label):
    # put label in original values
    return SemanticKitti.map(label, self.learning_map_inv)

  def to_xentropy(self, label):
    # put label in xentropy values
    return SemanticKitti.map(label, self.learning_map)

  def to_color(self, label):
    # put label in original values
    label = SemanticKitti.map(label, self.learning_map_inv)
    # put label in color
    return SemanticKitti.map(label, self.color_map)
