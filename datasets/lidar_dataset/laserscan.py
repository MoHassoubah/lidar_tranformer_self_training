#!/usr/bin/env python3
# This file is covered by the LICENSE file in the root of this project.
import numpy as np
from numpy.random import default_rng


class LaserScan:
  """Class that contains LaserScan with x,y,z,r"""
  EXTENSIONS_SCAN = ['.bin']

  def __init__(self, project=False, H=64, W=1024, fov_up=3.0, fov_down=-25.0, nuscenes_dataset=False, pretrain=False, evaluate=False,drop_percentage=None):
    self.project = project
    self.proj_H = H
    self.proj_W = W
    self.proj_fov_up = fov_up
    self.proj_fov_down = fov_down
    self.nuscenes_dataset = nuscenes_dataset
    self.pretrain = pretrain
    self.evaluate = evaluate
    self.drop_percentage = drop_percentage
    self.reset()

  def reset(self):
    """ Reset scan members. """
    self.points = np.zeros((0, 3), dtype=np.float32)        # [m, 3]: x, y, z
    self.remissions = np.zeros((0, 1), dtype=np.float32)    # [m ,1]: remission

    # projected range image - [H,W] range (-1 is no data)
    self.proj_range = np.full((self.proj_H, self.proj_W), -1,
                              dtype=np.float32)
                              
    

    # projected point cloud xyz - [H,W,3] xyz coord (-1 is no data)
    self.proj_xyz = np.full((self.proj_H, self.proj_W, 3), -1,
                            dtype=np.float32)
                            

    # projected remission - [H,W] intensity (-1 is no data)
    self.proj_remission = np.full((self.proj_H, self.proj_W), -1,
                                  dtype=np.float32)
                                                                    

    # projected index (for each pixel, what I am in the pointcloud)
    # [H,W] index (-1 is no data)
    self.proj_idx = np.full((self.proj_H, self.proj_W), -1,
                            dtype=np.int32)

    # mask containing for each pixel, if it contains a point or not
    self.proj_mask = np.zeros((self.proj_H, self.proj_W),
                              dtype=np.int32)       # [H,W] mask
    
    
    ######################################
    if self.pretrain or self.evaluate:
        self.indicies_remained_aft_drop = np.zeros((0, 1), dtype=np.int32)    # [m ,1]: indicies_remained_aft_drop
        
        self.reduced_proj_range = np.full((self.proj_H, self.proj_W), -1,
                                  dtype=np.float32)
                                  
        self.reduced_proj_xyz = np.full((self.proj_H, self.proj_W, 3), -1,
                                dtype=np.float32)
                                  
        self.reduced_proj_remission = np.full((self.proj_H, self.proj_W), -1,
                                      dtype=np.float32)
        self.reduced_proj_mask = np.zeros((self.proj_H, self.proj_W),
                                  dtype=np.int32)       # [H,W] mask
        self.reduced_proj_idx = np.full((self.proj_H, self.proj_W), -1,
                                dtype=np.int32)
                                
    #######################################

    ### all the next variables changes depending if there is dropping or not

    if not (self.pretrain or self.evaluate):
        # unprojected range (list of depths for each point)
        self.unproj_range = np.zeros((0, 1), dtype=np.float32)
        # for each point, where it is in the range image
        self.proj_x = np.zeros((0, 1), dtype=np.int32)        # [m, 1]: x
        self.proj_y = np.zeros((0, 1), dtype=np.int32)        # [m, 1]: y

  def size(self):
    """ Return the size of the point cloud. """
    return self.points.shape[0]

  def __len__(self):
    return self.size()

  def open_scan(self, filename):
    """ Open raw scan and fill in attributes
    """
    # reset just in case there was an open structure
    self.reset()

    # check filename is string
    if not isinstance(filename, str):
      raise TypeError("Filename should be string type, "
                      "but was {type}".format(type=str(type(filename))))

    # check extension is a laserscan
    if not any(filename.endswith(ext) for ext in self.EXTENSIONS_SCAN):
      raise RuntimeError("Filename extension is not valid scan file.")

    # if all goes well, open pointcloud
    scan = np.fromfile(filename, dtype=np.float32)
    if(self.nuscenes_dataset == False):
        scan = scan.reshape((-1, 4))
    else:
        scan = scan.reshape((-1, 5))[:, :4]

    # put in attribute
    points = scan[:, 0:3]    # get xyz
    remissions = scan[:, 3]  # get remission
    self.set_points(points, remissions)

  def set_points(self, points, remissions=None):
    """ Set scan attributes (instead of opening from file)
    """
    # reset just in case there was an open structure
    self.reset()

    # check scan makes sense
    if not isinstance(points, np.ndarray):
      raise TypeError("Scan should be numpy array")

    # check remission makes sense
    if remissions is not None and not isinstance(remissions, np.ndarray):
      raise TypeError("Remissions should be numpy array")
    
    if(self.nuscenes_dataset == False):
    # put in attribute
        self.points = points    # get xyz
    else:
        #rotate the nuscenes corrdinates to the kitti coordinates by 90 degrees
        #in the kitti coordinates y-nuscenes is the x_kitti
        #and -ve x-nuscnes is the y 
        # x_kitti = np.copy(points[:,1])
        # y_kitti = -1 * points[:,0]
        self.points = np.stack((points[:,1], -1*points[:,0], points[:,2]), axis=1)
        
    if remissions is not None:
      self.remissions = remissions  # get remission
    else:
      self.remissions = np.zeros((points.shape[0]), dtype=np.float32)
      
    if self.pretrain or self.evaluate:
        aug_points = self.do_cloud_augmentatiion()
        self.drop_x_percent_frm_pntcloud()

    # if projection is wanted, then do it and fill in the structure
    if self.project:
        if self.pretrain or self.evaluate:    
            proj_y, proj_x, depth, ret_points, ret_remission, indices = self.do_range_projection(param_aug_points = aug_points)
        else:
            proj_y, proj_x, depth, ret_points, ret_remission, indices = self.do_range_projection()
        
        self.proj_range[proj_y, proj_x] = depth
        self.proj_xyz[proj_y, proj_x] = ret_points
        self.proj_remission[proj_y, proj_x] = ret_remission
        self.proj_idx[proj_y, proj_x] = indices
        self.proj_mask = (self.proj_idx > 0).astype(np.int32)
        
        # do the projection one more time for the reduced PointCloud
        if self.pretrain or self.evaluate:
            proj_y, proj_x, depth, ret_points, ret_remission, indices = self.do_range_projection(True,param_aug_points = aug_points)
            
            self.reduced_proj_range[proj_y, proj_x] = depth
            self.reduced_proj_xyz[proj_y, proj_x] = ret_points
            self.reduced_proj_remission[proj_y, proj_x] = ret_remission
            self.reduced_proj_idx[proj_y, proj_x] = indices
            self.reduced_proj_mask = (self.reduced_proj_idx > 0).astype(np.int32)
      
   
  def do_translation_augmentation(self):
    std = 0.2**0.5
    delta_x = np.random.normal(0,std)
    aug_points = np.zeros_like(self.points)
    aug_points[:, 0] = self.points[:, 0] + delta_x
    delta_y = np.random.normal(0,std)
    aug_points[:, 1] = self.points[:, 1] + delta_y
    delta_z = np.random.normal(0,std)
    aug_points[:, 2] = self.points[:, 2] + delta_z
    return aug_points
    
  def do_random_scaling(self, aug_points):
    t= 0.05
    s = np.random.uniform(1.0-t, 1.0+t)
    
    aug_points = aug_points * s
    return aug_points
    
  def do_random_flip_x_y_axis(self, aug_points):
    do_flipping = np.random.choice([0, 1])
    if do_flipping:
        x_is_0_y_is_1 = np.random.choice([0, 1])
        if x_is_0_y_is_1:#flip around y axis
            aug_points[:, 0] = aug_points[:, 0] * (-1)
        else:#flip around x axis
            aug_points[:, 1] = aug_points[:, 1] * (-1)
            
    return aug_points
            
  def do_random_rotation(self, aug_points):
    self.rot_ang_around_z_axis = np.random.randint(0, 4) 
    
    if self.rot_ang_around_z_axis:
        theta = np.radians(self.rot_ang_around_z_axis*90)
        c, s = np.cos(theta), np.sin(theta)
        R = np.array(((c,-s,0), (s,c,0),(0,0, 1)))
                
        aug_points = np.matmul(aug_points,R)   
    return aug_points
    
  def do_cloud_augmentatiion(self):
    aug_points = self.do_translation_augmentation()
    aug_points = self.do_random_scaling(aug_points)
    if not self.evaluate:
        aug_points = self.do_random_flip_x_y_axis(aug_points)
        aug_points = self.do_random_rotation(aug_points)
    return aug_points
    
  def drop_x_percent_frm_pntcloud(self):
    if self.evaluate:
        dropping_ratio = self.drop_percentage
    else:
        dropping_ratio = np.random.uniform(0.5, 0.76)
    num_pnts = self.points.shape[0]
    num_dropped_pnts = int(num_pnts*dropping_ratio)
    rng = default_rng()
    indcies_dropped_pnt = rng.choice(num_pnts, size=num_dropped_pnts, replace=False)
    self.indicies_remained_aft_drop =  ~np.isin(np.arange(num_pnts), indcies_dropped_pnt) 
    
    
  def do_range_projection(self, drop_pnts=False,param_aug_points=None):
    """ Project a pointcloud into a spherical projection image.projection.
        Function takes no arguments because it can be also called externally
        if the value of the constructor was not set (in case you change your
        mind about wanting the projection)
    """
    if self.pretrain or self.evaluate:
        if drop_pnts:
            pointcloud = param_aug_points[self.indicies_remained_aft_drop]
            pnt_remission = self.remissions[self.indicies_remained_aft_drop]
        else:
            pointcloud = param_aug_points
            pnt_remission = self.remissions
    else:
        pointcloud = self.points
        pnt_remission = self.remissions
        
    # laser parameters
    fov_up = self.proj_fov_up / 180.0 * np.pi      # field of view up in rad
    fov_down = self.proj_fov_down / 180.0 * np.pi  # field of view down in rad
    
        
    fov = abs(fov_down) + abs(fov_up)  # get field of view total in rad

    # get depth of all points
    depth = np.linalg.norm(pointcloud, 2, axis=1)

    # get scan components
    scan_x = pointcloud[:, 0]
    scan_y = pointcloud[:, 1]
    scan_z = pointcloud[:, 2]

    # get angles of all points
    yaw = -np.arctan2(scan_y, scan_x)
    pitch = np.arcsin(scan_z / depth)

    # get projections in image coords
    proj_x = 0.5 * (yaw / np.pi + 1.0)          # in [0.0, 1.0]
    proj_y = 1.0 - (pitch + abs(fov_down)) / fov        # in [0.0, 1.0]

    # scale to image size using angular resolution
    proj_x *= self.proj_W                              # in [0.0, W]
    proj_y *= self.proj_H                              # in [0.0, H]
    
    if not (self.pretrain or self.evaluate):
        # copy of depth in original order
        self.unproj_range = np.copy(depth)

    #proj_x/y are the indicies of the available points in the 2d domain
    # round and clamp for use as index
    proj_x = np.floor(proj_x)
    proj_x = np.minimum(self.proj_W - 1, proj_x)
    proj_x = np.maximum(0, proj_x).astype(np.int32)   # in [0,W-1]
    if not (self.pretrain or self.evaluate):
        self.proj_x = np.copy(proj_x)  # store a copy in orig order

    proj_y = np.floor(proj_y)
    proj_y = np.minimum(self.proj_H - 1, proj_y)
    proj_y = np.maximum(0, proj_y).astype(np.int32)   # in [0,H-1]
    if not (self.pretrain or self.evaluate):
        self.proj_y = np.copy(proj_y)  # store a copy in original order

    # order in decreasing depth
    indices = np.arange(depth.shape[0])
    order = np.argsort(depth)[::-1]
    depth = depth[order]
    indices = indices[order]
    points = pointcloud[order]
    remission = pnt_remission[order]
    proj_y = proj_y[order]
    proj_x = proj_x[order]

    
    return proj_y, proj_x, depth, points, remission, indices


class SemLaserScan(LaserScan):
  """Class that contains LaserScan with x,y,z,r,sem_label,sem_color_label,inst_label,inst_color_label"""
  EXTENSIONS_LABEL = ['.label']

  def __init__(self,  sem_color_dict=None, project=False, H=64, W=1024, fov_up=3.0, fov_down=-25.0, nuscenes_dataset=False, max_classes=300, pretrain=False, evaluate=False, drop_percentage=None):
    super(SemLaserScan, self).__init__(project, H, W, fov_up, fov_down, nuscenes_dataset,pretrain,evaluate,drop_percentage)
    self.reset()

    # make semantic colors
    if sem_color_dict:
      # if I have a dict, make it
      max_sem_key = 0
      for key, data in sem_color_dict.items():
        if key + 1 > max_sem_key:
          max_sem_key = key + 1
      self.sem_color_lut = np.zeros((max_sem_key + 100, 3), dtype=np.float32)
      for key, value in sem_color_dict.items():
        self.sem_color_lut[key] = np.array(value, np.float32) / 255.0
    else:
      # otherwise make random
      max_sem_key = max_classes
      self.sem_color_lut = np.random.uniform(low=0.0,
                                             high=1.0,
                                             size=(max_sem_key, 3))
      # force zero to a gray-ish color
      self.sem_color_lut[0] = np.full((3), 0.1)

    # make instance colors
    max_inst_id = 100000
    self.inst_color_lut = np.random.uniform(low=0.0,
                                            high=1.0,
                                            size=(max_inst_id, 3))
    # force zero to a gray-ish color
    self.inst_color_lut[0] = np.full((3), 0.1)

  def reset(self):
    """ Reset scan members. """
    super(SemLaserScan, self).reset()

    # semantic labels
    self.sem_label = np.zeros((0, 1), dtype=np.int32)          # [m, 1]: label
    self.sem_label_color = np.zeros((0, 3), dtype=np.float32)  # [m ,3]: color

    # instance labels
    self.inst_label = np.zeros((0, 1), dtype=np.int32)          # [m, 1]: label
    self.inst_label_color = np.zeros((0, 3), dtype=np.float32)  # [m ,3]: color

    # projection color with semantic labels
    self.proj_sem_label = np.zeros((self.proj_H, self.proj_W),
                                   dtype=np.int32)              # [H,W]  label
    self.proj_sem_color = np.zeros((self.proj_H, self.proj_W, 3),
                                   dtype=np.float)              # [H,W,3] color

    # projection color with instance labels
    self.proj_inst_label = np.zeros((self.proj_H, self.proj_W),
                                    dtype=np.int32)              # [H,W]  label
    self.proj_inst_color = np.zeros((self.proj_H, self.proj_W, 3),
                                    dtype=np.float)              # [H,W,3] color

  def open_label(self, filename):
    """ Open raw scan and fill in attributes
    """
    # check filename is string
    if not isinstance(filename, str):
      raise TypeError("Filename should be string type, "
                      "but was {type}".format(type=str(type(filename))))

    if(self.nuscenes_dataset == False):
        # check extension is a laserscan
        if not any(filename.endswith(ext) for ext in self.EXTENSIONS_LABEL):
          raise RuntimeError("Filename extension is not valid label file.")

    # if all goes well, open label
    # if(self.nuscenes_dataset == False):
        label = np.fromfile(filename, dtype=np.int32)
        label = label.reshape((-1))
    else:
        label = np.fromfile(filename, dtype=np.uint8) #resotring from the bin file with data type int32 instead of uint8 of course would decrease the number of points as here 4 points would be considered one point
        
    # set it
    self.set_label(label)

  def set_label(self, label):
    """ Set points for label not from file but from np
    """
    # check label makes sense
    if not isinstance(label, np.ndarray):
      raise TypeError("Label should be numpy array")

    # only fill in attribute if the right size
    if label.shape[0] == self.points.shape[0]:
      self.sem_label = label & 0xFFFF  # semantic label in lower half
      self.inst_label = label >> 16    # instance id in upper half
    else:
      print("Points shape: ", self.points.shape)
      print("Label shape: ", label.shape)
      raise ValueError("Scan and Label don't contain same number of points")

    # sanity check
    assert((self.sem_label + (self.inst_label << 16) == label).all())

    if self.project:
      self.do_label_projection()

  def colorize(self):
    """ Colorize pointcloud with the color of each semantic label
    """
    self.sem_label_color = self.sem_color_lut[self.sem_label]
        
    self.sem_label_color = self.sem_label_color.reshape((-1, 3))

    self.inst_label_color = self.inst_color_lut[self.inst_label]
        
    self.inst_label_color = self.inst_label_color.reshape((-1, 3))
    
    
  def do_label_projection(self):
    
    
    # only map colors to labels that exist
    mask = self.proj_idx >= 0
    loc_proj_idx = self.proj_idx
    pnts_sem_label = self.sem_label
    pnts_inst_label = self.inst_label
        
    
    #example of indexing
    # # # x=np.random.randn(10)
    # # # >>> x
    # # # array([-0.14319588,  0.66993997, -0.07680863, -0.47198734,  0.42516838,
        # # # 2.19093034,  1.23173677, -1.45327592,  0.61254348, -0.35860868])
    # # # >>> y = np.array([[0,1],[2,3]])
    # # # >>> x[y]
    # # # array([[-0.14319588,  0.66993997],
       # # # [-0.07680863, -0.47198734]])
       

    # proj_idx carries the original ids in the points array but in the 2d matrix, and mask defines the location in this 2d matrix
    # sem_label carries the labels and sem_color_lut carries the color of each label (which should not be affected by point dropping)
    # semantics
    self.proj_sem_label[mask] = pnts_sem_label[loc_proj_idx[mask]]#each cell in the 2d matrix carries an index
    self.proj_sem_color[mask] = self.sem_color_lut[pnts_sem_label[loc_proj_idx[mask]]]

    # instances
    self.proj_inst_label[mask] = pnts_inst_label[loc_proj_idx[mask]]
    self.proj_inst_color[mask] = self.inst_color_lut[pnts_inst_label[loc_proj_idx[mask]]]
