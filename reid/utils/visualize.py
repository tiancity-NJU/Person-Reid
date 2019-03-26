import numpy as np
from PIL import Image
import cv2
import os
from os.path import dirname as ospdn



def add_border(im, border_width, value):
  """Add color border around an image. The resulting image size is not changed.
  Args:
    im: numpy array with shape [3, im_h, im_w]
    border_width: scalar, measured in pixel
    value: scalar, or numpy array with shape [3]; the color of the border
  Returns:
    im: numpy array with shape [3, im_h, im_w]
  """
  assert (im.ndim == 3) and (im.shape[0] == 3)
  im = np.copy(im)

  if isinstance(value, np.ndarray):
    # reshape to [3, 1, 1]
    value = value.flatten()[:, np.newaxis, np.newaxis]
  im[:, :border_width, :] = value
  im[:, -border_width:, :] = value
  im[:, :, :border_width] = value
  im[:, :, -border_width:] = value

  return im

def make_im_grid(ims,names, n_rows, n_cols, space, pad_val):
  """Make a grid of images with space in between.
  Args:
    ims: a list of [3, im_h, im_w] images
    names: a list of like [C01_0000042,....]    img info
    n_rows: num of rows
    n_cols: num of columns
    space: the num of pixels between two images
    pad_val: scalar, or numpy array with shape [3]; the color of the space
  Returns:
    ret_im: a numpy array with shape [3, H, W]
  """
  assert (ims[0].ndim == 3) and (ims[0].shape[0] == 3)
  assert len(ims) <= n_rows * n_cols
  h, w = ims[0].shape[1:]
  H = h * n_rows + space * (n_rows - 1) + 300
  W = w * n_cols + space * (n_cols - 1)
  if isinstance(pad_val, np.ndarray):
    # reshape to [3, 1, 1]
    pad_val = pad_val.flatten()[:, np.newaxis, np.newaxis]
  ret_im = (np.ones([3, H, W]) * pad_val).astype(ims[0].dtype)
  title = (np.ones((300,W,3))* pad_val).astype(ims[0].dtype)

  #tmp = np.asarray(Image.open('/home/ztc/zhang/PycharmProjects/open-reid/data/test/images/00000000_01_00031760.jpg'))
  #print(title.shape,tmp.shape)
  for n, im in enumerate(ims):

    #im = im.transpose(1,2,0)

    #im = im.transpose(2,0,1)
    r = n // n_cols
    c = n % n_cols
    h1 = r * (h + space)
    h2 = r * (h + space) + h
    w1 = c * (w + space)
    w2 = c * (w + space) + w
    ret_im[:, h1:h2, w1:w2] = im
    title = cv2.putText(title, names[n], (w1, 130), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 255), 1)  # zuoyou   gaodi

  title = title.transpose(2,0,1)
  ret_im[:,-300:,:] = title

  return ret_im


def get_rank_list(dist_vec, q_id, q_cam, q_frame, g_ids, g_cams, g_frames, rank_list_size, temporal = True):
  """Get the ranking list of a query image
  Args:
    dist_vec: a numpy array with shape [num_gallery_images], the distance
      between the query image and all gallery images
    q_id: a scalar, query id
    q_cam: a scalar, query camera
    g_ids: a numpy array with shape [num_gallery_images], gallery ids
    g_cams: a numpy array with shape [num_gallery_images], gallery cameras
    rank_list_size: a scalar, the number of images to show in a rank list
    q_frame: query frame number
    g_frames: array of gallery frame number
    temporal: whether consider spatio-temporal information

  Returns:
    rank_list: a list, the indices of gallery images to show
    same_id: a list, len(same_id) = rank_list, whether each ranked image is
      with same id as query
  """
  sort_inds = np.argsort(dist_vec)
  rank_list = []
  same_id = []
  i = 0

  for ind, g_id, g_cam, g_frame in zip(sort_inds, g_ids[sort_inds], g_cams[sort_inds], g_frames[sort_inds]):
    # Skip gallery images with same id and same camera as query
    #if (q_id == g_id) and (q_cam == g_cam):

    if not temporal:
      if q_cam == g_cam:
        continue
    else:
      if q_cam == g_cam or not near(q_frame,g_frame):    #   change the code. we think image pair can not appear in same cam
        continue

    same_id.append(q_id == g_id)
    rank_list.append(ind)
    i += 1
    if i >= rank_list_size:
      break
  return rank_list, same_id


def read_im(im_path):
  # shape [H, W, 3]
  im = np.asarray(Image.open(im_path))
  # Resize to (im_h, im_w) = (128, 64)
  resize_h_w = (1024, 512)
  if (im.shape[0], im.shape[1]) != resize_h_w:
    im = cv2.resize(im, resize_h_w[::-1], interpolation=cv2.INTER_LINEAR)
  # shape [3, H, W]
  im = im.transpose(2, 0, 1)
  return im


def save_im(im, im_name,save_path):
  """im: shape [3, H, W]"""
  im = im.transpose(1, 2, 0)

  if not os.path.exists(save_path):
    os.makedirs(save_path)
  Image.fromarray(im).save(os.path.join(save_path,'vis_'+im_name))


def save_rank_list_to_im(rank_list, same_id, q_im_path, g_im_paths, save_path,ignore_id = False):
  """Save a query and its rank list as an image.
  Args:
    rank_list: a list, the indices of gallery images to show
    same_id: a list, len(same_id) = rank_list, whether each ranked image is
      with same id as query
    q_im_path: query image path
    g_im_paths: ALL gallery image paths
    save_path: path to save the query and its rank list as an image
    ignore: wheather add label info on images
  """
  ims = [read_im(q_im_path)]
  names = [parse_visual_img_name(q_im_path)]

  # print(q_im_path,end=' ')
  # for i,ind in enumerate(rank_list):
  #   print(g_im_paths[ind].split('/')[-1],end=' ')
  #   i+=1
  #   if i==10:
  #     print()
  #     print()
  #     break

  for ind, sid in zip(rank_list, same_id):
    im = read_im(g_im_paths[ind])
    # Add green boundary to true positive, red to false positive
    color = np.array([0, 255, 0]) if sid else np.array([255, 0, 0])
    if ignore_id:
      color = np.array([0, 0, 0])

    im = add_border(im, 3, color)
    ims.append(im)
    names.append(parse_visual_img_name(g_im_paths[ind]))

  im = make_im_grid(ims,names, 1, len(rank_list) + 1, 8, 255)
  save_im(im, q_im_path.split('/')[-1],save_path)



def get_frame(img):
  return int(img.split('_')[-1][:-4])


def parse_visual_img_name(path):
  """

  :param path:   xxx/xxx/00000000_01_00000042.jpg
  :return: C01_00000042
  """
  img = path.split('/')[-1][:-4]
  info = img.split('_')
  return 'C'+info[1]+'_'+str(int(info[2]))


def near(q_frame,g_frame,threshold = 1000):
  if abs(q_frame-g_frame)<threshold:
    return True

  return False
