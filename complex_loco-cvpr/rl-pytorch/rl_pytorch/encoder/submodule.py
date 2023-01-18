from cv2 import reprojectImageTo3D
import torch
import torch.nn as nn
import torch.nn.functional as F


def euler2mat(angle):
  """Convert euler angles to rotation matrix.
   Reference: https://github.com/pulkitag/pycaffe-utils/blob/master/rot_utils.py#L174
  Args:
      angle: rotation angle along 3 axis (a, b, y) in radians -- size = [B, 3]
  Returns:
      Rotation matrix corresponding to the euler angles -- size = [B, 3, 3]
  """
  B = angle.size(0)
  # import pdb; pdb.set_trace()

  x, y, z = angle[:, 1], angle[:, 0], angle[:, 2]

  cosz = torch.cos(z)
  sinz = torch.sin(z)

  zeros = z.detach() * 0
  ones = zeros.detach() + 1
  zmat = torch.stack([cosz, -sinz, zeros,
                      sinz, cosz, zeros,
                      zeros, zeros, ones], dim=1).reshape(B, 3, 3)

  cosy = torch.cos(y)
  siny = torch.sin(y)

  ymat = torch.stack([cosy, zeros, siny,
                      zeros, ones, zeros,
                      -siny, zeros, cosy], dim=1).reshape(B, 3, 3)

  cosx = torch.cos(x)
  sinx = torch.sin(x)

  xmat = torch.stack([ones, zeros, zeros,
                      zeros, cosx, -sinx,
                      zeros, sinx, cosx], dim=1).reshape(B, 3, 3)

  rotMat = xmat @ ymat @ zmat

  v_trans = angle[:, 3:]

  # F.affine_grid takes 3x4
  rotMat = torch.cat([rotMat, v_trans.view([B, 3, 1]).to(rotMat.device)], 2)

  return rotMat


def padmat(mat):
  mat_shape = mat.shape
  padding_vec = torch.zeros(mat_shape[:-2] + (1, 4), device=mat.device)
  padding_vec[..., -1] = 1
  transMat = torch.cat([
      mat, padding_vec
  ], dim=-2)
  return transMat


def stn(x, theta, padding_mode='zeros'):
  grid = F.affine_grid(theta, x.size())
  img = F.grid_sample(x, grid, padding_mode=padding_mode)
  return img


def conv(in_planes, out_planes, kernel_size=3):
  return nn.Sequential(
      nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size,
                padding=(kernel_size - 1) // 2, stride=2),
      nn.ReLU(inplace=True)
  )
