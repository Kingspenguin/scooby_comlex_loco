# import urdfpy

from lxml import etree as ET
import os
import six
# file_obj = "assets/terrains/obstacles/obstacles.urdf"
file_obj = "output.xml"
if isinstance(file_obj, six.string_types):
  if os.path.isfile(file_obj):
    parser = ET.XMLParser(remove_comments=True,
                          remove_blank_text=True)
    tree = ET.parse(file_obj, parser=parser)
    path, _ = os.path.split(file_obj)
  else:
    raise ValueError('{} is not a file'.format(file_obj))
else:
  parser = ET.XMLParser(remove_comments=True, remove_blank_text=True)
  tree = ET.parse(file_obj, parser=parser)
  path, _ = os.path.split(file_obj.name)

node = tree.getroot()
# return URDF._from_xml(node, path)
node = node[0]
print(node)
obj_list = []
for obj in node:
  # print(child.tag)
  if obj.tag == "collision":
    ori, geo = obj[:2]
    # for  in child:
    #     print(grand)
    ori_list = [float(it) for it in ori.get("xyz").strip().split(" ")]
    # print(ori_list)
    geo_list = [float(it) for it in geo[0].get("size").strip().split(" ")]
    # print(geo_list)
    obj_list.append((ori_list, geo_list))
# print(node.keys())

print(obj_list)

import numpy as np
env_spacing = 6
map_scale = 0.02

num_rows = int(2 * env_spacing // map_scale)
num_cols = int(2 * env_spacing // map_scale)

print(num_rows)

env_map = np.zeros((num_rows, num_cols))
for obj in obj_list:
  ori, geo_shape = obj
  x_upper = ori[0] + geo_shape[0] / 2
  x_lower = ori[0] - geo_shape[0] / 2
  x_upper_idx = int((x_upper - (-env_spacing)) // map_scale)
  x_lower_idx = int((x_lower - (-env_spacing)) // map_scale)

  y_upper = ori[1] + geo_shape[1] / 2
  y_lower = ori[1] - geo_shape[1] / 2
  y_upper_idx = int((y_upper - (-env_spacing)) // map_scale)
  y_lower_idx = int((y_lower - (-env_spacing)) // map_scale)

  env_map[x_lower_idx:x_upper_idx,
          y_lower_idx: y_upper_idx] = ori[2] + geo_shape[2] / 2

from matplotlib import pyplot as plt
plt.imshow(env_map)
plt.show()
