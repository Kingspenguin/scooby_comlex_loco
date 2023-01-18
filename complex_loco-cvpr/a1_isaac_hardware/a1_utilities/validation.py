import numpy as np
from A1_inverse_kinematics import *

arr_list = np.load("a1_utilities/cpg/logged_data_0.npy", allow_pickle=True)

end_feet_pos = arr_list[0]["end_feet_pos"]
default_feet_pos = arr_list[0]["default_feet_pos"]

leg_length = torch.tensor([0.08505, 0.2, 0.2])

# this command comes from minghao's validation data.
# under this position command all foot have target joint
# angles [0.0, 0.9, -1.7] in the validation data.
target_pos_isaac = torch.tensor([
    [0.1478, -0.11459, -0.45576],
    [0.1478, 0.11688, -0.45576],
    [-0.2895, -0.11459, -0.45576],
    [-0.2895, 0.11688, -0.45576]
])

# this set of values is the target joint position where
# my IK will output [0.0,0.9,-1.7] for all foot.
# I have made sure that the axis align.
command_to_issue = torch.tensor([
    [0.01319415, -0.08505, -0.26366335],
    [0.01319415, 0.08505, -0.26366335],
    [0.01319415, -0.08505, -0.26366335],
    [0.01319415, 0.08505, -0.26366335]
])

# Therefore, in order to convert minghao's position
# command into the position command recognized by my IK,
# the following offset should be added to minghao's command.
#offset = command_to_issue - target_pos_isaac

offset = torch.tensor([
    [-0.1346, 0.0295, 0.1921],
    [-0.1346, -0.0318, 0.1921],
    [0.3027, 0.0295, 0.1921],
    [0.3027, -0.0318, 0.1921]
])

print("Offset: ", offset)

# I validate the offset computation with the following set of data:
# given target_feet_pos in minghao's coordinate:

max_diff = []

for arr_list_element in arr_list:
  target_feet_pos_isaac = torch.tensor(arr_list_element["targets_feet_pos"])
  correct_joint_angle_output = torch.tensor(arr_list_element["targets_pos"])

  # offset added implicitly as default parameter value defined in function
  my_joint_angle_output = multiple_leg_inverse_kinematics_isaac(
      target_feet_pos_isaac, leg_length)

  # print("Target output:", correct_joint_angle_output)
  # print("My output:", my_joint_angle_output.reshape((4,3)))

  max_diff_val = torch.max(
      torch.abs(correct_joint_angle_output - my_joint_angle_output.reshape((4, 3))))
  max_diff.append(max_diff_val)

max_diff_arr = np.array(max_diff)
print("Max diff mean: ", np.mean(max_diff_arr))
print("Max diff std: ", np.std(max_diff_arr))
print("Max diff max: ", np.max(max_diff_arr))

'''
Output is: 

Max diff mean:  0.044117667
Max diff std:  0.032135192
Max diff max:  0.106487274
'''


'''
# this is the forward kinematics code I used to see what xyz will output [0.0,0.9,-1.7] in my IK code

angle = np.array([0.0,0.9,-1.7])

x = leg_length[1] * np.cos(angle[1]) * np.sin(angle[0]) - leg_length[0] * np.cos(angle[0]) + leg_length[1] * np.cos(angle[1]) * np.cos(angle[2]) * np.sin(angle[0]) - leg_length[2] *  np.sin(angle[1]) * np.sin(angle[2]) * np.sin(angle[0])

y = leg_length[2] * np.cos(angle[0]) * np.sin(angle[1]) * np.sin(angle[2]) - leg_length[1] * np.cos(angle[0]) * np.cos(angle[1]) - leg_length[2] * np.cos(angle[0]) * np.cos(angle[1]) * np.cos(angle[2]) - leg_length[0] * np.sin(angle[0])

z = leg_length[1] * np.sin(angle[1]) + leg_length[2] * np.cos(angle[1]) * np.sin(angle[2]) + leg_length[2] * np.cos(angle[2]) * np.sin(angle[1])
'''
