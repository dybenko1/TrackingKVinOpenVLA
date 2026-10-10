"""Oracle instance segmentation aligned to OpenVLA policy pixels and 16x16 tokens."""
import numpy as np
BACKGROUND_HINTS=("table","floor","wall","background","arena","skybox","ground")
ROBOT_HINTS=("panda","rethinkmount","gripper")
def rendered_label_mapping(instances_to_ids):
 return {i+1:name for i,name in enumerate(instances_to_ids.keys())}
def is_foreground(name):
 n=name.lower()
 return any(x in n for x in ROBOT_HINTS) or not any(x in n for x in BACKGROUND_HINTS)
def foreground_mask(seg, mapping):
 fg=np.zeros(seg.shape,dtype=bool)
 for label,name in mapping.items():
  if is_foreground(name): fg|=seg==label
 return fg
def policy_mask(mask, center_crop=True):
 import tensorflow as tf
 x=tf.convert_to_tensor(mask[...,None],dtype=tf.float32);x=tf.image.resize(x,(224,224),method="nearest");
 if center_crop:
  e=tf.sqrt(tf.constant(.9));box=tf.reshape(tf.stack([(1-e)/2,(1-e)/2,(1+e)/2,(1+e)/2]),(1,4));x=tf.image.crop_and_resize(x[None],box,[0],(224,224),method="nearest")[0]
 return x[...,0].numpy()>.5
def token_mask(mask):
 if mask.shape!=(224,224): raise ValueError(mask.shape)
 return mask.reshape(16,14,16,14).any(axis=(1,3)).reshape(-1)
