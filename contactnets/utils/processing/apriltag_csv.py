import pdb  # noqa
import sys

import numpy as np
import rosbag
import yaml

# static parameters
CUBE_TOPIC_STRING = '/tagslam/odom/body_cube'
BOARD_TOPIC_STRING = '/tagslam/odom/body_surface'

if len(sys.argv) < 3:
    print("usage: python[2] apriltag_csv.py [ROSBAG] [CSV_OUT]")
    sys.exit()

inbag = sys.argv[1]
outcsv = sys.argv[2]

# open rosbag
bag = rosbag.Bag(inbag)

# get summary info from rosbag as a dictionary
info = yaml.load(bag._get_yaml_info())

# extract metadata from cube and board topics
cube_topic = [topic for topic in info['topics'] if topic['topic'] == CUBE_TOPIC_STRING][0]
board_topic = [topic for topic in info['topics'] if topic['topic'] == BOARD_TOPIC_STRING][0]

# ensure there are an equal number of cube and board odom messages
num_msg = cube_topic['messages']
if not num_msg == board_topic['messages']:
    raise Exception('Missing odom messages for board and/or cube!')


def extract_times(messages):
    t_ros = np.zeros(len(messages))
    for i, data in enumerate(list(messages)):
        (_, msg, _) = data
        tstamp = msg.header.stamp
        t_ros[i] = tstamp.secs + tstamp.nsecs * 1e-9

    return t_ros


def extract_poses(messages):
    poses = np.zeros((7, len(messages)))
    for i, data in enumerate(messages):
        (_, msg, _) = data
        pose = msg.pose.pose
        pose_pos = np.asarray([pose.position.x, pose.position.y, pose.position.z])
        pose_quat = np.asarray([pose.orientation.x, pose.orientation.y,
                                pose.orientation.z, pose.orientation.w])
        poses[:4, i] = pose_quat
        poses[4:7, i] = pose_pos

    return poses


t_ros = extract_times(list(bag.read_messages(topics=[CUBE_TOPIC_STRING])))
cube_ros = extract_poses(list(bag.read_messages(topics=[CUBE_TOPIC_STRING])))
board_ros = extract_poses(list(bag.read_messages(topics=[BOARD_TOPIC_STRING])))

bag.close()

# assemble CSV data and save
data = np.concatenate((np.expand_dims(t_ros, axis=0), cube_ros, board_ros), axis=0)
data = data.T
np.savetxt(outcsv, data, delimiter=',')
