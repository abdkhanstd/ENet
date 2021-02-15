# coding: utf-8
import glob

import os
import subprocess
import argparse
from multiprocessing import Pool, current_process
import logging
from multiprocessing_logging import install_mp_handler
def calc_tvl1_flow(vid_item):
    vid_path = vid_item[0]
    vid_name = os.path.basename(vid_path).split('.')[0]
    vid_idx = vid_item[1]
    path=os.path.abspath(os.path.join(os.path.dirname(vid_path), '..'))
    class_name=os.path.basename(path)
    cur_out_dir = os.path.join(OUT_DIR,class_name,vid_name)
    if not os.path.exists(cur_out_dir):
        os.makedirs(cur_out_dir)
    #print('***********',cur_out_dir)
    cur_img_path = os.path.join(cur_out_dir, 'i')
    cur_flow_x_path = os.path.join(cur_out_dir, 'x')
    cur_flow_y_path = os.path.join(cur_out_dir, 'y')

    current = current_process()
    dev_id = (int(current._identity[0]) - 1) % NUM_GPU

    flow_mode_dict = {'tvl1': './denseFlowGpu', 'warp_tvl1': 'extract_warp_gpu'}

    if not os.path.exists(cur_img_path):
        os.makedirs(cur_img_path)
    if not os.path.exists(cur_flow_x_path):
        os.makedirs(cur_flow_x_path)
    if not os.path.exists(cur_flow_y_path):
        os.makedirs(cur_flow_y_path)
    command = '{} -f={} -x={} -y={} -i={} -s=1 -t=1 -d=4 -h=0 -w=0 -b=20'.format(
        flow_mode_dict[FLOW_TYPE], vid_path, cur_flow_x_path+'/x', cur_flow_y_path+'/y', cur_img_path+'/img'
    )
    #
    print(command)
    # read num of files in prev folder
    file_limits=5
    len_img=len(glob.glob(os.path.join(cur_img_path,'*.jpg')))
    len_x=len(glob.glob(os.path.join(cur_flow_x_path,'*.jpg')))
    len_y=len(glob.glob(os.path.join(cur_flow_y_path,'*.jpg')))
    #print('**********',os.path.join(cur_img_path,'*.jpg'))
    if (len_img<=file_limits and len_x <=file_limits and len_y <=file_limits) or (len_img != len_x and len_x !=len_y):
        shell_output = subprocess.getstatusoutput(command)
        #print(command)
        if 'no frame' not in shell_output[1]:
            log_info = '{}/{} {} finished!'.format(vid_idx, VID_NUM, vid_name)
            print (log_info)
            logging.info(log_info)
        else:
            log_info = '>>> Errors occurred when parsing video {}/{} {} <<<'.format(vid_idx, VID_NUM, vid_name)
            print (log_info)
            logging.error(log_info)
    else:
        print('Already ',len_img,' Frames generated for ',vid_name,'\n')
        
        



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Extract optical flows with multi-gpu support.")
    parser.add_argument("vid_txt_path", type=str, help='Input txt files containing video paths.')
    parser.add_argument("out_dir", type=str, help='Destination directory to store flow results.')

    parser.add_argument("--flow_type", type=str, default='tvl1', choices=['tvl1', 'warp_tvl1'], help='Optical flow type. Default: tvl1')
    parser.add_argument("--out_fmt", type=str, default='dir', choices=['dir','zip'], help='Output file format. Default: zip')
    parser.add_argument("--num_gpu", type=int, default=1, help='Number of GPUs. Default: 4')
    parser.add_argument("--step", type=int, default=1, help='Specify the step for frame sampling. Default: 1')
    parser.add_argument("--keep_frames", type=lambda x: (str(x).lower() == 'true'), default=False, help='Whether to save frame data. Default: False')
    parser.add_argument("--width", type=int, default=0, help='Resize image width. Default: 0 (no resize)')
    parser.add_argument("--height", type=int, default=0, help='Resize image height. Default: 0 (no resize)')
    parser.add_argument("--log", type=str, default='./out.log', help='Output log file path. Default: ./out.log')

    args = parser.parse_args()
    f = open(args.vid_txt_path, 'r').readlines()
    vid_paths = [line.strip() for line in f]
    VID_NUM = len(vid_paths)
    NUM_GPU = args.num_gpu
    FLOW_TYPE = args.flow_type
    OUT_DIR = args.out_dir
    OUT_FMT = args.out_fmt
    KEEP_FRAMES = args.keep_frames
    STEP = args.step
    WIDTH = args.width
    HEIGHT = args.height
    LOG = args.log



    #logging.basicConfig(level=logging.DEBUG, format='%(asctime)s %(levelname)s %(message)s',
    #                datefmt='%a, %d %b %Y %H:%M:%S', filename=LOG, filemode='w')
    install_mp_handler()
    pool = Pool(1)
    pool.map(calc_tvl1_flow, zip(vid_paths, range(VID_NUM)))
    pool.close()
