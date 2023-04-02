from curses import tparm
import os
from tqdm import tqdm
import argparse

def generate(data_dir, val_size, output_path):
    all_data = os.listdir(data_dir)
    # remove extension
    all_data = [data.split('.')[0] for data in all_data]

    val_size = int(val_size * len(all_data))
    train_size = len(all_data) - val_size

    with open(f'{output_path}/train.txt', 'w') as train:
        for data in tqdm(all_data[:train_size]):
            train.write(f'{data}\n')

    with open(f'{output_path}/val.txt', 'w') as val:
        for data in tqdm(all_data[train_size:]):
            val.write(f'{data}\n')

    print('[INFO] Finish...')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='/root/autodl-tmp/RTM3D/kitti_format/data/nuscenes_kitti/image', help='Data path to iterate')
    parser.add_argument('--val_size', type=float, default=0.03, help='Validation size')
    parser.add_argument('--output_path', type=str, default='/root/autodl-tmp/RTM3D/kitti_format/data/nuscenes_kitti', help='Output directory')
    args = parser.parse_args()

    generate(args.data_path, args.val_size, args.output_path)
