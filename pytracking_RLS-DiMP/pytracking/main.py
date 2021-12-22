"""
将跟踪结果转换为trackingnet需要的格式：
488.000,43.000,763.000,569.000 //保留小数点后三位
410.013,-176.648,765.221,574.119 //注意换行符是\n，不是\r\n
322.992,-288.570,767.369,578.141 //用逗号分隔，没有空格
261.732,-291.000,770.155,583.000
216.403,-293.505,772.937,588.010
//末尾有空行
"""
import os
import glob


def process_per_txt(file_path, save_dir):
    print(file_path)
    with open(file_path, 'r') as f:
        file_content = f.read().splitlines()
    new_file_content = []
    '''处理每行数据'''
    for line in file_content:
        int_list = list(map(int, line.split('\t')))
        float_list = ['{:.3f}'.format(x) for x in int_list]
        new_file_content.append(','.join(float_list))
    '''写入新文件'''
    new_file_name = file_path.split('\\')[-1]
    new_file_path = os.path.join(save_dir, new_file_name)
    with open(new_file_path, 'w') as f:
        for line in new_file_content:
            f.write(line)
            f.write('\n')
    return


def main():
    save_dir_name = dir_name + '_TrackingNet'
    save_dir = os.path.join(root_dir, save_dir_name)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    '''读取文件'''
    file_paths = glob.glob(os.path.join(root_dir, dir_name, '*.txt'))
    for file_path in file_paths:
        process_per_txt(file_path, save_dir)


if __name__ == '__main__':
    root_dir = '/home/jgao/VisualTracking/ATOM_SGD/Original/pytracking_dimp_RLS_withR/pytracking/'
    dir_name = 'RLStest34'
    main()
