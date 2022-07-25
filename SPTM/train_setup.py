import yaml
import numpy as np

config_data = yaml.load(open('/home/charles-chen/TopoNavigation/SPTM/turtlebot_nav.yaml', "r"), Loader=yaml.FullLoader)
scene = ['Beechwood_0_int', 'Benevolence_0_int', 'Benevolence_1_int', 'Benevolence_2_int', 'Ihlen_0_int', 'Merom_0_int',
         'Merom_1_int', 'Pomaria_1_int', 'Pomaria_2_int ', 'Rs_int', 'Wainscott_0_int', 'Wainscott_1_int']
scene_id = 'Beechwood_0_int'
type = "random"

def to_categorical(y, num_classes=None, dtype='float32'):
    # 将输入y向量转换为数组
    y = np.array(y, dtype='int')
    # 获取数组的行列大小
    input_shape = y.shape
    if input_shape and input_shape[-1] == 1 and len(input_shape) > 1:
        input_shape = tuple(input_shape[:-1])
    # y变为1维数组
    y = y.ravel()
    # 如果用户没有输入分类个数，则自行计算分类个数
    if not num_classes:
        num_classes = np.max(y) + 1
    n = y.shape[0]
    # 生成全为0的n行num_classes列的值全为0的矩阵
    categorical = np.zeros((n, num_classes), dtype=dtype)
    # np.arange(n)得到每个行的位置值，y里边则是每个列的位置值
    categorical[np.arange(n), y] = 1
    # 进行reshape矫正
    output_shape = input_shape + (num_classes,)
    categorical = np.reshape(categorical, output_shape)
    return categorical