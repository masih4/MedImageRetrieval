# set all hyper parameters
opts = {}
opts['resize'] = 32
opts['top_k'] = 5
opts['data_path'] = '../dataset/28/adrenalmnist3d.npz'
opts['pretrained_network_name'] = 'VGG19'
# 'VGG19'
# 'ResNet50'
# 'DenseNet121'
# 'EfficientNetV2M'

opts['save_train_hard'] = '../dataset/28/adrenalmnist3d_28/train/'
opts['save_test_hard'] = '../dataset/28/adrenalmnist3d_28/test/'
opts['save_figures'] = '../results/figures/'

