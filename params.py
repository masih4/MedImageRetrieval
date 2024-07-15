# set all hyper parameters
opts = {}
opts['resize'] = 32
opts['top_k'] = 5
opts['data_path'] = '../dataset/28/synapsemnist3d.npz'
opts['pretrained_network_name'] = 'EfficientNetV2M'
# 'VGG19'
# 'ResNet50'
# 'DenseNet121'
# 'EfficientNetV2M'

opts['save_train_hard'] = '../dataset/28/synapsemnist3d_28/train/'
opts['save_test_hard'] = '../dataset/28/synapsemnist3d_28/test/'
opts['save_figures'] = '../results/figures/'

