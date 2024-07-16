# set all hyper parameters
opts = {}
opts['resize'] = 128
opts['top_k'] = 5
opts['data_path'] = '../dataset/128/pathmnist_128.npz'
opts['pretrained_network_name'] = 'biomedclip'
# 'VGG19'
# 'ResNet50'
# 'DenseNet121'
# 'EfficientNetV2M'
# 'biomedclip'

opts['save_train_hard'] = '../dataset/128/pathmnist_128/train/'
opts['save_test_hard'] = '../dataset/128/pathmnist_128/test/'
opts['save_figures'] = '../results/figures/'

