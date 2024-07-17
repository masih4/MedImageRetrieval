# set all hyper parameters
opts = {}
opts['resize'] = 64
opts['top_k'] = 5
opts['data_path'] = '../dataset/64/adrenalmnist3d_64.npz'
opts['pretrained_network_name'] = 'DenseNet121'
# 'VGG19'
# 'ResNet50'
# 'DenseNet121'
# 'EfficientNetV2M'
# 'biomedclip'

opts['save_train_hard'] = '../dataset/64/adrenalmnist3d_64/train/'
opts['save_test_hard'] = '../dataset/64/adrenalmnist3d_64/test/'
opts['save_figures'] = '../results/figures/'

opts['framework'] = 'tf'
# 'pytorch'
# tf

opts['CNN'] = True
