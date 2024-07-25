# set all hyper parameters
opts = {}
opts['resize'] = 64
opts['top_k'] = 5
opts['data_path'] = '../dataset/64/synapsemnist3d_64.npz'
opts['pretrained_network_name'] = 'openclip'
# 'VGG19'
# 'ResNet50'
# 'DenseNet121'
# 'EfficientNetV2M'
# 'biomedclip'
# 'medclip'
# 'UNI'
# 'openclip'

opts['save_train_hard'] = '../dataset/64/synapsemnist3d_64/train/'
opts['save_test_hard'] = '../dataset/64/synapsemnist3d_64/test/'
opts['save_figures'] = '../results/figures/'

opts['framework'] = 'pytorch'
# 'pytorch'
# tf

opts['CNN'] = False
opts['bath_size'] = 128
