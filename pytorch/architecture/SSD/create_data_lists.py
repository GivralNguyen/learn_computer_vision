from utils import create_data_lists

if __name__ == '__main__':
    create_data_lists(voc07_path='/media/HDD/learncv/learn_computer_vision/pytorch/architecture/SSD/ssd_data/VOCdevkit/VOC2007',
                      voc12_path='/media/HDD/learncv/learn_computer_vision/pytorch/architecture/SSD/ssd_data/VOCdevkit/VOC2012',
                      output_folder='./')
