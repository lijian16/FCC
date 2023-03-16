# Focal loss on CIFAR-100-LT-100
CUDA_VISIBLE_DEVICES=0 python main/train.py --cfg configs/cao_cifar/re_weighting/focal/cifar100_im100.yaml
#python main/valid.py --cfg configs/cao_cifar/re_weighting/focal/cifar100_im100.yaml --gpus 0 


# focal loss with FCC on CIFAR-100-LT-100
CUDA_VISIBLE_DEVICES=0 python main/train.py --cfg configs/FCC/re_weighting/focal/cifar100_im100.yaml
#python main/valid.py --cfg configs/FCC/re_weighting/focal/cifar100_im100.yaml --gpus 0 

