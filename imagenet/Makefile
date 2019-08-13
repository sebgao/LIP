train-lip_resnet50:
	CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python main.py --dist-url 'tcp://127.0.0.1:12004' --dist-backend 'nccl' --multiprocessing-distributed --world-size 1 --rank 0 ~/dataset/imagenet-cls/ --module lip_resnet --arch resnet50 -b 256 --num_gpus 8

val-lip_resnet50:
	CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python main.py --dist-url 'tcp://127.0.0.1:12004' --dist-backend 'nccl' --multiprocessing-distributed --world-size 1 --rank 0 ~/dataset/imagenet-cls/ --module lip_resnet --arch resnet50 -b 256 --num_gpus 8 --pretrained --evaluate

val-lip_resnet101:
	CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python main.py --dist-url 'tcp://127.0.0.1:12004' --dist-backend 'nccl' --multiprocessing-distributed --world-size 1 --rank 0 ~/dataset/imagenet-cls/ --module lip_resnet --arch resnet101 -b 256 --num_gpus 8 --pretrained --evaluate
