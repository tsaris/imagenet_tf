#!/bin/bash

# Hyperparameters tuned at scale (1024 nodes)
export HOROVOD_HIERARCHICAL_ALLGATHER=0
export HOROVOD_HIERARCHICAL_ALLREDUCE=0
export HOROVOD_GROUPED_ALLREDUCES=1
export HOROVOD_CYCLE_TIME=1
export HOROVOD_FUSION_THRESHOLD=8388608

if [ "$DATA_MODE" == "real" ]
then
    python resnet_ctl_imagenet_main.py \
    --base_learning_rate=9.5 \
    --batch_size=312 \
    --data_dir=/gpfs/alpine/world-shared/stf011/atsaris/imagenet_all/ \
    --datasets_num_private_threads=32 \
    --dtype=fp16 --enable_device_warmup \
    --enable_eager \
    --epochs_between_evals=4 \
    --eval_dataset_cache \
    --eval_offset_epochs=2 \
    --eval_prefetch_batchs=192 \
    --label_smoothing=0.1 \
    --log_steps=125 \
    --lr_schedule=polynomial \
    --model_dir=/gpfs/alpine/world-shared/stf011/atsaris/mlperf_imagenet_output/ \
    --num_gpus=6 \
    --optimizer=LARS \
    --noreport_accuracy_metrics \
    --single_l2_loss_op \
    --steps_per_loop=514 \
    --tf_gpu_thread_mode=gpu_private \
    --train_epochs=10 \
    --training_dataset_cache \
    --training_prefetch_batchs=128 \
    --verbosity=0 \
    --warmup_epochs=4 \
    --weight_decay=0.0002 \
    --distribution_strategy=$STRATEGY
else
    echo "Not sysntetic implementation yet"
fi

if [ $PMIX_RANK -eq 0 ]
then
  cp /mnt/bb/$USER/log.${LSB_JOBID} .
fi
