import os
import glob
import socket
import logging

import tensorflow as tf
import neuralgym as ng

from inpaint_model import InpaintCAModel
import utils
import cv2
import numpy as np

logger = logging.getLogger()


def multigpu_graph_def(model, data, config, gpu_id=0, loss_type='g', masks=None):
    with tf.device('/cpu:0'):
        images = data.data_pipeline(config.BATCH_SIZE)
    if gpu_id == 0 and loss_type == 'g':
        _, _, losses = model.build_graph_with_losses(
            images, config, summary=True, reuse=True, masks=masks)
    else:
        _, _, losses = model.build_graph_with_losses(
            images, config, reuse=True, masks=masks)
    if loss_type == 'g':
        return losses['g_loss']
    elif loss_type == 'd':
        return losses['d_loss']
    else:
        raise ValueError('loss type is not supported.')


if __name__ == "__main__":
    config = ng.Config('inpaint.yml')
    if config.GPU_ID != -1:
        ng.set_gpus(config.GPU_ID) 
    else:
        ng.get_gpus(config.NUM_GPUS,False)# TODO

    # training data
    with open(config.DATA_FLIST[config.DATASET][0]) as f:
        fnames = f.read().splitlines()
    data = ng.data.DataFromFNames(
        fnames, config.IMG_SHAPES, random_crop=config.RANDOM_CROP)
    images = data.data_pipeline(config.BATCH_SIZE)

    masks = []
    if config.FREE_FORM_MASK:
        timer = utils.ElapsedTimer('mask load')
        #print(config.MANGA_MASK_DIR)
        mask_paths = utils.file_paths(config.MANGA_MASK_DIR)
        num_all = 0
        for path in mask_paths:
            num_all += 1

            mask = cv2.imread(path,0)
            if mask is None:
                continue
            mask = (mask > 100).astype(np.float32)

            h,w = mask.shape[:2]
            if np.sum(mask) < h*w//40: # if mask is too sparse,
                continue
            masks.append(mask)

        timer.elapsed_time()
        print('# masks = ', len(masks), '/', num_all, 
              'ratio =', len(masks) / num_all, )

    # main model
    model = InpaintCAModel()
    g_vars, d_vars, losses = model.build_graph_with_losses(
        images, config=config, masks=masks)

    # validation images
    if config.VAL:
        with open(config.DATA_FLIST[config.DATASET][1]) as f:
            val_fnames = f.read().splitlines()
        # progress monitor by visualizing static images
        for i in range(config.STATIC_VIEW_SIZE):
            static_fnames = val_fnames[i:i+1]
            static_images = ng.data.DataFromFNames(
                static_fnames, config.IMG_SHAPES, nthreads=1,
                random_crop=config.RANDOM_CROP).data_pipeline(1)
            static_inpainted_images = model.build_static_infer_graph(
                static_images, config, name='static_view/%d' % i, masks=masks)

    # training settings
    lr = tf.get_variable(
        'lr', shape=[], trainable=False,
        initializer=tf.constant_initializer(1e-4))
    d_optimizer = tf.train.AdamOptimizer(lr, beta1=0.5, beta2=0.9)
    g_optimizer = d_optimizer

    # gradient processor
    if config.GRADIENT_CLIP:
        gradient_processor = lambda grad_var: (
            tf.clip_by_average_norm(grad_var[0], config.GRADIENT_CLIP_VALUE),
            grad_var[1])
    else:
        gradient_processor = None

    # log dir
    print('?? :', ng.date_uid(), socket.gethostname(), config.DATASET, 'MASKED' if config.GAN_WITH_MASK else 'NORMAL', config.GAN,config.LOG_DIR)
    log_prefix = 'model_logs/' + '_'.join([
        ng.date_uid(), socket.gethostname(), config.DATASET,
        'MASKED' if config.GAN_WITH_MASK else 'NORMAL',
        config.GAN,config.LOG_DIR])

    # train discriminator with secondary trainer, should initialize before
    # primary trainer.
    discriminator_training_callback = ng.callbacks.SecondaryTrainer(
        pstep=1,
        optimizer=d_optimizer,
        var_list=d_vars,
        max_iters=5,
        graph_def=multigpu_graph_def,
        graph_def_kwargs=dict(
            model=model, data=data, config=config, loss_type='d',
            masks=masks
        ),
    )

    # train generator with primary trainer
    trainer = ng.train.Trainer(
        optimizer=g_optimizer,
        var_list=g_vars,
        max_iters=config.MAX_ITERS,
        graph_def=multigpu_graph_def,
        grads_summary=config.GRADS_SUMMARY,
        gradient_processor=gradient_processor,
        graph_def_kwargs=dict(
            model=model, data=data, config=config, loss_type='g',
            masks=masks
        ),
        spe=config.TRAIN_SPE,
        log_dir=log_prefix,
    )

    # add all callbacks
    if not config.PRETRAIN_COARSE_NETWORK:
        trainer.add_callbacks(discriminator_training_callback)
    trainer.add_callbacks([
        ng.callbacks.WeightsViewer(),
        ng.callbacks.ModelRestorer(trainer.context['saver'], dump_prefix='model_logs/'+config.MODEL_RESTORE+'/snap', optimistic=True),
        ng.callbacks.ModelSaver(config.TRAIN_SPE, trainer.context['saver'], log_prefix+'/snap'),
        ng.callbacks.SummaryWriter((config.VAL_PSTEPS//1), trainer.context['summary_writer'], tf.summary.merge_all()),
    ])
    # launch training
    trainer.train()
