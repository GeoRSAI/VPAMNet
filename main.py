

# coding=utf-8
if __name__ == '__main__': #避免num_works>0,引起的来回重新调用程序
    import sys
    sys.path.append("/media/csu/sda2/xjwDLexp/exp010_wavelet/DinoUNet")
    ### 引用各种第三方库
    import os
    import re
    import random
    import numpy as np
    import time
    import shutil

    import torch
    import torch.nn.parallel
    import torch.backends.cudnn as cudnn
    import torch.optim

    import torch.utils.data
    import torch.utils.data.distributed
    from torch.optim.optimizer import Optimizer

    from data import Data
    from logger import Logger
    from utils import Timer
    from utils import plot_images_single, plot_images_single_color
    from utils import AverageMeter, ProgressMeter, get_evaluat, get_multi_evaluat, SegmentationEvaluator
    from utils import cross_entropy_loss, loss_calc, loss_calc_v2, FCCDN_loss_without_seg, logger_updata
    from utils import write_metrics_to_excel
    from model import model_dict
    from data import num_classes_dict
    from option import args
    import itertools
    from skimage import io
    from tqdm import tqdm
    from torch.autograd import Variable
    from sklearn.metrics import confusion_matrix
    import glob

    ### 加载功能模块
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    pjoin = os.path.join
    global logger
    logger = Logger(args)
    # logprint = logger.log_printer.logprint
    # accprint = logger.log_printer.accprint
    # netprint = logger.netprint
    # timer = Timer(args.epochs)


    def main():
        logprint = logger.log_printer.logprint
        accprint = logger.log_printer.accprint
        netprint = logger.netprint
        timer = Timer(args.epochs)

        if args.seed is not None:
            random.seed(args.seed)
            np.random.seed(args.seed)
            torch.manual_seed(args.seed)
            os.environ['PYTHONHASHSEED'] = str(args.seed)
            logprint("=>已使用seed= {}".format(args.seed))
        if args.gpu is not None or torch.cuda.is_available():
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            if device != 'cpu':
                args.gpu = device
                torch.cuda.manual_seed(args.seed)
                torch.cuda.manual_seed_all(args.seed)
                cudnn.benchmark = False
                cudnn.deterministic = True
                torch.use_deterministic_algorithms(True)
                # 向日志中写入使用的GPU信息
                logprint("=>设定显卡为：{},显卡中设定seed= {}".format(args.gpu, args.seed))
            else:
                logprint("=>显卡不可用，args.gpu将重置为空")
                args.gpu = None

        global best_index, best_epoch

        loader = Data(args)

        train_loader_task1 = loader.train_loader_task1
        val_loader_task1 = loader.test_loader_task1


        # num_classes = num_classes_dict[args.dataset]
        WEIGHTS = torch.ones(args.num_classes)
        if args.dataset == 'gid15' or args.dataset == "FUSU" or args.dataset == "Vaihingen_":
            WEIGHTS[0] = 0
        if args.dataset == "Vaihingen_":
            WEIGHTS[-1] = 2

        logprint("Use Model: '{}' for training".format(args.arch))
        model = model_dict[args.arch](num_classes=args.num_classes, num_channels=3, use_bn=False)
        model.to(args.gpu)

        # optimizer = torch.optim.SGD(model.parameters(), lr=args.LR, momentum=args.momentum, weight_decay=args.wd)
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.LR, weight_decay=args.weight_decay)
        # We define the scheduler
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.milestones, gamma=args.gamma)
        # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=16)

        criterion = loss_calc

        # @mst: save the model after initialization if necessary
        # 记录下初始化参数
        if args.save_init_model:
            state = {
                'arch': args.arch,
                'model': model,
                'state_dict': model.state_dict(),
                'ExpID': logger.ExpID,
            }
            save_model(state, mark='init')


        netprint(model, comment='base model arch')

        best_index = 0.
        best_epoch = 0
        # 是否从检查点恢复
        if args.resume:
            if os.path.isfile(args.resume):
                logprint("=> loading checkpoint ''".format(args.resume))
                if args.gpu is not None:
                    checkpoint = torch.load(args.resume)
                else:
                    #
                    loc = 'cuda:{}'.format(args.gpu)
                    checkpoint = torch.load(args.resume, map_location=loc)
                args.start_epoch = checkpoint['epoch']
                best_index = checkpoint['mIou']
                # if args.gpu is not None:
                #     best_index = best_index.to(args.gpu)
                try:
                    best_epoch = checkpoint['best_epoch']
                except:
                    pass
                model.load_state_dict(checkpoint['state_dict'])
                optimizer.load_state_dict(checkpoint['optimizer'])
                # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=checkpoint['epoch'])
                Resume_ExpID = checkpoint['ExpID']
                logger_updata(logger, Resume_ExpID)
                logprint = logger.log_printer.logprint



                logprint("=> loaded checkpoint '{}' (epoch {})"
                         .format(args.resume, checkpoint['epoch']))
            else:
                logprint("=> no checkpoint found at '{}'".format(args.resume))


        if hasattr(model, "module"):
            model = model.module
        else:
            model = model

        top_epochs = []  # 存 epoch 编号，按 mIoU 降序（top_epochs[0] 是最好）
        top_mious = []  # 存对应 mIoU

        for epoch in range(args.start_epoch+1,args.epochs+1):

            train_loss, train_mF1S, train_mIou\
                = train(train_loader_task1, model, criterion, optimizer, scheduler, epoch, args, print_log=True,weights=WEIGHTS)

            test_loss, test_mF1S, test_mIou, test_F1S, test_Iou, test_conf\
                = validate(val_loader_task1, model, criterion, args, logger, epoch,weights=WEIGHTS)

            # ✅ 【核心逻辑】更新 top-3 并删除最差者（若需）
            if len(top_epochs) < 3:
                top_epochs.append(epoch)
                top_mious.append(test_mIou)
                sorted_pairs = sorted(zip(top_mious, top_epochs), reverse=True)
                top_mious, top_epochs = map(list, zip(*sorted_pairs))
            else:
                if test_mIou > top_mious[-1]:
                    old_plot_dir = pjoin(logger.exp_path, "plot", f"Epoch{top_epochs[-1]}")
                    try:
                        if os.path.exists(old_plot_dir):
                            shutil.rmtree(old_plot_dir)
                            # print(f"🗑️  删除 Epoch{top_epochs[-1]} 可视化")
                    except Exception as e:
                        print(f"⚠️  删除失败: {e}")

                    # 更新列表：替换最后一个
                    top_epochs[-1] = epoch
                    top_mious[-1] = test_mIou
                    # 重新降序排序
                    sorted_pairs = sorted(zip(top_mious, top_epochs), reverse=True)
                    top_mious, top_epochs = map(list, zip(*sorted_pairs))
                else:
                    old_plot_dir = pjoin(logger.exp_path, "plot", f"Epoch{epoch}")
                    try:
                        if os.path.exists(old_plot_dir):
                            shutil.rmtree(old_plot_dir)
                            # print(f"🗑️  删除 Epoch{top_epochs[-1]} 可视化")
                    except Exception as e:
                        print(f"⚠️  删除失败: {e}")

            is_best = test_mIou > best_index
            if is_best:
                best_epoch = epoch
            best_index = max(test_mIou, best_index)

            accprint(
                "train_loss %.4f train_mF1S %.4f train_mIou %.4f  | Epoch %d (best_index %.4f @ Best_Epoch %d)" %
                (train_loss, train_mF1S, train_mIou, epoch, best_index, best_epoch))

            accprint(
                "test_loss %.4f test_F1S %.4f test_mIou %.4f | Epoch %d (best_index %.4f @ Best_Epoch %d) F1 %s Iou %s conf %s" %
                (test_loss, test_mF1S, test_mIou, epoch, best_index, best_epoch, test_F1S, test_Iou, test_conf))
            logprint('predicted finish time: %s' % timer())

            # 写入指标到Excel
            try:
                write_metrics_to_excel(
                    logger=logger,
                    epoch=epoch,
                    train_metrics=(train_loss, train_mF1S, train_mIou),
                    test_metrics=(test_loss, test_mF1S, test_mIou),
                    test_f1s=test_F1S,
                    test_ious=test_Iou,
                    test_conf=test_conf,
                    num_classes=args.num_classes-1
                )
            except Exception as e:
                print(f"Error in write_metrics_to_excel: {e}")

            if args.arch:
                # @mst: use our own save func
                state = {'epoch': epoch,
                         'arch': args.arch,
                         'model': model,
                         'state_dict': model.state_dict(),
                         'mF1S': test_mF1S,
                         'mIou': test_mIou,
                         'optimizer': optimizer.state_dict(),
                         'ExpID': logger.ExpID,
                         'best_epoch': best_epoch,
                         }

                save_model(state, is_best, mark='Last')

            if epoch % 10 == 0:  # 每10个epoch清理一次
                torch.cuda.empty_cache()


    def train(train_loader_task1,
              model,
              criterion,
              optimizer,
              scheduler,
              epoch,
              args,
              print_log,
              weights,
              ):
        batch_time = AverageMeter('Time', ':6.3f')
        data_time = AverageMeter('Data', ':6.3f')
        loss_ = AverageMeter('Loss', ':.4e')
        ## 精度
        mF1S = AverageMeter('mF1S', ':6.2f')
        mIou = AverageMeter('mIou', ':6.2f')

        progress = ProgressMeter(
            len(train_loader_task1),
            [batch_time, data_time, loss_,  mF1S, mIou],
            prefix="Epoch: [{}]".format(epoch))
        model.train()
        weights = weights.cuda()
        if args.dataset == 'gid15' or args.dataset == "FUSU":
            conf_matrix = get_multi_evaluat(num_classes=args.num_classes, ignore=0)
            # conf_matrix = SegmentationEvaluator(num_classes=args.num_classes, ignore=0)
        else:
            conf_matrix = get_multi_evaluat(num_classes=args.num_classes,ignore=0)
            # conf_matrix = SegmentationEvaluator(num_classes=args.num_classes, ignore=0)
            # conf_matrix = SegmentationEvaluator(num_classes=args.num_classes)
        optimizer.zero_grad()
        end = time.time()
        batch_time.update(time.time() - end)
        for i, (images, target) in enumerate(train_loader_task1):
            if args.gpu is not  None:
                images = images.cuda(args.gpu,non_blocking=True)
                target = target.cuda(args.gpu,non_blocking=True).to(torch.int32)
            if torch.sum(target) == 0:
                continue

            output = model(images)
            output = torch.nn.Softmax(dim=1)(output)
            loss= criterion(output, target,weights) / args.itersize
            # loss= criterion(output, target) / args.itersize
            loss.backward()

            output = torch.argmax(output,dim=1)
            conf_matrix.calculator(output, target)
            mIoU, IoU = conf_matrix.cal_miou()
            mean_F1, F1 = conf_matrix.cal_F1score()

            if (i+1)% args.itersize == 0:
                optimizer.step()
                optimizer.zero_grad()

                loss_.update(loss.item()*args.itersize, images.size(0))
                # 精度
                mF1S.update(mean_F1, 1)
                mIou.update(mIoU, 1)
                batch_time.update(time.time() - end)
                end = time.time()
            if print_log and i % args.print_freq == 0:
                progress.display(i)

        if scheduler is not None:
            scheduler.step()
        return loss_.val, mF1S.val, mIou.val
        ###
    def validate(val_loader, model, criterion, args,logger, epoch, weights):
        batch_time = AverageMeter('Time', ':6.3f')
        loss_ = AverageMeter('Loss', ':.4e')
        mF1S = AverageMeter('mF1S', ':6.2f')
        mIou = AverageMeter('mIou', ':6.2f')

        progress = ProgressMeter(
            len(val_loader),
            [batch_time, loss_, mF1S, mIou],
            prefix='Test: ')


        model.eval()
        weights = weights.cuda()
        if args.dataset == 'gid15' or args.dataset == "FUSU":
            conf_matrix = get_multi_evaluat(num_classes=args.num_classes, ignore=0)
            # conf_matrix = SegmentationEvaluator(num_classes=args.num_classes, ignore=0)
        else:
            conf_matrix = get_multi_evaluat(num_classes=args.num_classes, ignore=0)
            # conf_matrix = SegmentationEvaluator(num_classes=args.num_classes, ignore=0)
            # conf_matrix = SegmentationEvaluator(num_classes=args.num_classes + 1, ignore=5)
            conf_matrix2 = get_multi_evaluat(num_classes=args.num_classes, ignore=0, device='cpu')
        val_path_list = val_loader.dataset.img_path
        filename = [os.path.splitext(os.path.basename(path))[0] for path in val_path_list]
        # with torch.no_grad():
        with torch.inference_mode():
            end = time.time()
            for i, (images, target) in enumerate(val_loader):
                if args.gpu is not  None:
                    images = images.cuda(args.gpu,non_blocking=True)
                    target = target.cuda(args.gpu,non_blocking=True).to(torch.int32)
                    # target = target.cuda(args.gpu,non_blocking=True)
                output = model(images.to(args.gpu))
                output = torch.nn.Softmax(dim=1)(output)
                if torch.sum(target) == 0: #标签全部为0的时候，loss会报错，跳过去，但是不反向传播，对结果没有任何影响。
                    loss = torch.tensor(loss_.avg)
                else:
                    loss = criterion(output, target, weights)
                    # loss = criterion(output, target)
                    # loss_change, diceloss, foclaloss = criterion(output, target) / args.itersize
                    # loss = loss_change.mean()

                output = torch.argmax(output, dim=1)
                conf_matrix.calculator(output, target)
                mIoU, IoU = conf_matrix.cal_miou()
                mean_F1, F1 = conf_matrix.cal_F1score()

                loss_.update(loss.item(), images.size(0))
                mF1S.update(mean_F1, 1)
                mIou.update(mIoU, 1)

                # measure elapsed time
                batch_time.update(time.time() - end)
                end = time.time()

                if i != 0 and i % args.print_freq == 0:
                    progress.display(i)
                plot_images_single_color(args, logger, epoch, i, output, filename)
                # plot_images_single_color(args, logger, epoch, i, output.squeeze(), filename)
            conf_matrix2 = tes_whole_img(args, model, test_ids=[],conf_matrix = conf_matrix2, all=False, stride=256, batch_size=10, window_size=(256, 256))
            mIoU, IoU = conf_matrix2.cal_miou()
            mean_F1, F1 = conf_matrix2.cal_F1score()
            mF1S.update(mean_F1, 1)
            mIou.update(mIoU, 1)
        # return loss_.avg, mF1S.val, mIou.val, ' '.join(map(str, F1.tolist())), ' '.join(map(str, IoU.tolist())), ' '.join(map(str, conf_matrix.return_conf().flatten().tolist()))
        return loss_.avg, mF1S.val, mIou.val, ' '.join(map(str, F1.tolist())), ' '.join(map(str, IoU.tolist())), ' '.join(map(str, conf_matrix2.return_conf().flatten().tolist()))




    def convert_from_color(arr_3d):
        """ RGB-color encoding to grayscale labels """

        # palette = {0: (255, 255, 255),  # Impervious surfaces (white)
        #            1: (0, 0, 255),  # Buildings (blue)
        #            2: (0, 255, 255),  # Low vegetation (cyan)
        #            3: (0, 255, 0),  # Trees (green)
        #            4: (255, 255, 0),  # Cars (yellow)
        #            5: (255, 0, 0),  # Clutter (red)
        #            6: (0, 0, 0)}  # Undefined (black)

        palette = {1: (255, 255, 255),  # Impervious surfaces (white)
                   2: (0, 0, 255),  # Buildings (blue)
                   3: (0, 255, 255),  # Low vegetation (cyan)
                   4: (0, 255, 0),  # Trees (green)
                   5: (255, 255, 0),  # Cars (yellow)
                   0: (255, 0, 0),  # Clutter (red)
                   6: (0, 0, 0)}  # Undefined (black)

        invert_palette = {v: k for k, v in palette.items()}
        palette = invert_palette
        arr_2d = np.zeros((arr_3d.shape[0], arr_3d.shape[1]), dtype=np.uint8)

        for c, i in palette.items():
            m = np.all(arr_3d == np.array(c).reshape(1, 1, 3), axis=2)
            arr_2d[m] = i
        arr_2d[arr_2d] = 0
        return arr_2d

    def grouper(n, iterable):
        """ Browse an iterator by chunk of n elements """
        it = iter(iterable)
        while True:
            chunk = tuple(itertools.islice(it, n))
            if not chunk:
                return
            yield chunk

    def sliding_window(top, step=10, window_size=(20, 20)):
        """ Slide a window_shape window across the image with a stride of step """
        for x in range(0, top.shape[0], step):
            if x + window_size[0] > top.shape[0]:
                x = top.shape[0] - window_size[0]
            for y in range(0, top.shape[1], step):
                if y + window_size[1] > top.shape[1]:
                    y = top.shape[1] - window_size[1]
                yield x, y, window_size[0], window_size[1]


    def metrics(predictions, gts, label_values):
        cm = confusion_matrix(
            gts,
            predictions,
            labels=range(len(label_values)))

        print("Confusion matrix :")
        print(cm)
        # Compute global accuracy
        total = sum(sum(cm))
        accuracy = sum([cm[x][x] for x in range(len(cm))])
        accuracy *= 100 / float(total)
        print("%d pixels processed" % (total))
        print("Total accuracy : %.2f" % (accuracy))

        Acc = np.diag(cm) / cm.sum(axis=1)
        for l_id, score in enumerate(Acc):
            print("%s: %.4f" % (label_values[l_id], score))
        print("---")

        # Compute F1 score
        F1Score = np.zeros(len(label_values))
        for i in range(len(label_values)):
            try:
                F1Score[i] = 2. * cm[i, i] / (np.sum(cm[i, :]) + np.sum(cm[:, i]))
            except:
                # Ignore exception if there is no element in class i for test set
                pass
        print("F1Score :")
        for l_id, score in enumerate(F1Score):
            print("%s: %.4f" % (label_values[l_id], score))
        print('mean F1Score: %.4f' % (np.nanmean(F1Score[:5])))
        print("---")

        # Compute kappa coefficient
        total = np.sum(cm)
        pa = np.trace(cm) / float(total)
        pe = np.sum(np.sum(cm, axis=0) * np.sum(cm, axis=1)) / float(total * total)
        kappa = (pa - pe) / (1 - pe)
        print("Kappa: %.4f" % (kappa))

        # Compute MIoU coefficient
        MIoU = np.diag(cm) / (np.sum(cm, axis=1) + np.sum(cm, axis=0) - np.diag(cm))
        print(MIoU)
        MIoU = np.nanmean(MIoU[:5])
        print('mean MIoU: %.4f' % (MIoU))
        print("---")

        return MIoU

    # def tes_whole_img(args, net, test_ids,conf_matrix, all=False, stride=256, batch_size=10, window_size=(256, 256)):
    #     root_path = args.root_path
    #     img_path = root_path + '/test/images'
    #     lab_path = root_path + '/test/labels'
    #     test_img = glob.glob(img_path + '/*.tif')
    #     test_eroded = glob.glob(lab_path + '/*.tif')
    #     test_img.sort(key=lambda x: x.split('/')[-1].split('.tif')[0])
    #     test_eroded.sort(key=lambda x: x.split('/')[-1].split('.tif')[0])
    #     # Use the network on the test set
    #     test_images = (1 / 255 * np.asarray(io.imread(id), dtype='float32') for id in test_img)
    #     # test_labels = (np.asarray(io.imread(id), dtype='uint8') for id in test_ids)
    #     eroded_labels = (convert_from_color(io.imread(id)) for id in test_eroded)
    #     all_preds = []
    #     all_gts = []
    #     # Switch the network to inference mode
    #     with torch.no_grad():
    #         for img, gt_e in tqdm(zip(test_images, eroded_labels), total=len(test_ids)):
    #             pred = np.zeros(img.shape[:2] + (args.num_classes,))
    #             # total = count_sliding_window(img, step=stride, window_size=window_size) // batch_size
    #             # for i, coords in enumerate(
    #             #         tqdm(grouper(batch_size, sliding_window(img, step=stride, window_size=window_size)), total=total,
    #             #             leave=False)):
    #             for i, coords in enumerate(
    #                     grouper(batch_size, sliding_window(img, step=stride, window_size=window_size))):
    #                 # Build the tensor
    #                 image_patches = [np.copy(img[x:x + w, y:y + h]).transpose((2, 0, 1)) for x, y, w, h in coords]
    #                 image_patches = np.asarray(image_patches)
    #                 with torch.no_grad():
    #                     image_patches = Variable(torch.from_numpy(image_patches).cuda())
    #                 # image_patches = Variable(torch.from_numpy(image_patches).cuda(), volatile=True)
    #
    #                 # Do the inference
    #                 outs = net(image_patches)
    #                 outs = outs.data.cpu().numpy()
    #
    #                 # Fill in the results array
    #                 for out, (x, y, w, h) in zip(outs, coords):
    #                     out = out.transpose((1, 2, 0))
    #                     pred[x:x + w, y:y + h] += out
    #                 del (outs)
    #
    #             pred = np.argmax(pred, axis=-1)
    #
    #             all_preds.append(pred)
    #             gt_e = gt_e +1
    #             gt_e[gt_e==7]=0
    #             gt_e[gt_e==6]=0
    #             all_gts.append(gt_e)
    #     conf_matrix.calculator(torch.tensor(np.concatenate([p.ravel() for p in all_preds]),requires_grad=False).cuda(), torch.tensor(np.concatenate([p.ravel() for p in all_gts]).ravel(),requires_grad=False).to(torch.int64).cuda())
    #     if all:
    #         # return accuracy, all_preds, all_gts
    #         return conf_matrix, all_preds, all_gts
    #     else:
    #         # return accuracy
    #         return conf_matrix

    def tes_whole_img(args, net, test_ids, conf_matrix, all=False, stride=256, batch_size=10, window_size=(256, 256)):
        root_path = args.root_path
        img_path = os.path.join(root_path, 'test', 'images')
        lab_path = os.path.join(root_path, 'test', 'labels')
        test_img = glob.glob(os.path.join(img_path, '*.tif'))
        test_eroded = glob.glob(os.path.join(lab_path, '*.tif'))

        # ==============================
        def get_area_id(filepath):
            """提取 'area10' 这样的 ID"""
            basename = os.path.basename(filepath)
            match = re.search(r'area(\d+)', basename)
            return f"area{match.group(1)}" if match else basename
        # 构建映射
        img_map = {get_area_id(f): f for f in test_img}
        eroded_map = {get_area_id(f): f for f in test_eroded}
        # 取交集并按数字排序（确保 area2 < area10）
        common_areas = sorted(
            set(img_map.keys()) & set(eroded_map.keys()),
            key=lambda x: int(re.search(r'\d+', x).group())  # 按数字升序：area2, area4, ..., area10...
        )
        # 更新原变量：严格对齐！
        test_img = [img_map[aid] for aid in common_areas]
        test_eroded = [eroded_map[aid] for aid in common_areas]

        test_images = (1 / 255 * np.asarray(io.imread(id), dtype='float32') for id in test_img)
        eroded_labels = (convert_from_color(io.imread(id)) for id in test_eroded)

        # 不再收集 all_preds / all_gts，而是直接更新 conf_matrix
        with torch.no_grad():
            for img, gt_e in tqdm(zip(test_images, eroded_labels), total=len(test_img)):
                pred = np.zeros(img.shape[:2] + (args.num_classes,), dtype='float32')
                for coords in grouper(batch_size, sliding_window(img, step=stride, window_size=window_size)):
                    image_patches = [np.copy(img[x:x + w, y:y + h]).transpose((2, 0, 1)) for x, y, w, h in coords]
                    image_patches = np.asarray(image_patches, dtype='float32')
                    image_patches = torch.from_numpy(image_patches).cuda()

                    outs = net(image_patches)  # [B, C, H, W]
                    outs = outs.cpu().numpy()  # 移到 CPU 节省显存

                    for out, (x, y, w, h) in zip(outs, coords):
                        out = out.transpose((1, 2, 0))
                        pred[x:x + w, y:y + h] += out

                pred = np.argmax(pred, axis=-1).astype(np.int64)  # [H, W]
                gt_e = gt_e
                gt_e[gt_e == 6] = 0
                gt_e = gt_e.astype(np.int32)

                # >>>>>> 关键修改：逐图更新混淆矩阵，且在 CPU 上 <<<<<<
                pred_flat = torch.from_numpy(pred.ravel())
                target_flat = torch.from_numpy(gt_e.ravel())
                # 只保留有效标签（可选）
                valid = target_flat != 255  # 如果有 ignore label
                pred_flat = pred_flat[valid]
                target_flat = target_flat[valid]

                # 在 CPU 上更新，避免 GPU OOM
                conf_matrix.calculator(pred_flat, target_flat)

        if all:
            return conf_matrix, None, None  # 如果需要返回预测，可额外处理
        else:
            return conf_matrix

    def save_model(state, is_best=False, mark=''):
        # out = pjoin(logger.weights_path, "checkpoint.pth")
        # torch.save(state, out)
        if is_best:
            out_best = pjoin(logger.weights_path, "checkpoint_best.pth")
            torch.save(state, out_best)
        if mark:
            out_mark = pjoin(logger.weights_path, "checkpoint_{}.pth".format(mark))
            torch.save(state, out_mark)



    def apply_weights( weights, model):
        """flush weights to model"""
        model.load_state_dict(weights)


    main()

