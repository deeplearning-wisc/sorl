import time
import os
import torch
import torch.nn as nn
import torch.nn.functional as F 
import torchvision
from itertools import cycle
import numpy as np
import argparse
from arguments import set_deterministic, Namespace, csv, shutil, yaml
from augmentations import get_aug
from models import get_model
from datasets import get_dataset
from optimizers import get_optimizer, LR_Scheduler
from datetime import date
from utils import plot_cluster, accuracy, torch_l2_dis_batch
from sklearn.cluster import KMeans
from ylib.ytool import cluster_acc

from PIL import ImageFile
from sklearn import metrics

ImageFile.LOAD_TRUNCATED_IMAGES = True

def main(log_writer, log_file, device, args):
    iter_count = 0

    import open_world_cifar as datasets
    dataroot = args.data_dir
    if args.dataset.name == 'cifar10':
        train_set_l = datasets.OPENWORLDCIFAR10(root=dataroot, labeled=True, train=True, labeled_num=args.labeled_num, labeled_ratio=args.labeled_ratio, download=True, transform=get_aug(train=True, **args.aug_kwargs))
        train_set_u = datasets.OPENWORLDCIFAR10(root=dataroot, labeled=False, train=True, labeled_num=args.labeled_num, labeled_ratio=args.labeled_ratio, download=True, transform=get_aug(train=True, **args.aug_kwargs), unlabeled_idxs=train_set_l.unlabeled_idxs)
        eval_set_l = datasets.OPENWORLDCIFAR10(root=dataroot, labeled=True, train=True, labeled_num=args.labeled_num, labeled_ratio=args.labeled_ratio, download=True, transform=get_aug(train=False, train_classifier=False, **args.aug_kwargs))
        eval_set_u = datasets.OPENWORLDCIFAR10(root=dataroot, labeled=False, train=True, labeled_num=args.labeled_num, labeled_ratio=args.labeled_ratio, download=True, transform=get_aug(train=False, train_classifier=False, **args.aug_kwargs), unlabeled_idxs=train_set_l.unlabeled_idxs)
        args.num_classes = 10
    elif args.dataset.name == 'cifar100':
        # known_class_division_1 = [
        #     "beaver", "dolphin", "otter", "seal", "whale", "aquarium_fish", "flatfish", "ray", "shark", "trout",
        #     "orchid", "poppy", "rose", "sunflower", "tulip", "bottle", "bowl", "can", "cup", "plate",
        #     "apple", "mushroom", "orange", "pear", "sweet_pepper", "clock", "keyboard", "lamp",
        #     "telephone", "television", "bed", "chair", "couch", "table", "wardrobe", "bee", "beetle", "butterfly",
        #     "caterpillar", "cockroach", "bear", "leopard", "lion", "tiger", "wolf", "bridge", "castle", "house", "road",
        #     "skyscraper"
        # ]
        #
        # known_class_division_2 = [
        #     "beaver", "dolphin", "otter", "aquarium_fish", "flatfish", "orchid", "poppy", "rose", "bottle",
        #     "bowl", "apple", "mushroom", "orange", "clock", "keyboard", "bed", "chair", "couch", "bee", "beetle",
        #     "bear", "leopard", "lion", "bridge", "castle", "cloud", "forest", "mountain", "camel", "cattle", "fox",
        #     "porcupine", "possum", "crab", "lobster", "baby", "boy", "girl", "crocodile", "dinosaur", "hamster",
        #     "mouse", "rabbit", "maple_tree", "oak_tree", "bicycle", "bus", "motorcycle", "lawn_mower", "rocket"
        # ]
        class_list = None
        
        train_set_l = datasets.OPENWORLDCIFAR100(root=dataroot, labeled=True, train=True, labeled_num=args.labeled_num, labeled_ratio=args.labeled_ratio, download=True, transform=get_aug(train=True, **args.aug_kwargs), class_list=class_list)
        train_set_u = datasets.OPENWORLDCIFAR100(root=dataroot, labeled=False, train=True, labeled_num=args.labeled_num, labeled_ratio=args.labeled_ratio, download=True, transform=get_aug(train=True, **args.aug_kwargs), unlabeled_idxs=train_set_l.unlabeled_idxs, class_list=class_list)
        eval_set_l = datasets.OPENWORLDCIFAR100(root=dataroot, labeled=True, train=True, labeled_num=args.labeled_num, labeled_ratio=args.labeled_ratio, download=True, transform=get_aug(train=False, train_classifier=False, **args.aug_kwargs), class_list=class_list)
        eval_set_u = datasets.OPENWORLDCIFAR100(root=dataroot, labeled=False, train=True, labeled_num=args.labeled_num, labeled_ratio=args.labeled_ratio, download=True, transform=get_aug(train=False, train_classifier=False, **args.aug_kwargs), unlabeled_idxs=train_set_l.unlabeled_idxs, class_list=class_list)
        args.num_classes = 100

    args.penul_feat_dim = 512


    labeled_len = len(train_set_l)
    unlabeled_len = len(train_set_u)
    labeled_batch_size = int(args.train.batch_size * labeled_len / (labeled_len + unlabeled_len))

    # Initialize the splits
    train_label_loader = torch.utils.data.DataLoader(train_set_l, batch_size=labeled_batch_size, shuffle=True, num_workers=2, drop_last=True)
    train_unlabel_loader = torch.utils.data.DataLoader(train_set_u, batch_size=args.train.batch_size - labeled_batch_size, shuffle=True, num_workers=2, drop_last=True)
    test_label_loader = torch.utils.data.DataLoader(eval_set_l, batch_size=100, shuffle=False, num_workers=1)
    test_unlabel_loader = torch.utils.data.DataLoader(eval_set_u, batch_size=100, shuffle=False, num_workers=1)

    # define model
    model = get_model(args.model, args).to(device)
    # model = torch.nn.DataParallel(model)

    if args.dataset.name == 'cifar10':
        state_dict = torch.load('./pretrained/spectral_cifar_10.pth.tar')['state_dict']
        state_dict.pop('label_stat', None)
    elif args.dataset.name == 'cifar100':
        state_dict = torch.load('./pretrained/spectral_cifar_100.pth.tar')['state_dict']

    state_dict = {k: v for k, v in state_dict.items() if 'encoder' not in k}
    model.load_state_dict(state_dict, strict=False)

    for name, param in model.named_parameters():
        if 'proj' not in name and 'layer4' not in name:
            param.requires_grad = False


    # for name, param in model.named_parameters():
    #     print(f"{name}: {param.requires_grad}")
    model = model.to(device)

    # define optimizer
    optimizer = get_optimizer(
        args.train.optimizer.name, model, 
        lr=args.train.base_lr*args.train.batch_size/256, 
        momentum=args.train.optimizer.momentum,
        weight_decay=args.train.optimizer.weight_decay)

    lr_scheduler = LR_Scheduler(
        optimizer,
        args.train.warmup_epochs, args.train.warmup_lr*args.train.batch_size/256, 
        args.train.num_epochs, args.train.base_lr*args.train.batch_size/256, args.train.final_lr*args.train.batch_size/256, 
        len(train_label_loader),
        constant_predictor_lr=True # see the end of section 4.2 predictor
    )

    ckpt_dir = os.path.join(args.log_dir, "checkpoints")
    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)

    for epoch in range(0, args.train.stop_at_epoch):
        model.reset_stat()

        model.train()
        #######################  Train #######################
        print("number of iters this epoch: {}".format(len(train_label_loader)))
        unlabel_loader_iter = cycle(train_unlabel_loader)
        for idx, ((x1, x2), target) in enumerate(train_label_loader):

            ((ux1, ux2), target_unlabeled) = next(unlabel_loader_iter)
            x1, x2, ux1, ux2, target, target_unlabeled = x1.to(device), x2.to(device), ux1.to(device), ux2.to(device), target.to(device), target_unlabeled.to(device)

            model.zero_grad()
            data_dict = model.forward_sorl(x1, x2, ux1, ux2, target)
            loss = data_dict['loss'].mean()
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            data_dict.update({'lr':lr_scheduler.get_lr()})
            model.sync_prototype()

            if (idx + 1) % args.print_freq == 0:
                prob_msg = "\t".join([f"{val * 100:.0f}" for val in
                                      list((model.label_stat / (1e-6 + model.label_stat.sum())).data.cpu().numpy())])

                if args.model.name == 'spectral':
                    loss1, loss2, loss3, loss4, loss5 = 0, data_dict["d_dict"]["loss2"].item(), 0, 0, data_dict["d_dict"]["loss5"].item()
                else:
                    loss1, loss2, loss3, loss4, loss5 = data_dict["d_dict"]["loss1"].item(), data_dict["d_dict"]["loss2"].item(), data_dict["d_dict"]["loss3"].item(), data_dict["d_dict"]["loss4"].item(), data_dict["d_dict"]["loss5"].item()
                print('Train: [{0}][{1}/{2}]\t Loss_all {3:.3f} \tc1:{4:.2e}\tc2:{5:.3f}\tc3:{6:.2e}\tc4:{7:.2e}\tc5:{8:.3f}\t{9}'.format(
                    epoch, idx + 1, len(train_label_loader), loss.item(), loss1, loss2, loss3, loss4, loss5, prob_msg
                ))
        print(f"lr: {lr_scheduler.get_lr():.6f}")

        #######################  Evaluation #######################
        model.eval()

        def feat_extract(loader, proto_type, layer='penul'):
            targets = np.array([])
            features = []
            preds = np.array([])
            for idx, (x, labels) in enumerate(loader):
                # feat = model.backbone.features(x.to(device, non_blocking=True))
                ret_dict = model.forward_eval(x.to(device, non_blocking=True), proto_type=proto_type, layer=layer)
                pred = ret_dict['label_pseudo']
                feat = ret_dict['features']
                preds = np.append(preds, pred.cpu().numpy())
                targets = np.append(targets, labels.cpu().numpy())
                features.append(feat.data.cpu().numpy())
            return np.concatenate(features), targets.astype(int), preds


        if (epoch + 1) % args.deep_eval_freq == 0:

            normalizer = lambda x: x / (np.linalg.norm(x, ord=2, axis=-1, keepdims=True) + 1e-10)

            # features_train_l, ltrain_l, _ = feat_extract(train_label_loader, proto_type='known')
            # features_train_u, ltrain_u, _ = feat_extract(train_unlabel_loader, proto_type='novel')
            features_test_l, ltest_l, preds_l = feat_extract(test_label_loader, proto_type='all', layer=args.layer)
            features_test_u, ltest_u, preds_u = feat_extract(test_unlabel_loader, proto_type='all', layer=args.layer)

            ftest_l = normalizer(features_test_l)
            ftest_u = normalizer(features_test_u)

            seen_mask = ltest_u < args.labeled_num
            unseen_mask = ~seen_mask

            #######################  K-Means #######################
            X = ftest_u
            centroids_size = args.num_classes

            if args.layer == 'penul':
                featdim = args.penul_feat_dim
            else:
                featdim = args.proj_feat_dim
            centroids = np.zeros((centroids_size, featdim))
            for li in range(args.labeled_num):
                centroids[li, :] = ftest_l[ltest_l == li].mean(0)

            import numpy.linalg as linalg
            from copy import deepcopy

            X = torch.Tensor(X).cuda()
            centroids = torch.Tensor(centroids).cuda()
            # K-Means++ initialization
            for icls in range(args.labeled_num, centroids_size):
                # distances = torch.norm(X - centroids[:icls, None, :], dim=2)
                distances = torch_l2_dis_batch(X, centroids[:icls, None, :])
                idx_min = torch.argmin(distances, dim=0)
                dist_min = distances[tuple(np.stack((idx_min.data.cpu().numpy(), np.arange(len(idx_min)))))]
                imax = torch.argmax(dist_min) % len(X)
                centroids[icls, :] = X[imax.item(), :]

            # print(f"init done")
            ndata = X.shape[0]
            nfeature = X.shape[1]
            centers_init = centroids
            # Store new centers
            centers = deepcopy(centers_init)

            preds_kmeans = np.zeros(ndata)
            for iter in range(100):
                # print(iter)
                # distances = torch.norm(X - centers[:, None, :], dim=2)
                distances = torch_l2_dis_batch(X, centers[:, None, :])
                preds_kmeans = torch.argmin(distances, dim=0)

                for icls in range(args.labeled_num, args.num_classes):
                    if (preds_kmeans == icls).sum().item() > 0:
                        centers[icls, :] = torch.mean(X[preds_kmeans == icls], dim=0)
                # centers[icls, :] = torch.mean(torch.cat([X[preds_kmeans == icls], torch.Tensor(ftest_l)[ltest_l == icls].to(device)]), dim=0)
                # seen_mask = ltest_u < args.labeled_num
                # unseen_mask = ~seen_mask

            preds = preds_kmeans.data.cpu().numpy().astype(int)
            overall_acc = cluster_acc(preds, ltest_u)
            seen_acc = accuracy(preds[seen_mask], ltest_u[seen_mask])
            unseen_acc = cluster_acc(preds[unseen_mask], ltest_u[unseen_mask])
            unseen_nmi = metrics.normalized_mutual_info_score(ltest_u[unseen_mask], preds[unseen_mask])

            print(f"Seen Acc: {seen_acc:.4f}\t Unseen ACC: {unseen_acc:.4f}\t Overall Acc: {overall_acc:.4f}")
            torch.cuda.empty_cache()

            write_dict = {
                'epoch': epoch,
                'lr': lr_scheduler.get_lr(),
                'overall_acc': overall_acc,
                'seen_acc': seen_acc,
                'unseen_acc': unseen_acc,
                'unseen_nmi': unseen_nmi,
            }

            log_writer.writerow(write_dict)
            log_file.flush()

        #######################  Save Epoch #######################
        if (epoch + 1) % args.log_freq == 0:
            model_path = os.path.join(ckpt_dir, f"{epoch + 1}.pth")
            torch.save({
                'epoch': epoch + 1,
                'state_dict': model.state_dict()
            }, model_path)
            print(f"Model saved to {model_path}")

    # Save checkpoint
    model_path = os.path.join(ckpt_dir, f"latest_{epoch+1}.pth")
    torch.save({
        'epoch': epoch+1,
        'state_dict':model.state_dict()
    }, model_path)
    print(f"Model saved to {model_path}")
    with open(os.path.join(args.log_dir, "checkpoints", f"checkpoint_path.txt"), 'w+') as f:
        f.write(f'{model_path}')

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config-file', default='configs/supspectral_resnet_mlp1000_norelu_cifar100.yaml', type=str)
    # parser.add_argument('-c', '--config-file', default='configs/supspectral_resnet_mlp1000_norelu_cifar10.yaml', type=str)
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--log_freq', type=int, default=10000)
    parser.add_argument('--workers', type=int, default=32)
    parser.add_argument('--test_bs', type=int, default=80)
    parser.add_argument('--download', action='store_true', help="if can't find dataset, download from web")
    parser.add_argument('--data_dir', type=str, default='/home/sunyiyou/workspace/orca/datasets')
    parser.add_argument('--dist_url', type=str, default='tcp://localhost:10001')
    parser.add_argument('--log_dir', type=str, default='./log/')
    parser.add_argument('--ckpt_dir', type=str, default='~/.cache/')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--eval_from', type=str, default=None)
    parser.add_argument('--hide_progress', action='store_true')
    parser.add_argument('--deep_eval_freq', type=int, default=5)
    parser.add_argument('--print_freq', type=int, default=10)
    parser.add_argument('--labeled-num', default=50, type=int)
    parser.add_argument('--labeled-ratio', default=0.5, type=float)

    parser.add_argument('--gamma_l', default=0.0225, type=float)
    parser.add_argument('--gamma_u', default=3, type=float)
    parser.add_argument('--c3_rate', default=3, type=float)
    parser.add_argument('--c4_rate', default=2, type=float)
    parser.add_argument('--c5_rate', default=2, type=float)
    parser.add_argument('--layer', default='proj', type=str)
    parser.add_argument('--proj_feat_dim', default=1000, type=int)
    parser.add_argument('--went', default=0.0, type=float)
    parser.add_argument('--momentum_proto', default=0.9, type=float)
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--base_lr', default=0.05, type=float)
    parser.add_argument('--epochs', default=400, type=float)
    parser.add_argument('--batch_size', default=512, type=int)
    args = parser.parse_args()

    with open(args.config_file, 'r') as f:
        for key, value in Namespace(yaml.load(f, Loader=yaml.FullLoader)).__dict__.items():
            if key not in vars(args):
                vars(args)[key] = value

    args.train.batch_size = args.batch_size
    args.train.stop_at_epoch = int(args.epochs)
    args.train.num_epochs = int(args.epochs)
    args.train.base_lr = args.base_lr

    assert not None in [args.log_dir, args.data_dir, args.ckpt_dir, args.name]

    alpha = args.gamma_l
    beta = args.gamma_u
    scale = 1
    args.c1, args.c2 = 2 * alpha * scale, 2 * beta * scale
    args.c3, args.c4, args.c5 = alpha ** 2 * scale * args.c3_rate, \
                 alpha * beta * scale * args.c4_rate, \
                 beta ** 2 * scale * args.c5_rate

    disc = f"labelnum-{args.labeled_num}-lyr-{args.layer}-c1-{args.c1:.2f}-c2-{args.c2:.1f}-c3-{args.c3:.1e}-c4-{args.c4:.1e}-c5-{args.c5:.1e}-gamma_l-{args.gamma_l:.2f}-gamma_u-{args.gamma_u:.2f}-r345-{args.c3_rate}-{args.c4_rate}-{args.c5_rate}"+ \
           f"-fdim-{args.proj_feat_dim}-went{args.went}-mm{args.momentum_proto}-lr{args.base_lr}-seed{args.seed}"
    args.log_dir = os.path.join(args.log_dir, 'in-progress-'+'{}'.format(date.today())+args.name+'-{}'.format(disc))

    os.makedirs(args.log_dir, exist_ok=True)
    print(f'creating file {args.log_dir}')
    os.makedirs(args.ckpt_dir, exist_ok=True)

    shutil.copy2(args.config_file, args.log_dir)
    set_deterministic(args.seed)

    vars(args)['aug_kwargs'] = {
        'name': args.model.name,
        'image_size': args.dataset.image_size
    }
    vars(args)['dataset_kwargs'] = {
        'dataset':args.dataset.name,
        'data_dir': args.data_dir,
        'download':args.download,
    }
    vars(args)['dataloader_kwargs'] = {
        'drop_last': True,
        'pin_memory': True,
        'num_workers': args.dataset.num_workers,
    }

    log_file = open(os.path.join(args.log_dir, 'log.csv'), mode='w')
    fieldnames = ['epoch', 'lr', 'unseen_acc', 'seen_acc', 'overall_acc', 'unseen_nmi']
    log_writer = csv.DictWriter(log_file, fieldnames=fieldnames)
    log_writer.writeheader()

    return args, log_file, log_writer


if __name__ == "__main__":
    args, log_file, log_writer = get_args()

    main(log_writer, log_file, device=args.device, args=args)

    completed_log_dir = args.log_dir.replace('in-progress', 'debug' if args.debug else 'completed')

    os.rename(args.log_dir, completed_log_dir)
    print(f'Log file has been saved to {completed_log_dir}')