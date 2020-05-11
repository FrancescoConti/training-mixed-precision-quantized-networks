import argparse
import PIL
import os
import time
import logging
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import models
from torch.autograd import Variable
from data import get_dataset, get_num_classes
from preprocess import get_transform
from utils import *
from datetime import datetime
from ast import literal_eval
import json
from torchvision.utils import save_image
import quantization 
from quantization.quant_auto import memory_driven_quant
from tqdm import tqdm
import nemo
import warnings
import math
import copy
import collections 

# filter out ImageNet EXIF warnings
warnings.filterwarnings("ignore", "(Possibly )?corrupt EXIF data", UserWarning)
warnings.filterwarnings("ignore", "Metadata Warning", UserWarning)

model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch ConvNet Training')

parser.add_argument('--results_dir', metavar='RESULTS_DIR', default='./results',
                    help='results dir')
parser.add_argument('--save', metavar='SAVE', default='',
                    help='saved folder')
parser.add_argument('--dataset', metavar='DATASET', default='cifar10',
                    help='dataset name or folder')
parser.add_argument('--model', '-a', metavar='MODEL', default='vgg_cifar10_binary',
                    choices=model_names,
                    help='model architecture: ' +
                    ' | '.join(model_names) +
                    ' (default: alexnet)')
parser.add_argument('--input_size', type=int, default=None,
                    help='image input size')
parser.add_argument('--model_config', default='',
                    help='additional architecture configuration')
parser.add_argument('--type', default='torch.cuda.FloatTensor',
                    help='type of tensor - e.g torch.cuda.HalfTensor')
parser.add_argument('--gpus', default='0,1,2,3',
                    help='gpus used for training - e.g 0,1,3')
parser.add_argument('-j', '--workers', default=8, type=int, metavar='N',
                    help='number of data loading workers (default: 8)')
parser.add_argument('--epochs', default=150, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('--optimizer', default='SGD', type=str, metavar='OPT',
                    help='optimizer function used')
parser.add_argument('--lr', '--learning_rate', default=1e-4, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--print-freq', '-p', default=100, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default=None, type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', action='store_true',
                    help='run model on validation set')
parser.add_argument('--save_check', action='store_true',
                    help='saving the checkpoint')
parser.add_argument('--terminal', action='store_true')
# quantization parameters
parser.add_argument('--quantize', action='store_true',
                    help='quantize the network')
parser.add_argument('--type_quant', default=None,
                    help='Type of binarization process')
parser.add_argument('--weight_bits', default=1,
                    help='Number of bits for the weights')
parser.add_argument('--activ_bits', default=1,
                    help='Number of bits for the activations')

parser.add_argument('--initial_folding', default=False, action='store_true',
                    help='Fold BNs into Linear layers before training')
parser.add_argument('--initial_equalization', default=False, action='store_true',
                    help='Perform Linear layer weight equalization before training')
parser.add_argument('--quant_add_config', default='', type=str, 
                    help='Additional config of per-layer quantization')

# mobilenet params
parser.add_argument('--mobilenet_width', default=1.0, type=float,
                    help='Mobilenet Width Muliplier')
parser.add_argument('--mobilenet_input', default=224, type=int,
                    help='Mobilenet input resolution ')

# mixed-precision params
parser.add_argument('--mem_constraint', default='', type=str,
                    help='Memory constraints for automatic bitwidth quantization')
parser.add_argument('--mixed_prec_quant', default='MixPL', type=str, 
                    help='Type of quantization for mixed-precision low bitwidth: MixPL | MixPC')
parser.add_argument('--mixed_prec_dict', default=None, type=str)


def main():
    global args, best_prec1
    best_prec1 = 0
    args = parser.parse_args()
    
    weight_bits = int(args.weight_bits)
    activ_bits = int(args.activ_bits)


    if args.save is '':
        args.save = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    save_path = os.path.join(args.results_dir, args.save)
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    setup_logging(os.path.join(save_path, 'log.txt'))
    results_file = os.path.join(save_path, 'results.%s')
    results = ResultsLog(results_file % 'csv', results_file % 'html')

    logging.info("saving to %s", save_path)
    logging.debug("run arguments: %s", args)

    if 'cuda' in args.type:
        args.gpus = [int(i) for i in args.gpus.split(',')]
        print('Selected GPUs: ', args.gpus)
        torch.cuda.set_device(args.gpus[0])
        cudnn.benchmark = True
    else:
        args.gpus = None

    # create model
    logging.info("creating model %s", args.model)
    model = models.__dict__[args.model]
    nClasses = get_num_classes(args.dataset)
    model_config = {'input_size': args.input_size, 'dataset': args.dataset, 'num_classes': nClasses, \
                    'width_mult': float(args.mobilenet_width), 'input_dim': float(args.mobilenet_input) }

    if args.model_config is not '':
        model_config = dict(model_config, **literal_eval(args.model_config))

    model = model(**model_config)
    logging.info("created model with configuration: %s", model_config)
    print(model)


    num_parameters = sum([l.nelement() for l in model.parameters()])
    logging.info("number of parameters: %d", num_parameters)

    # Data loading code
    default_transform = {
        'train': get_transform(args.dataset,
                               input_size=args.input_size, augment=True),
        'eval': get_transform(args.dataset,
                              input_size=args.input_size, augment=False)
    }
    transform = getattr(model, 'input_transform', default_transform)
    regime = getattr(model, 'regime', {0: {'optimizer': args.optimizer,
                                           'lr': args.lr,
                                           'momentum': args.momentum,
                                           'weight_decay': args.weight_decay}})
    print(transform)
    # define loss function (criterion) and optimizer
    criterion = getattr(model, 'criterion', nn.CrossEntropyLoss)()    
    criterion.type(args.type)


    val_data = get_dataset(args.dataset, 'val', transform['eval'])
    val_loader = torch.utils.data.DataLoader(
        val_data,
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    train_data = get_dataset(args.dataset, 'train', transform['train'])
    train_loader = torch.utils.data.DataLoader(
        train_data,
        batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True)

    #define optimizer
    params_dict = dict(model.named_parameters())
    params = []
    for key, value in params_dict.items():
        if 'alpha' in key or 'beta' in key:
            params += [{'params':value, 'weight_decay': 1e-4}]
        else:
            params += [{'params':value, 'weight_decay': 1e-5}]

    mixed_prec_dict = None
    if args.mixed_prec_dict is not None:
        mixed_prec_dict = nemo.utils.precision_dict_from_json(args.mixed_prec_dict)
        print("Load mixed precision dict from outside")
    elif args.mem_constraint is not '':
        mem_contraints = json.loads(args.mem_constraint)
        print('This is the memory constraint:', mem_contraints )
        if mem_contraints is not None:
            x_test = torch.Tensor(1,3,args.mobilenet_input,args.mobilenet_input)
            mixed_prec_dict = memory_driven_quant(model, x_test, mem_contraints[0], mem_contraints[1], args.mixed_prec_quant)
    
    #multi gpus
    if args.gpus and len(args.gpus) > 1:
        model = torch.nn.DataParallel(model).cuda()
    else:
        model.type(args.type)

    mobilenet_width = float(args.mobilenet_width)
    mobilenet_width_s = args.mobilenet_width
    mobilenet_input = int(args.mobilenet_input) 

    # if args.resume is None:
    #     val_loss, val_prec1, val_prec5 = validate(val_loader, model, criterion, 0, None)
    #     print("[NEMO] Full-precision model: top-1=%.2f top-5=%.2f" % (val_prec1, val_prec5))   

    if args.quantize:

        # transform the model in a NEMO FakeQuantized representation
        model = nemo.transform.quantize_pact(model, dummy_input=torch.randn((1,3,mobilenet_input,mobilenet_input)).to('cuda'))
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)

        if args.resume is not None:
            checkpoint_file = args.resume
            if os.path.isfile(checkpoint_file):
                logging.info("loading checkpoint '%s'", args.resume)
                checkpoint_loaded = torch.load(checkpoint_file)
                checkpoint = checkpoint_loaded['state_dict']
                model.load_state_dict(checkpoint, strict=True)
                prec_dict = checkpoint_loaded.get('precision')
            else:
                logging.error("no checkpoint found at '%s'", args.resume)
                import sys; sys.exit(1)

        if args.resume is None:
            print("[NEMO] Model calibration")
            model.change_precision(bits=20)
            model.reset_alpha_weights()
            
            if args.initial_folding:
                model.fold_bn()
                # use DFQ for weight equalization
                if args.initial_equalization:
                    model.equalize_weights_dfq()
            elif args.initial_equalization:
                model.equalize_weights_lsq(verbose=True)
                model.reset_alpha_weights()
#                model.reset_alpha_weights(use_method='dyn_range', dyn_range_cutoff=0.05, verbose=True)

            # calibrate after equalization
            with model.statistics_act():
                val_loss, val_prec1, val_prec5 = validate(val_loader, model, criterion, 0, None)

            # # use this in place of the usual calibration, because PACT_Act's descend from ReLU6 and
            # # the trained weights already assume the presence of a clipping effect
            # # this should be integrated in NEMO by saving the "origin" of the PACT_Act!
            # for i in range(0,27):
            #     model.model[i][3].alpha.data[:] = min(model.model[i][3].alpha.item(), model.model[i][3].max)

            val_loss, val_prec1, val_prec5 = validate(val_loader, model, criterion, 0, None)
 
            print("[NEMO] 20-bit calibrated model: top-1=%.2f top-5=%.2f" % (val_prec1, val_prec5))   
            nemo.utils.save_checkpoint(model, optimizer, 0, acc=val_prec1, checkpoint_name='mobilenet_%s_%d_calibrated' % (mobilenet_width_s, mobilenet_input), checkpoint_suffix='')

            model.change_precision(bits=activ_bits)
            model.change_precision(bits=weight_bits, scale_activations=False)

        else:
            print("[NEMO] Not calibrating model, as it is pretrained")
            model.change_precision(bits=1, min_prec_dict=prec_dict)

            ### val_loss, val_prec1, val_prec5 = validate(val_loader, model, criterion, 0, None) 
            ### print("[NEMO] pretrained model: top-1=%.2f top-5=%.2f" % (val_prec1, val_prec5))

        if mixed_prec_dict is not None:
            mixed_prec_dict_all = model.export_precision()
            for k in mixed_prec_dict.keys():
                mixed_prec_dict_all[k] = mixed_prec_dict[k]
            model.change_precision(bits=1, min_prec_dict=mixed_prec_dict_all)

            # freeze and quantize BN parameters
            # nemo.transform.bn_quantizer(model, precision=nemo.precision.Precision(bits=20))
            # model.freeze_bn()
            # model.fold_bn()
            # model.equalize_weights_dfq(verbose=True)
            val_loss, val_prec1, val_prec5 = validate(val_loader, model, criterion, 0, None)

#            print("[NEMO] Rounding weights")
#            model.round_weights()

    if args.terminal:
        fqs = copy.deepcopy(model.state_dict())
        model.freeze_bn(reset_stats=True, disable_grad=True)
        bin_fq, bout_fq, _ = nemo.utils.get_intermediate_activations(model, validate, val_loader, model, criterion, 0, None, shorten=1)

        torch.save({'in': bin_fq['model.0.0'][0]}, "input_fq.pth")

        val_loss, val_prec1, val_prec5 = validate(val_loader, model, criterion, 0, None)
        print("[NEMO] FQ model: top-1=%.2f top-5=%.2f" % (val_prec1, val_prec5))
       
        input_bias_dict  = {} #{'model.0.0' : +1.0, 'model.0.1' : +1.0}
        remove_bias_dict = {} #{'model.0.1' : 'model.0.2'}
        input_bias       = 0. #math.floor(1.0 / (2./255)) * (2./255)

        model.qd_stage(eps_in=2./255, add_input_bias_dict=input_bias_dict, remove_bias_dict=remove_bias_dict, precision=nemo.precision.Precision(bits=20), int_accurate=True)
        # model.round_weights()
        # model.harden_weights()
        # fix ConstantPad2d
#        model.model[0][0].value = input_bias

#        val_loss, val_prec1, val_prec5 = validate(val_loader, model, criterion, 0, None, input_bias=input_bias)
#        print("[NEMO] QD model: top-1=%.2f top-5=%.2f" % (val_prec1, val_prec5))

        qds = copy.deepcopy(model.state_dict())
        bin_qd, bout_qd, _ = nemo.utils.get_intermediate_activations(model, validate, val_loader, model, criterion, 0, None, input_bias=input_bias, shorten=1)
        import IPython; IPython.embed()

        torch.save({'qds': qds, 'fqs': fqs}, "states.pth")
        torch.save({'in': bin_qd['model.0.0'][0]}, "input_qd.pth")

        diff = collections.OrderedDict()
        for k in bout_fq.keys():
            diff[k] = (bout_fq[k] - bout_qd[k]).to('cpu').abs()

        for i in range(0,26):
            for j in range(3,4):
                k  = 'model.%d.%d' % (i,j)
                kn = 'model.%d.%d' % (i if j<3 else i+1, j+1 if j<3 else 0)
                eps = model.get_eps_at(kn, eps_in=2./255)[0]
                print("%s:" % k)
                idx = diff[k]>eps
                n = idx.sum()
                t = (diff[k]>-1e9).sum()
                max_eps = torch.ceil(diff[k].max() / model.get_eps_at('model.%d.0' % (i+1), 2./255)[0]).item()
                mean_eps = torch.ceil(diff[k][idx].mean() / model.get_eps_at('model.%d.0' % (i+1), 2./255)[0]).item()
                try:
                    print("  max:   %.3f (%d eps)" % (diff[k].max().item(), max_eps))
                    print("  mean:  %.3f (%d eps) (only diff. elements)" % (diff[k][idx].mean().item(), mean_eps))
                    print("  #diff: %d/%d (%.1f%%)" % (n, t, float(n)/float(t)*100)) 
                except ValueError:
                    print("  #diff: 0/%d (0%%)" % (t,)) 

        model.id_stage()
        # fix ConstantPad2d
        model.model[0][0].value = input_bias / (2./255)

        ids = model.state_dict()
        bin_id, bout_id, _ = nemo.utils.get_intermediate_activations(model, validate, val_loader, model, criterion, 0, None, input_bias=input_bias, shorten=1, eps_in=2./255) 

        torch.save({'in': bin_fq['model.0.0'][0]}, "input_id.pth")

        diff = collections.OrderedDict()
        for i in range(0,26):
            for j in range(3,4):
                k  = 'model.%d.%d' % (i,j)
                kn = 'model.%d.%d' % (i if j<3 else i+1, j+1 if j<3 else 0)
                eps = model.get_eps_at(kn, eps_in=2./255)[0]
                diff[k] = (bout_id[k]*eps - bout_qd[k]).to('cpu').abs()
                print("%s:" % k)
                idx = diff[k]>=eps
                n = idx.sum()
                t = (diff[k]>-1e9).sum()
                max_eps  = torch.ceil(diff[k].max() / eps).item()
                mean_eps = torch.ceil(diff[k][idx].mean() / eps).item()
                try:
                    print("  max:   %.3f (%d eps)" % (diff[k].max().item(), max_eps))
                    print("  mean:  %.3f (%d eps) (only diff. elements)" % (diff[k][idx].mean().item(), mean_eps))
                    print("  #diff: %d/%d (%.1f%%)" % (n, t, float(n)/float(t)*100)) 
                except ValueError:
                    print("  #diff: 0/%d (0%%)" % (t,)) 

        import IPython; IPython.embed()

        val_loss, val_prec1, val_prec5 = validate(val_loader, model, criterion, 0, None, input_bias=input_bias, eps_in=2./255)
        print("[NEMO] ID model: top-1=%.2f top-5=%.2f" % (val_prec1, val_prec5))

        import IPython; IPython.embed()
        import sys; sys.exit(0)

    for epoch in range(args.start_epoch, args.epochs):
#        optimizer = adjust_optimizer(optimizer, epoch, regime)
        
        # train for one epoch
        train_loss, train_prec1, train_prec5 = train(train_loader, model, criterion, epoch, optimizer, freeze_bn=True if epoch>0 else False, absorb_bn=True if epoch==0 else False)
        val_loss, val_prec1, val_prec5 = validate(val_loader, model, criterion, epoch)

        # remember best prec@1 and save checkpoint
        is_best = val_prec1 > best_prec1
        best_prec1 = max(val_prec1, best_prec1)
          
        #save_model
        if args.save_check:
            nemo.utils.save_checkpoint(model, optimizer, 0, acc=val_prec1, checkpoint_name='mobilenet_%s_%d%s_checkpoint' % (mobilenet_width_s, mobilenet_input, "_mixed" if mixed_prec_dict is not None else ""), checkpoint_suffix='')

        if is_best:
            nemo.utils.save_checkpoint(model, optimizer, 0, acc=val_prec1, checkpoint_name='mobilenet_%s_%d%s_best' % (mobilenet_width_s, mobilenet_input, "_mixed" if mixed_prec_dict is not None else ""), checkpoint_suffix='')

        logging.info('\n Epoch: {0}\t'
                     'Training Loss {train_loss:.4f} \t'
                     'Training Prec@1 {train_prec1:.3f} \t'
                     'Training Prec@5 {train_prec5:.3f} \t'
                     'Validation Loss {val_loss:.4f} \t'
                     'Validation Prec@1 {val_prec1:.3f} \t'
                     'Validation Prec@5 {val_prec5:.3f} \t'
                     .format(epoch + 1, train_loss=train_loss, val_loss=val_loss,
                             train_prec1=train_prec1, val_prec1=val_prec1,
                             train_prec5=train_prec5, val_prec5=val_prec5))

        results.add(epoch=epoch + 1, train_loss=train_loss, val_loss=val_loss,
                    train_error1=100 - train_prec1, val_error1=100 - val_prec1,
                    train_error5=100 - train_prec5, val_error5=100 - val_prec5)
        results.save()

def forward(data_loader, model, criterion, epoch=0, training=True, optimizer=None, quantizer=None, verbose=True, input_bias=0.0, eps_in=None, integer=False, shorten=None):

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    end = time.time()

    # apply transofrms at the begininng of each epoch
    print('Training: ',training )

    # input quantization
    if eps_in is None:
        scale_factor = 1.
        div_factor   = 1.
    elif not integer:
        scale_factor = 1./eps_in
        div_factor   = 1./eps_in
    else:
        scale_factor = 1./eps_in
        div_factor   = 1.

    if shorten is not None:
        length = shorten
    else:
        length = len(data_loader)
    
    with tqdm(total=length,
          desc='Epoch     #{}'.format(epoch),
          disable=not verbose) as t:
        for i,(inputs,target) in enumerate(data_loader):
            # measure data loading time
            if i==length:
                break
            data_time.update(time.time() - end)
            if args.gpus is not None:
#                inputs = inputs.cuda(async=True)
                target = target.cuda(async=True)
    
            with torch.no_grad():
                if eps_in is None:
                    input_var = (inputs.to('cuda') + input_bias)
                else:
                    # input_var = torch.floor((inputs.to('cuda') + input_bias) * scale_factor) / scale_factor
                    input_var = (inputs.to('cuda') + input_bias) * scale_factor
                    # input_var = torch.floor((inputs.to('cuda') + input_bias) * 255 / 2.)

                target_var = target
    
            # compute output
            output = model(input_var)
    
            loss = criterion(output, target_var)
            if type(output) is list:
                output = output[0]
    
            # measure accuracy and record loss
            prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
            losses.update(loss.data.item(), inputs.size(0))
            top1.update(prec1.item(), inputs.size(0))
            top5.update(prec5.item(), inputs.size(0))
    
            if training:
                # compute gradient and do SGD step
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
    
            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
    
            t.set_postfix({'loss': losses.avg, 'top1': top1.avg, 'top5': top5.avg})
            t.update(1)

    return losses.avg, top1.avg, top5.avg


def train(data_loader, model, criterion, epoch, optimizer, quantizer=None, freeze_bn=True, absorb_bn=False, shorten=None):
    
    # switch to train mode
    model.train()
    if freeze_bn or absorb_bn:
        if absorb_bn:
            print("Freezing BN statistics, but not disabling BN trained parameter gradients")
        else:
            print("Freezing BN statistics and disabling BN trained parameter gradients")
        model.freeze_bn(reset_stats=True, disable_grad=freeze_bn and not absorb_bn)
    return forward(data_loader, model, criterion, epoch,
                   training=True, optimizer=optimizer, quantizer=quantizer, shorten=shorten)


def validate(data_loader, model, criterion, epoch, quantizer=None, input_bias=0.0, eps_in=None, integer=False, shorten=None):
    
    # switch to evaluate mode
    model.eval()
    return forward(data_loader, model, criterion, epoch,
                   training=False, optimizer=None, quantizer=quantizer, input_bias=input_bias, eps_in=eps_in, shorten=shorten)

if __name__ == '__main__':
    main()
