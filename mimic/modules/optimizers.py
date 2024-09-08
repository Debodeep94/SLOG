import torch


def build_optimizer(args, model):
    ve_params = list(map(id, model.visual_extractor.parameters()))
    ed_params = filter(lambda x: id(x) not in ve_params, model.parameters())
    #regression_params = surrogate_model.parameters()
    optimizer = getattr(torch.optim, args.optim)(
        [{'params': model.visual_extractor.parameters(), 'lr': args.lr_ve},
         {'params': ed_params, 'lr': args.lr_ed}],
        weight_decay=args.weight_decay,
        amsgrad=args.amsgrad
    )
    #print(optimizer)
    return optimizer


def build_lr_scheduler(args, optimizer):
    lr_scheduler = getattr(torch.optim.lr_scheduler, args.lr_scheduler)(optimizer, args.step_size, args.gamma)
    return lr_scheduler
