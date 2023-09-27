import argparse
from config import cfg
import torch
from base import Trainer
import torch.backends.cudnn as cudnn

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=str, dest='gpu_ids')
    parser.add_argument('--continue', dest='continue_train', action='store_true')
    args = parser.parse_args()

    if not args.gpu_ids:
        assert 0, "Please set proper gpu ids"

    if '-' in args.gpu_ids:
        gpus = args.gpu_ids.split('-')
        gpus[0] = int(gpus[0])
        gpus[1] = int(gpus[1]) + 1
        args.gpu_ids = ','.join(map(lambda x: str(x), list(range(*gpus))))

    return args

def main():
    
    # argument parse and create log
    args = parse_args()
    cfg.set_args(args.gpu_ids, args.continue_train)
    cudnn.fastest = True
    cudnn.benchmark = True

    trainer = Trainer()
    trainer._make_batch_generator()
    trainer._make_model()

    # train
    for epoch in range(trainer.start_epoch, cfg.end_epoch):
        
        trainer.tot_timer.tic()
        trainer.read_timer.tic()

        for itr, (input_img, joint_img, joint_vis, joints_have_depth) in enumerate(trainer.batch_generator):
            step = itr/len(trainer.batch_generator)+epoch
            trainer.set_lr_v2(step)
            trainer.read_timer.toc()
            trainer.gpu_timer.tic()

            # forward
            trainer.optimizer.zero_grad()
            target = {'coord': joint_img, 'vis': joint_vis, 'have_depth': joints_have_depth}
            loss_coord = trainer.model(input_img, target)
            loss_coord = loss_coord.mean()

            # backward
            loss = loss_coord
            loss.backward()
            trainer.optimizer.step()
            trainer.optimizer.zero_grad(set_to_none=True)

            
            trainer.gpu_timer.toc()
            screen = [
                'Epoch %d/%d itr %d/%d:' % (epoch, cfg.end_epoch, itr, trainer.itr_per_epoch),
                'lr: %g' % (trainer.get_lr()),
                'speed: %.2f(%.2fs r%.2f)s/itr' % (
                    trainer.tot_timer.average_time, trainer.gpu_timer.average_time, trainer.read_timer.average_time),
                '%.2fh/epoch' % (trainer.tot_timer.average_time / 3600. * trainer.itr_per_epoch),
                '%s: %.4f' % ('loss_coord', loss_coord.detach()),
                ]
            trainer.logger.info(' '.join(screen))
            trainer.tot_timer.toc()
            trainer.tot_timer.tic()
            trainer.read_timer.tic()

        if (epoch+1)%cfg.save_interval == 0:
            trainer.save_model({
                'epoch': epoch,
                'network': trainer.model.state_dict(),
                'optimizer': trainer.optimizer.state_dict(),
            }, epoch)
        

if __name__ == "__main__":
    main()
