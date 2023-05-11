from tools.utils import *
from tools.config import Param
from tools.datatools.detection_dataloader import Data as DetData
from tools.datatools.discover_dataloader import Data as DisData
from pipline.exe_pipline import run_pipline

def run(args, go_test=False):
    args.logger.info(args.method)
    args.logger.info('Data Preparation...')
    args.logger.info('task type: {}'.format(args.task_type))
    
    if args.task_type in ['relation_detection']:
        data = DetData(args)
    elif args.task_type in ['relation_discovery']:
        data = DisData(args)

    manager = get_manager(args, data)
    
    if args.train_model:
        if not go_test:
            args.logger.info('Training Begin...')
            manager.train(args, data)
            args.logger.info('Training Finished...')
    args.logger.info('Evaluation begin...')
    # args.load_ckpt = '/home/sharing/disk1/zk/weight_cache/relation_discover/MORE/ckpt_semeval--0-2e-05.pth'
    manager.restore_model(args)
    manager.eval(args, data, is_test=True)
    
if __name__ == '__main__':
    is_pipe = False
    if not is_pipe:
        print('Parameters Initialization...')
        args = Param()
        go_test = True
        # if args.dataname in ["semeval"] and args.seed in [0] and args.known_cls_ratio in [0.25]:
        #     go_test = True
        torch.cuda.set_device(args.gpu_id)
        run(args, go_test)
    else:

        run_pipline()

            
            
