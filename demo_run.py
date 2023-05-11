from tools.utils import *
from tools.config import Param
from tools.datatools.detection_dataloader import Data as DetData
from tools.datatools.discover_dataloader import Data as DisData
from pipline.exe_pipline import run_pipline

def run(args):
    args.logger.info(args.method)
    args.logger.info('Data Preparation...')
    args.logger.info('task type: {}'.format(args.task_type))
    time.sleep(20)
    if args.task_type in ['relation_detection']:
        data = DetData(args)
    elif args.task_type in ['relation_discover']:
        data = DisData(args)

    manager = get_manager(args, data)
    
    if args.train_model:
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
    else:
        run_pipline()

            
            
