from UGATIT import UGATIT
import argparse
from utils import *

"""parsing and configuration"""

def parse_args():
     # Popis programu
    desc = "Pytorch implementation of U-GAT-IT"
    parser = argparse.ArgumentParser(description=desc)
    
    # # Fáze trénování nebo testování
    parser.add_argument('--phase', type=str, default='train', help='[train / test]')
    
    # Verze modelu U-GAT-IT (plná verze nebo lehká verze)
    parser.add_argument('--light', type=str2bool, default=False, help='[U-GAT-IT full version / U-GAT-IT light version]')
    
    #  Název datové sady
    parser.add_argument('--dataset', type=str, default='training', help='dataset_name')

    # Počet iterací trénování
    parser.add_argument('--iteration', type=int, default=5000, help='The number of training iterations')
    
    # Velikost dávky dat
    parser.add_argument('--batch_size', type=int, default=1, help='The size of batch size')
    
    # Frekvence tisku obrázků během trénování
    parser.add_argument('--print_freq', type=int, default=1000, help='The number of image print freq')
    
    # Frekvence ukládání modelů během trénování
    parser.add_argument('--save_freq', type=int, default=100000, help='The number of model save freq')
    
    # Příznak poklesu rychlosti učení
    parser.add_argument('--decay_flag', type=str2bool, default=True, help='The decay_flag')

    # Rychlost učení
    parser.add_argument('--lr', type=float, default=0.0001, help='The learning rate')
    
    # Váha pro adversariální ztrátu
    parser.add_argument('--weight_decay', type=float, default=0.0001, help='The weight decay')
    
    # Váha pro adversariální ztrátu
    parser.add_argument('--adv_weight', type=int, default=100, help='Weight for GAN')
    
    # # Váha pro cyklickou konsistenci
    parser.add_argument('--cycle_weight', type=int, default=10, help='Weight for Cycle')
    
    # Weight for identity loss
    parser.add_argument('--identity_weight', type=int, default=10, help='Weight for Identity')
    
    # Weight for Class Activation Mapping (CAM) loss
    parser.add_argument('--cam_weight', type=int, default=100, help='Weight for CAM')

    # Base channel number per layer
    parser.add_argument('--ch', type=int, default=64, help='base channel number per layer')
    
    # Number of residual blocks
    parser.add_argument('--n_res', type=int, default=4, help='The number of resblock')
    
    # Number of discriminator layers
    parser.add_argument('--n_dis', type=int, default=6, help='The number of discriminator layer')

    # Size of image
    parser.add_argument('--img_size', type=int, default=256, help='The size of image')
    
    # Number of image channels
    parser.add_argument('--img_ch', type=int, default=3, help='The size of image channel')

    # Directory name to save the results
    parser.add_argument('--result_dir', type=str, default='results', help='Directory name to save the results')
    
    # Device mode (CPU or CUDA)
    parser.add_argument('--device', type=str, default='cuda', choices=['cpu', 'cuda'], help='Set gpu mode; [cpu, cuda]')
    
    # Benchmark flag
    parser.add_argument('--benchmark_flag', type=str2bool, default=False)
    
    # Resume training flag
    parser.add_argument('--resume', type=str2bool, default=False)

    return check_args(parser.parse_args())

"""checking arguments"""
def check_args(args):
    # --result_dir
    check_folder(os.path.join(args.result_dir, args.dataset, 'model'))
    check_folder(os.path.join(args.result_dir, args.dataset, 'img'))
    check_folder(os.path.join(args.result_dir, args.dataset, 'test'))

    # --epoch
    try:
        assert args.epoch >= 1
    except:
        print('number of epochs must be larger than or equal to one')

    # --batch_size
    try:
        assert args.batch_size >= 1
    except:
        print('batch size must be larger than or equal to one')
    return args



"""main"""
def main():
    # parse arguments
    args = parse_args()
    
    if args is None:
      exit()

    # open session
    gan = UGATIT(args)

    # build graph
    gan.build_model()

    if args.phase == 'train' :
        gan.train()
        print(" [*] Training finished!")

    if args.phase == 'test':
        gan.test()
        print(" [*] Test finished!")

if __name__ == '__main__':
    main()
