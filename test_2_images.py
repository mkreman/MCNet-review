import numpy as np
import os, time, pprint
import torch
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ["PATH"] = "/usr/local/cuda-11.8/bin:" + os.environ.get("PATH", "")
os.environ["CUDA_HOME"] = "/usr/local/cuda-11.8"
os.environ['LD_LIBRARY_PATH'] = '/usr/local/cuda-11.8/lib64'
print(f"CUDA: {torch.cuda.is_available()}")


import argparse
import warnings
warnings.filterwarnings("ignore")
import datasets
from network import *
from utils import *
from homo_utils import *


def save_outputs(prediction, batch, output_path):
    four_point_org = batch['four_gt'].cpu().numpy()
    top_left = batch['top_left'].squeeze().cpu().numpy()
    bottom_right = batch['bottom_right'].squeeze().cpu().numpy()

    ##! set custom points for img2 here
    four_point_org_ = np.array(
            [[50, top_left[1]], 
            [178, top_left[1]], 
            [178, bottom_right[1]], 
            [50, bottom_right[1]]], dtype=np.float32
        )
    
    four_point_new = prediction + four_point_org_
    
    img1 = batch['img1'].squeeze(0).cpu().numpy()
    img2 = batch['img2'].squeeze(0).cpu().numpy()

    img1_four_points = img1.copy()
    cv2.polylines(img1_four_points, np.int32([four_point_org]), 1, (0,0,255))

    img2_four_points = img2.copy()
    cv2.polylines(img2_four_points, np.int32([four_point_new]), 1, (0,0,255))

    H = cv2.getPerspectiveTransform(four_point_org, four_point_new)
    img_pred = cv2.warpPerspective(img2, H, (img2.shape[1], img2.shape[0]), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    cv2.imwrite(f'{output_path}/img1.jpg', img1)
    cv2.imwrite(f'{output_path}/img2.jpg', img2)
    cv2.imwrite(f'{output_path}/img1_four_points.jpg', img1_four_points)
    cv2.imwrite(f'{output_path}/img2_four_points.jpg', img2_four_points)
    cv2.imwrite(f'{output_path}/img_pred.jpg', img_pred)


def test(args, homo_model=None):
    if torch.cuda.is_available():
        device = torch.device('cuda:'+ str(args.gpuid))
    else:
        device = 'cpu'
    test_loader = datasets.fetch_dataloader(args, split="test")
    if homo_model == None:
        homo_model = MCNet(args).to(device)
        if args.checkpoint is None:
            print("ERROR : no checkpoint")
            exit()
        state = torch.load(args.checkpoint, map_location='cpu')
        homo_model.load_state_dict(state['homo_model'])
        print("test with pretrained model")
    homo_model.eval()

    with torch.no_grad():
        for data_batch in test_loader:
            for key, value in data_batch.items(): 
                data_batch[key] = value.to(device)
            pred_h4p_12 = homo_model(data_batch)
            print(f"Prediction: {pred_h4p_12[-1]}")
    
    save_outputs(pred_h4p_12[-1].cpu().numpy(), data_batch, args.log_full_dir)
    print("Results saved to", args.log_full_dir)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='test', help='Train or test', choices=['train', 'test'])
    parser.add_argument('--gpuid', type=int, default=0)
    parser.add_argument('--note', type=str, default='', help='experiment notes')
    parser.add_argument('--dataset', type=str, default='custom', help='dataset')

    parser.add_argument('--img1', type=str, default='Photos-002/20250402_135410.jpg', help='path to image 1')
    parser.add_argument('--img2', type=str, default='Photos-002/20250402_135412.jpg', help='path to image 2')
    parser.add_argument('--output_path', type=str, default='/home/mayank.mds2023/CV/MCNet/output', help='Path to the folder of output images')

    parser.add_argument('--log_dir', type=str, default='logs', help='The log path')
    parser.add_argument('--nolog', action='store_false', default=True, help='save log file or not')
    parser.add_argument('--checkpoint', type=str, default="./weight/mscoco.pth", help='Test model name')
    parser.add_argument('--batch_size', type=int, default=1) 
    parser.add_argument('--print_freq', type=int, default=100)
    parser.add_argument('--val_freq', type=int, default=10000)
    parser.add_argument('--seed', type=int, default=512)
    parser.add_argument('--num_steps', type=int, default=120000)
    parser.add_argument('--lr', type=float, default=4e-4, help='Max learning rate')
    parser.add_argument('--log_full_dir', type=str)
    parser.add_argument('--epsilon', type=float, default=0.1, help='loss parameter')
    parser.add_argument('--loss', type=str, default="speedup", help="speedup or l1 or l2 or speedupl1")
    parser.add_argument('--downsample', type=int, nargs='+', default=[4,2,1])
    parser.add_argument('--iter', type=int, nargs='+', default=[2,2,2])
    parser.add_argument('--speed_threshold', type=float, default=1, help='use speed-up when L1 < x')
    args = parser.parse_args()
    
    if args.nolog:
        args.log_full_dir = os.path.join(args.log_dir, time.strftime("%Y-%m-%d-%H_%M_%S", time.localtime()) + "_" + args.dataset + "_" + args.note)
        if not os.path.exists(args.log_full_dir): os.makedirs(args.log_full_dir)
        sys.stdout = Logger_(os.path.join(args.log_full_dir, f'record.log'), sys.stdout)
    pprint.pprint(vars(args))
    
    seed_everything(args.seed)
   
    test(args)

if __name__ == "__main__":
    main()
