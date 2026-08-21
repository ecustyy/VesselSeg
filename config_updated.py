import argparse

def parse_args():
    parser = argparse.ArgumentParser()

    # in/out
    parser.add_argument('--outf', default='./experiments',
                        help='trained model will be saved at here')
    parser.add_argument('--save', default='DirectionAware_UNet_vessel_seg',
                        help='save name of experiment in args.outf directory')

    # data
    parser.add_argument('--train_data_path_list',
                        default='./prepare_dataset/data_path_list/DRIVE/train.txt')
    parser.add_argument('--test_data_path_list',
                        default='./prepare_dataset/data_path_list/DRIVE/test.txt')
    parser.add_argument('--train_patch_height', default=224)
    parser.add_argument('--train_patch_width', default=224)
    parser.add_argument('--N_patches', default=1500,
                        help='Number of training image patches')
    parser.add_argument('--inside_FOV', default='center',
                        help='Choose from [not,center,all]')
    parser.add_argument('--val_ratio', default=0.1,
                        help='The ratio of the validation set in the training set')
    parser.add_argument('--sample_visualization', default=True,
                        help='Visualization of training samples')
    
    # model parameters
    parser.add_argument('--in_channels', default=1, type=int,
                        help='input channels of model')
    parser.add_argument('--classes', default=2, type=int, 
                        help='output channels of model')

    # training
    parser.add_argument('--N_epochs', default=50, type=int,
                        help='number of total epochs to run')
    parser.add_argument('--batch_size', default=16,
                        type=int, help='batch size')
    parser.add_argument('--early-stop', default=6, type=int,
                        help='early stopping')
    parser.add_argument('--lr', default=0.0005, type=float,
                        help='initial learning rate')
    parser.add_argument('--val_on_test', default=False, type=bool,
                        help='Validation on testset')

    # for pre_trained checkpoint
    parser.add_argument('--start_epoch', default=1, 
                        help='Start epoch')
    parser.add_argument('--pre_trained', default=None,
                        help='(path of trained _model)load trained model to continue train')

    # testing
    parser.add_argument('--test_patch_height', default=224)
    parser.add_argument('--test_patch_width', default=224)
    parser.add_argument('--stride_height', default=16)
    parser.add_argument('--stride_width', default=16)

    # hardware setting
    parser.add_argument('--cuda', default=True, type=bool,
                        help='Use GPU calculating')

    # TransUNet
    parser.add_argument('--vit_name', type=str,
                        default='R50-ViT-B_16', help='select one vit model')
    parser.add_argument('--img_size', type=int,
                        default=224, help='input patch size of network input')
    parser.add_argument('--vit_patches_size', type=int,
                        default=16, help='vit_patches_size, default is 16')

    # ========== Direction-Aware Innovations ==========
    
    # Direction consistency loss
    parser.add_argument('--use_direction_loss', action='store_true',
                        help='Use direction consistency loss')
    parser.add_argument('--direction_loss_weight', default=0.1, type=float,
                        help='Weight for direction consistency loss')
    parser.add_argument('--direction_alpha', default=1.0, type=float,
                        help='Weight for horizontal gradient in direction loss')
    parser.add_argument('--direction_beta', default=1.0, type=float,
                        help='Weight for vertical gradient in direction loss')
    parser.add_argument('--direction_gamma', default=0.5, type=float,
                        help='Weight for diagonal gradient in direction loss')
    
    # Direction-aware attention
    parser.add_argument('--use_direction_attention', action='store_true',
                        help='Use direction-aware attention mechanism')
    parser.add_argument('--attention_type', default='channel', type=str,
                        choices=['spatial', 'channel', 'transformer'],
                        help='Type of direction-aware attention')
    parser.add_argument('--attention_reduction', default=16, type=int,
                        help='Reduction ratio for channel attention')
    parser.add_argument('--num_directions', default=4, type=int,
                        help='Number of directional branches')
    
    # Hybrid Mamba-Transformer parameters
    parser.add_argument('--use_hybrid', action='store_true',
                        help='Use hybrid Mamba-Transformer architecture')
    parser.add_argument('--hybrid_fusion_type', default='parallel', type=str,
                        choices=['parallel', 'sequential', 'gated'],
                        help='Fusion type for hybrid architecture')
    parser.add_argument('--hybrid_d_model', default=256, type=int,
                        help='Model dimension for hybrid architecture')
    parser.add_argument('--hybrid_num_layers', default=2, type=int,
                        help='Number of hybrid layers')
    parser.add_argument('--hybrid_num_heads', default=8, type=int,
                        help='Number of attention heads')
    parser.add_argument('--hybrid_d_state', default=16, type=int,
                        help='Mamba state dimension')
    
    # Combined loss weights
    parser.add_argument('--ce_weight', default=1.0, type=float,
                        help='Weight for cross-entropy loss')
    parser.add_argument('--focal_loss_gamma', default=2.0, type=float,
                        help='Gamma parameter for focal loss (if used)')
    
    # Data augmentation
    parser.add_argument('--use_data_aug', default=True, type=bool,
                        help='Use data augmentation')
    parser.add_argument('--aug_flip_lr', default=True, type=bool,
                        help='Use left-right flip augmentation')
    parser.add_argument('--aug_flip_ud', default=True, type=bool,
                        help='Use up-down flip augmentation')
    parser.add_argument('--aug_rotate', default=True, type=bool,
                        help='Use rotation augmentation')
    parser.add_argument('--aug_rotate_limit', default=15, type=int,
                        help='Maximum rotation angle in degrees')
    
    # Advanced training options
    # Note: Using store_true to properly handle boolean flags
    parser.add_argument('--use_mixed_precision', action='store_true',
                        help='Use mixed precision training (AMP)')
    parser.add_argument('--gradient_clip', default=1.0, type=float,
                        help='Gradient clipping value (set to 0 to disable)')
    parser.add_argument('--label_smoothing', default=0.0, type=float,
                        help='Label smoothing factor')
    
    # Logging and visualization
    parser.add_argument('--log_interval', default=10, type=int,
                        help='Log every N batches')
    parser.add_argument('--visualize_interval', default=5, type=int,
                        help='Visualize predictions every N epochs')
    parser.add_argument('--save_best_only', default=True, type=bool,
                        help='Save only the best model')

    args = parser.parse_args()

    return args


def get_config_dict(args):
    """
    Convert argparse namespace to dictionary for easy serialization.
    
    Args:
        args: ArgumentParser namespace
        
    Returns:
        dict: Configuration dictionary
    """
    config = vars(args).copy()
    
    # Add dataset path (can be overridden)
    config['dataset_path'] = '/media/yy/03b6a5be-c6a9-43f3-a09e-1b55cb38185f/yy/task11/DRIVE'
    
    # Add innovation flags
    config['innovations'] = {
        'direction_consistency_loss': args.use_direction_loss,
        'direction_aware_attention': args.use_direction_attention,
        'attention_type': args.attention_type,
    }
    
    return config


if __name__ == '__main__':
    args = parse_args()
    config = get_config_dict(args)
    
    print("Configuration:")
    print("=" * 60)
    for key, value in sorted(config.items()):
        print(f"{key:30s}: {value}")
    print("=" * 60)
