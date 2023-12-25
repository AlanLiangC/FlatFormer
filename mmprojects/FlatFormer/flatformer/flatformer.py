from typing import Tuple

from torch import Tensor

from mmdet3d.models.detectors import VoxelNet
from mmdet3d.registry import MODELS
from mmdet3d.utils import ConfigType, OptConfigType, OptMultiConfig
# from mmdet3d.models.voxel_encoders import DynamicVFE

@MODELS.register_module()
class FlatFormer(VoxelNet):
    def __init__(self, 
                 voxel_encoder: ConfigType, 
                 middle_encoder: ConfigType, 
                 backbone: ConfigType, 
                 neck: OptConfigType = None, 
                 bbox_head: OptConfigType = None, 
                 train_cfg: OptConfigType = None, 
                 test_cfg: OptConfigType = None, 
                 data_preprocessor: OptConfigType = None, 
                 init_cfg: OptMultiConfig = None) -> None:
        
        super().__init__(voxel_encoder, 
                         middle_encoder, 
                         backbone, 
                         neck, bbox_head, 
                         train_cfg, 
                         test_cfg, 
                         data_preprocessor, 
                         init_cfg)
        
    def extract_feat(self, batch_inputs_dict: dict) -> Tuple[Tensor]:
        """Extract features from points."""
        voxel_dict = batch_inputs_dict['voxels']
        voxel_features, voxel_coors = self.voxel_encoder(voxel_dict['voxels'],
                                            voxel_dict['coors'])
        batch_size = voxel_dict['coors'][-1, 0].item() + 1
        x = self.middle_encoder(voxel_features, voxel_coors,
                                batch_size)
        x = self.backbone(x)
        if self.with_neck:
            x = self.neck(x)
        return x