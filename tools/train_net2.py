"""
The script is based on https://github.com/facebookresearch/detectron2/blob/master/tools/train_net.py. 
"""

import logging
import os
import json
from collections import OrderedDict
import glob

import pprint
import sys

# sys.path.append('/content/drive/MyDrive/PROJECT/201_HaMaruki/201_32_Layout_parser')
# sys.path.append('/content/drive/MyDrive/PROJECT/201_HaMaruki/201_32_Layout_parser/detectron2')
# sys.path.append('/content/drive/MyDrive/PROJECT/201_HaMaruki/201_32_Layout_parser/detectron2/detectron2')

# sys.path.append(os.path.join(os.path.dirname(__file__), '../../'))

print(">>>>>>> train_net >>>>>>")
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(os.path.join(os.path.dirname(__file__), '.'))
pprint.pprint(sys.path)

import detectron2.utils.comm as comm
import detectron2.data.transforms as T
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.data import DatasetMapper, build_detection_train_loader

from detectron2.data.datasets import register_coco_instances

from engine import (
    DefaultTrainer,
    DefaultPredictor,
    default_argument_parser,
    default_setup,
    hooks,
    launch,
)
from dtt2.evaluation import (
    COCOEvaluator,
    verify_results,
)
from dtt2.utils.visualizer import Visualizer


from detectron2.modeling import GeneralizedRCNNWithTTA
import pandas as pd
import cv2

def get_augs(cfg):
    """Add all the desired augmentations here. A list of availble augmentations
    can be found here:
       https://detectron2.readthedocs.io/en/latest/modules/data_transforms.html
    """
    augs = [
        T.ResizeShortestEdge(
            cfg.INPUT.MIN_SIZE_TRAIN,
            cfg.INPUT.MAX_SIZE_TRAIN,
            cfg.INPUT.MIN_SIZE_TRAIN_SAMPLING,
        )
    ]
    if cfg.INPUT.CROP.ENABLED:
        augs.append(
            T.RandomCrop_CategoryAreaConstraint(
                cfg.INPUT.CROP.TYPE,
                cfg.INPUT.CROP.SIZE,
                cfg.INPUT.CROP.SINGLE_CATEGORY_MAX_AREA,
                cfg.MODEL.SEM_SEG_HEAD.IGNORE_VALUE,
            )
        )
    horizontal_flip: bool = cfg.INPUT.RANDOM_FLIP == "horizontal"
    # augs.append(T.RandomFlip(horizontal=horizontal_flip, vertical=not horizontal_flip))
    # Rotate the image between -90 to 0 degrees clockwise around the centre
    # augs.append(T.RandomRotation(angle=[-90.0, 0.0]))
    return augs


class Trainer(DefaultTrainer):
    """
    We use the "DefaultTrainer" which contains pre-defined default logic for
    standard training workflow. They may not work for you, especially if you
    are working on a new research project. In that case you can use the cleaner
    "SimpleTrainer", or write your own training loop. You can use
    "tools/plain_train_net.py" as an example.

    Adapted from:
        https://github.com/facebookresearch/detectron2/blob/master/projects/DeepLab/train_net.py
    """

    @classmethod
    def build_train_loader(cls, cfg):
        mapper = DatasetMapper(cfg, is_train=True, augmentations=get_augs(cfg))
        return build_detection_train_loader(cfg, mapper=mapper)

    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        """
        Returns:
            DatasetEvaluator or None

        It is not implemented by default.
        """
        return COCOEvaluator(dataset_name, cfg, True, output_folder)

    @classmethod
    def test_with_TTA(cls, cfg, model):
        logger = logging.getLogger("detectron2.trainer")
        # In the end of training, run an evaluation with TTA
        # Only support some R-CNN models.
        logger.info("Running inference with test-time augmentation ...")
        model = GeneralizedRCNNWithTTA(cfg, model)
        evaluators = [
            cls.build_evaluator(
                cfg, name, output_folder=os.path.join(cfg.OUTPUT_DIR, "inference_TTA")
            )
            for name in cfg.DATASETS.TEST
        ]
        res = cls.test(cfg, model, evaluators)
        res = OrderedDict({k + "_TTA": v for k, v in res.items()})
        return res

    @classmethod
    def eval_and_save(cls, cfg, model):
        evaluators = [
            cls.build_evaluator(
                cfg, name, output_folder=os.path.join(cfg.OUTPUT_DIR, "inference")
            )
            for name in cfg.DATASETS.TEST
        ]
        res = cls.test(cfg, model, evaluators)
        pd.DataFrame(res).to_csv(os.path.join(cfg.OUTPUT_DIR, "eval.csv"))
        return res


def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()

    if args.config_file != "":
        cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)

    with open(args.json_annotation_train, "r") as fp:
        anno_file = json.load(fp)

    cfg.MODEL.ROI_HEADS.NUM_CLASSES = len(anno_file["categories"])
    del anno_file

    cfg.DATASETS.TRAIN = (f"{args.dataset_name}-train",)
    cfg.DATASETS.TEST = (f"{args.dataset_name}-val",)
    #cfg.MODEL.WEIGHTS = r"/content/drive/MyDrive/CANE/output/PRImA/fast_rcnn_R_50_FPN_3x/012/model_0000999.pth"
    
    
    num_gpu = 1
    bs = (num_gpu * 2)
    #cfg.SOLVER.BASE_LR = 0.02 * bs / 16  # pick a good LR
    #cfg.SOLVER.BASE_LR = 0.001
    
    # cfg.freeze()
    default_setup(cfg, args)
    return cfg


def main(args):
    # Register Datasets
    register_coco_instances(
        f"{args.dataset_name}-train",
        {},
        args.json_annotation_train,
        args.image_path_train,
    )

    register_coco_instances(
        f"{args.dataset_name}-val", 
        {}, 
        args.json_annotation_val, 
        args.image_path_val
    )
    cfg = setup(args)
    
    #args.resume = False
    
    
    print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>")
    
    print("args.resume          : {}".format(args.resume))
    print("cfg.MODEL.WEIGHTS    : {}".format(cfg.MODEL.WEIGHTS))
    print("cfg.OUTPUT_DIR       : {}".format(cfg.OUTPUT_DIR))
    print("cfg.NUM_CLASSES      : {}".format(cfg.MODEL.ROI_HEADS.NUM_CLASSES))
    print("cfg.SOLVER.BASE_LR   : {}".format(cfg.SOLVER.BASE_LR))
    
    

    if args.eval_only:
        
        
        
        
        pth_list = sorted(glob.glob(cfg.OUTPUT_DIR + "/*.pth"))
        print("===============================================")
        print("pth_list")
        pprint.pprint(pth_list)
        for pth_path in pth_list:
            print(">>>>>>>>>>>>>>>>>>>>>>>>>>")
            print(pth_path)
            # ------------------------------------------
            # init
            #
            cfg.MODEL.WEIGHTS = pth_path
            model = Trainer.build_model(cfg)
            DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
                pth_path, resume=args.resume
            )
            res = Trainer.test(cfg, model)

            # ------------------------------------------
            # check point dir
            #
            Checkpoint_name = pth_path.split("/")[-1].split(".")[0]
            Checkpoint_dir = cfg.OUTPUT_DIR + "/Checkpoint"
            os.makedirs(Checkpoint_dir, exist_ok=True)

            # ------------------------------------------
            # Aug
            #
            if cfg.TEST.AUG.ENABLED:
                res.update(Trainer.test_with_TTA(cfg, model))
            if comm.is_main_process():
                verify_results(cfg, res)

            # Save the evaluation results
            pd.DataFrame(res).to_csv(f"{cfg.OUTPUT_DIR}/eval.csv")
            
            # ------------------------------------------
            # DefaultPredictor
            #
            predictor = DefaultPredictor(cfg)
            print("Predictor has been initialized.")
            image = cv2.imread("downloaded-annotations_mini/images/information_extraction_from_te-3_6774.jpg")
            pred_result = predictor(image)

            # ------------------------------------------
            # Visual
            #
            v = Visualizer(image)
            out = v.draw_instance_predictions(pred_result['instances'])
            im = out.get_image()[:, :, ::-1]
            visual_save_path = Checkpoint_dir + '/im_{}_pred.png'.format(Checkpoint_name)
            print("visual_save_path : {}".format(visual_save_path))
            cv2.imwrite(visual_save_path, im)
            
            
            # ------------------------------------------
            # backbone
            #
            # model = predictor.get_model()
            # features = model.backbone(image)
            # features_keys = features.keys()
            # print(">>>> features_keys : {}".format(features_keys))
        
        
        
        
    else:

        # Ensure that the Output directory exists
        os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

        """
        If you'd like to do anything fancier than the standard training logic,
        consider writing your own training loop (see plain_train_net.py) or
        subclassing the trainer.
        """
        trainer = Trainer(cfg)
        print("resume_or_load ........")
        trainer.resume_or_load(resume=args.resume)
        trainer.register_hooks(
            [hooks.EvalHook(0, lambda: trainer.eval_and_save(cfg, trainer.model))]
        )
        if cfg.TEST.AUG.ENABLED:
            trainer.register_hooks(
                [hooks.EvalHook(0, lambda: trainer.test_with_TTA(cfg, trainer.model))]
            )
        return trainer.train()


if __name__ == "__main__":
    parser = default_argument_parser()

    # Extra Configurations for dataset names and paths
    parser.add_argument(
        "--dataset_name", 
        help="The Dataset Name"
    )
    parser.add_argument(
        "--json_annotation_train",
        help="The path to the training set JSON annotation",
    )
    parser.add_argument(
        "--image_path_train",
        help="The path to the training set image folder",
    )
    parser.add_argument(
        "--json_annotation_val",
        help="The path to the validation set JSON annotation",
    )
    parser.add_argument(
        "--image_path_val",
        help="The path to the validation set image folder",
    )
    parser.add_argument(
        "--model_path",
        help="The path  set model path",
    )

    args = parser.parse_args()
    print("Command Line Args:", args)

    # Dataset Registration is moved to the main function to support multi-gpu training
    # See ref https://github.com/facebookresearch/detectron2/issues/253#issuecomment-554216517

    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )
