# Copyright (c) Facebook, Inc. and its affiliates.
import argparse
import glob
import multiprocessing as mp
import numpy as np
import os
import tempfile
import time
import warnings
import cv2
import tqdm
import json
from itertools import groupby
import requests

from detectron2.config import get_cfg
from detectron2.data.detection_utils import read_image
from detectron2.utils.logger import setup_logger

from predictor import VisualizationDemo

# constants
WINDOW_NAME = "COCO detections"


def setup_cfg(args):
    # load config from file and command-line arguments
    cfg = get_cfg()
    # To use demo for Panoptic-DeepLab, please uncomment the following two lines.
    # from detectron2.projects.panoptic_deeplab import add_panoptic_deeplab_config  # noqa
    # add_panoptic_deeplab_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    # Set score_threshold for builtin models
    cfg.MODEL.RETINANET.SCORE_THRESH_TEST = args.confidence_threshold
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = args.confidence_threshold
    cfg.MODEL.PANOPTIC_FPN.COMBINE.INSTANCES_CONFIDENCE_THRESH = args.confidence_threshold
    cfg.freeze()
    return cfg


def get_parser():
    parser = argparse.ArgumentParser(description="Detectron2 demo for builtin configs")
    parser.add_argument(
        "--config-file",
        default="configs/quick_schedules/mask_rcnn_R_50_FPN_inference_acc_test.yaml",
        metavar="FILE",
        help="path to config file",
    )
    parser.add_argument("--webcam", action="store_true", help="Take inputs from webcam.")
    parser.add_argument("--video-input", help="Path to video file.")
    parser.add_argument(
        "--input",
        nargs="+",
        default=["/home/jayaram/research/research_tracks/multibody_slam/instance_segmentation_detectron2/detectron2/demo_images/"],
        help="A list of space separated input images; "
        "or a single glob pattern such as 'directory/*.jpg'",
    )
    parser.add_argument(
        "--output",
        help="A file or directory to save output visualizations. "
        "If not given, will show output in an OpenCV window.",
    )

    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.5,
        help="Minimum score for instance predictions to be shown",
    )
    parser.add_argument(
        "--opts",
        help="Modify config options using the command-line 'KEY VALUE' pairs",
        default=[],
        nargs=argparse.REMAINDER,
    )
    return parser


def test_opencv_video_format(codec, file_ext):
    with tempfile.TemporaryDirectory(prefix="video_format_test") as dir:
        filename = os.path.join(dir, "test_file" + file_ext)
        writer = cv2.VideoWriter(
            filename=filename,
            fourcc=cv2.VideoWriter_fourcc(*codec),
            fps=float(30),
            frameSize=(10, 10),
            isColor=True,
        )
        [writer.write(np.zeros((10, 10, 3), np.uint8)) for _ in range(30)]
        writer.release()
        if os.path.isfile(filename):
            return True
        return False

def binary_mask_to_rle(binary_mask):
    rle = {'counts': [], 'size': list(binary_mask.shape)}
    counts = rle.get('counts')
    for i, (value, elements) in enumerate(groupby(binary_mask.ravel(order='F'))):
        if i == 0 and value == 1:
            counts.append(0)
        counts.append(len(list(elements)))
    return rle

def download_image(url, save_path):
    response = requests.get(url)
    if response.status_code == 200:
        with open(save_path, 'wb') as file:
            file.write(response.content)
        print(f"Image downloaded successfully: {save_path}")
    else:
        print(f"Failed to download image: {url}")

def extract_masks(img, image_name, file, annotations_folder_path, masks_info_folder_path):
    start_time = time.time()
    predictions, visualized_output = demo.run_on_image(img)
    logger.info(
        "{}: {} in {:.2f}s".format(
            file, # path,
            "detected {} instances".format(len(predictions["instances"]))
            if "instances" in predictions
            else "finished",
            time.time() - start_time,
        )
    )

    file_path = os.path.join(annotations_folder_path, image_name + ".txt")  # Specify the path to the output file
    with open(file_path, "w") as file:
        # Write content to the file
        coco_annotations = {
            "annotations": [],
            # "images": [],
            # "categories": []
        }

        file.write("num instances: " + str(len(predictions["instances"])) + "\n")
        for i in range(len(predictions["instances"])):
            mask = predictions["instances"][i]._fields['pred_masks'].squeeze().cpu().numpy()
            # ones_indices = np.argwhere(mask == 1)
            # print('first and last indices where elements are 1: {}, {}'.format(ones_indices[0], ones_indices[-1]))

            file.write("COCO class for " + str(i) + " instance: " + str(predictions["instances"][i]._fields['pred_classes'].item()) + "\n")
            file.write("score for " + str(i) + " instance: " + str(predictions["instances"][i]._fields['scores'].item()) + "\n")
            rle_mask = binary_mask_to_rle(mask)

            # Creating annotation entry
            annotation = {
                "id": i + 1,  # Unique identifier for each mask
                # "image_id": 1,  # Unique identifier for the corresponding image
                # "category_id": 1,  # Unique identifier for the category
                "segmentation": rle_mask,
                "area": int(mask.sum().item())  # Area of the mask
            }
            
            coco_annotations["annotations"].append(annotation)       

            file.write("**************************************************************************************************************" + "\n")

    # Save the annotations to a JSON file
    json_file_path = os.path.join(masks_info_folder_path, image_name + ".json")  # Specify the path to the output file
    with open(json_file_path, 'w') as json_file:
        json.dump(coco_annotations, json_file)

if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    args = get_parser().parse_args()
    setup_logger(name="fvcore")
    logger = setup_logger()
    logger.info("Arguments: " + str(args))

    cfg = setup_cfg(args)

    demo = VisualizationDemo(cfg)

    if args.input:
        if len(args.input) == 1:
            args.input = glob.glob(os.path.expanduser(args.input[0]))
            assert args.input, "The input path(s) was not found"

        annotations_folder_path = os.path.join(args.input[0], "annotations_folder")
        if not os.path.exists(annotations_folder_path):
            os.makedirs(annotations_folder_path)

        masks_info_folder_path = os.path.join(args.input[0], "masks_info_folder")
        if not os.path.exists(masks_info_folder_path):
            os.makedirs(masks_info_folder_path)

        n_image = 1
        # Loop through all files in the input dir
        base_url = 'http://platformpgh.cs.cmu.edu/live_stream/carfusion/Morewood/0/'     #change manually while running this script to extract masks
        corrupted_images = []
        for file in os.listdir(args.input[0]):
            if file.lower().endswith(".jpg") or file.lower().endswith(".jpeg"):
            # for path in tqdm.tqdm(args.input, disable=not args.output):
                # use PIL, to be consistent with evaluation
                num_tries = 0
                image_name = file.lower().split('.')[0]
                print('image name: {}'.format(image_name))
                try: 
                    if((image_name.isdigit() == False) or (image_name.isdigit() and (int(image_name)%5 != 0))):
                        continue
                    
                    # image_url = base_url + file
                    # save_path = os.path.join(args.input[0], file)
                    # download_image(image_url, save_path)
                    img = read_image(args.input[0] + file, format="BGR")    #exception occurs here mostly   (Note that / should be there at the end of --input arg)
                    # if(image_name.isdigit() == False):

                    extract_masks(img, image_name, file, annotations_folder_path, masks_info_folder_path)

                except (IOError, OSError) as e:
                    # Exception handling code
                    if(num_tries == 0):
                        corrupted_images.append(image_name)   #this will log only corrupted images which are multiples of 5
                        print(f"Error occurred while reading '{image_name}': {e}")

                    num_tries = num_tries + 1
                    continue  # Continue to the next iteration
                    
                    
            if args.output:
                if os.path.isdir(args.output):
                    assert os.path.isdir(args.output), args.output
                    out_filename = os.path.join(args.output, os.path.basename(file))
                else:
                    assert len(args.input) == 1, "Please specify a directory with args.output"
                    out_filename = args.output
                # visualized_output.save(out_filename)
                # else:
                #     cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
                #     cv2.imshow(WINDOW_NAME, visualized_output.get_image()[:, :, ::-1])
                #     if cv2.waitKey(0) == 27:
                #         break  # esc to quit

            n_image += 1

        corrupted_file_path = os.path.join(args.input[0], "corrupted_images.txt")
        with open(corrupted_file_path, "w") as file:
            # Write corrupted images to a text file
            for img in corrupted_images:
                file.write(img + "\n")


    elif args.webcam:
        assert args.input is None, "Cannot have both --input and --webcam!"
        assert args.output is None, "output not yet supported with --webcam!"
        cam = cv2.VideoCapture(0)
        for vis in tqdm.tqdm(demo.run_on_video(cam)):
            cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
            cv2.imshow(WINDOW_NAME, vis)
            if cv2.waitKey(1) == 27:
                break  # esc to quit
        cam.release()
        cv2.destroyAllWindows()
    elif args.video_input:
        video = cv2.VideoCapture(args.video_input)
        width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frames_per_second = video.get(cv2.CAP_PROP_FPS)
        num_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        basename = os.path.basename(args.video_input)
        codec, file_ext = (
            ("x264", ".mkv") if test_opencv_video_format("x264", ".mkv") else ("mp4v", ".mp4")
        )
        if codec == ".mp4v":
            warnings.warn("x264 codec not available, switching to mp4v")
        if args.output:
            if os.path.isdir(args.output):
                output_fname = os.path.join(args.output, basename)
                output_fname = os.path.splitext(output_fname)[0] + file_ext
            else:
                output_fname = args.output
            assert not os.path.isfile(output_fname), output_fname
            output_file = cv2.VideoWriter(
                filename=output_fname,
                # some installation of opencv may not support x264 (due to its license),
                # you can try other format (e.g. MPEG)
                fourcc=cv2.VideoWriter_fourcc(*codec),
                fps=float(frames_per_second),
                frameSize=(width, height),
                isColor=True,
            )
        assert os.path.isfile(args.video_input)
        for vis_frame in tqdm.tqdm(demo.run_on_video(video), total=num_frames):
            if args.output:
                output_file.write(vis_frame)
            else:
                cv2.namedWindow(basename, cv2.WINDOW_NORMAL)
                cv2.imshow(basename, vis_frame)
                if cv2.waitKey(1) == 27:
                    break  # esc to quit
        video.release()
        if args.output:
            output_file.release()
        else:
            cv2.destroyAllWindows()
