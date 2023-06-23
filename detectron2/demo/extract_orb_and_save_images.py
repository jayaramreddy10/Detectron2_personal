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
import random
import matplotlib.pyplot as plt

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

# util functions 
def plot_images(img_set, n_r, n_c, img_titles):
    fig = plt.figure(figsize = (19, 19))
    cnt = 0
    for i in range(n_r):
        for j in range(n_c):
            if cnt == len(img_set):
                break
            ax1 = fig.add_subplot(n_r, n_c, cnt + 1)
            ax1.imshow(img_set[cnt], cmap = 'gray')
            ax1.set_title(img_titles[cnt], fontsize = 15)
            cnt = cnt + 1
    plt.show() 

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

def extract_and_save_feature_matches(img_1, img_2):

    # step1: get corresponding points using SIFT
    minHessian = 400
    sift = cv2.SIFT_create()
    kps_1, descriptors_1 = sift.detectAndCompute(img_1, None)
    kps_2, descriptors_2 = sift.detectAndCompute(img_2, None)

    #FLANN matcher
    # -- Step 2: Matching descriptor vectors with a FLANN based matcher
    # Since SURF is a floating-point descriptor NORM_L2 is used
    matcher = cv2.DescriptorMatcher_create(cv2.DescriptorMatcher_FLANNBASED)
    knn_matches = matcher.knnMatch(descriptors_1, descriptors_2, 2)
    #-- Filter matches using the Lowe's ratio test
    ratio_thresh = 0.7
    good_matches = []
    for m,n in knn_matches:
        if m.distance < ratio_thresh * n.distance:
            good_matches.append(m)
    n_good_matches = len(good_matches)
    good_matches = random.sample(good_matches, n_good_matches//4)

    # -- Draw matches
    img_matches = np.empty((max(img_1.shape[0], img_2.shape[0]), img_1.shape[1]+img_2.shape[1], 3), dtype=np.uint8)
    img_3 = cv2.drawMatches(img_1, kps_1, img_2, kps_2, good_matches[:], img_matches, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

    #FLANN matcher -- version 2 for ORB
    # -- Step 2: Matching descriptor vectors with a FLANN based matcher
    # Since SURF is a floating-point descriptor NORM_L2 is used
    # index_params = dict(algorithm=6,
    #                         table_number=6,
    #                         key_size=12,
    #                         multi_probe_level=2)
    # search_params = {}
    # flann = cv2.FlannBasedMatcher(index_params, search_params)
    # knn_matches = flann.knnMatch(descriptors_1, descriptors_2, k=2)
    # #-- Filter matches using the Lowe's ratio test
    # ratio_thresh = 0.7
    # good_matches = []
    # for m,n in knn_matches:
    #     if m.distance < ratio_thresh * n.distance:
    #         good_matches.append(m)
    # n_good_matches = len(good_matches)
    # good_matches = random.sample(good_matches, n_good_matches//4)

    # # -- Draw matches
    # img_matches = np.empty((max(img_1.shape[0], img_2.shape[0]), img_1.shape[1]+img_2.shape[1], 3), dtype=np.uint8)
    # img_3 = cv2.drawMatches(img_1, kps_1, img_2, kps_2, good_matches[:], img_matches, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

    #bf MATCHER

    # fm = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)   #feature matching in cv2
    # matches = fm.match(descriptors_1,descriptors_2)
    # matches = sorted(matches, key = lambda x:x.distance)
    # img_3 = cv2.drawMatches(img_1, kps_1, img_2, kps_2, matches, img_2, flags=2)
    # #matches[:300]

    # Save the image to a file
    output_file = 'output_image.jpg'  # Replace 'output_image.jpg' with your desired file name
    cv2.imwrite(output_file, img_3)

    # plt.imshow(img_3)
    # img_set = []
    # img_set.append(img_1)
    # img_set.append(img_2)
    # img_set.append(img_3)
    # plot_images(img_set, 3, 1, ['image 1','image 2', 'image with feature matching'])

    #select best n feature matches
    n_matches = 50
    selected_matches = good_matches[:n_matches]

    # every match is cv2.DMatch object and has attributes like distance, queryIdx, trainIdx etc
    X1_list = [kps_1[match.queryIdx].pt for match in selected_matches] 
    X2_list = [kps_2[match.trainIdx].pt for match in selected_matches]
    # Save the matches to a file
    matches_file = 'matches.txt'  # Replace 'matches.txt' with your desired file name

    with open(matches_file, 'w') as f:
        for match in selected_matches:
            f.write(f"Distance: {match.distance}, Keypoint 1 idx: {match.queryIdx}, Keypoint 2 idx: {match.trainIdx}\n")

    X1_list_all = [kps_1[match.queryIdx].pt for match in good_matches] 
    X2_list_all = [kps_2[match.trainIdx].pt for match in good_matches]
    # print('no of correspondences used for estimating F matrix: {}'.format(len(X1_list)))
    print('no of correspondences detected by SIFT in total: {}'.format(len(X1_list_all)))

    assert len(X1_list) == len(X2_list), "no of features are not matching"


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

        orb_features_folder_path = os.path.join(args.input[0], "orb_features_folder")
        if not os.path.exists(orb_features_folder_path):
            os.makedirs(orb_features_folder_path)

        n_image = 1
        # Loop through all files in the input dir
        base_url = 'http://platformpgh.cs.cmu.edu/live_stream/carfusion/Morewood/0/'     #change manually while running this script to extract masks
        corrupted_images = []

        masks_json_directory = os.path.join(args.input[0], "masks_info_folder")
        image_directory = args.input[0]  # Replace with the directory path containing the images
        image_extension = ".jpg"  # Specify the file extension for the images

        # Get a list of image filenames in the directory
        masks_files = [f for f in os.listdir(masks_json_directory) if f.endswith('.json')]
        # image_files = [f for f in os.listdir(image_directory) if f.endswith('.jpg')]

        # Sort the image filenames numerically
        masks_files.sort(key=lambda x: int(x.split('.')[0]))
        # print('masks_files.{}'.format(masks_files))

        image_files = [file_name.replace(".json", ".jpg") for file_name in masks_files]
        # print('image_files.{}'.format(image_files))

        # Iterate over consecutive pairs of images
        for i in range(len(image_files) - 1):
            # Generate the file paths for the current pair of images
            image1_path = os.path.join(image_directory, image_files[i])
            image2_path = os.path.join(image_directory, image_files[i + 1])

            #image names with multiples of 5 have already been filtered in masks_json_folder (so no need to check here)

            # Load the images
            image1 = cv2.imread(image1_path)
            image2 = cv2.imread(image2_path)

            print('extracting feats and saving for images:{}, {}'.format(image_files[i], image_files[i+1]))
            extract_and_save_feature_matches(image1, image2)
            # if args.output:
            #     if os.path.isdir(args.output):
            #         assert os.path.isdir(args.output), args.output
            #         out_filename = os.path.join(args.output, os.path.basename(file))
            #     else:
            #         assert len(args.input) == 1, "Please specify a directory with args.output"
            #         out_filename = args.output
                # visualized_output.save(out_filename)
                # else:
                #     cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
                #     cv2.imshow(WINDOW_NAME, visualized_output.get_image()[:, :, ::-1])
                #     if cv2.waitKey(0) == 27:
                #         break  # esc to quit

            n_image += 1

        # corrupted_file_path = os.path.join(args.input[0], "corrupted_images.txt")
        # with open(corrupted_file_path, "w") as file:
        #     # Write corrupted images to a text file
        #     for img in corrupted_images:
        #         file.write(img + "\n")


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
