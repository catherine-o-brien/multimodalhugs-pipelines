#!/usr/bin/env python3

import os
import argparse
import importlib
import logging
import csv
import itertools
import cv2

import tensorflow as tf
import tensorflow_datasets as tfds
import sign_language_datasets.datasets

import numpy as np

from pose_format.utils.reader import BufferReader
from pose_format.pose import Pose, PoseHeader
from pose_format.numpy import NumPyPoseBody
from sign_language_datasets.datasets.config import SignDatasetConfig
from pose_format.pose_header import PoseHeader, PoseHeaderComponent, PoseHeaderDimensions
from pose_format.utils.openpose import hand_colors  

from typing import Iterator, Optional, Dict, List

import mediapipe as mp
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False,
                       max_num_hands=2,
                       min_detection_confidence=0.5,
                       min_tracking_confidence=0.5)


# Parse command-line arguments
def parse_arguments():
    parser = argparse.ArgumentParser(description="Process and transform a TSV file.")
    parser.add_argument("--pose-dir", type=str, help="Where to save poses.")
    parser.add_argument("--output-dir", type=str, help="Path to the output TSV files.")
    parser.add_argument("--encoder-prompt", type=str, default="__dgs__", help="encoder prompt string.")
    parser.add_argument("--decoder-prompt", type=str, default="__de__", help="decoder prompt string.")

    parser.add_argument("--tfds-data-dir", type=str, default=None,
                        help="TFDS data folder to cache downloads.", required=False)
    parser.add_argument("--dry-run", action="store_true", default=False,
                        help="Process very few elements only.", required=False)
    return parser.parse_args()


def load_pose_header(dataset) -> PoseHeader:
    # Take one example from the train split to get frame shape
    example = next(iter(tfds.as_numpy(dataset["train"].take(1))))
    first_frame = example["video"][0]
    frame_width, frame_height = first_frame.shape[:2]
    HAND_POINTS = mp_hands.HandLandmark._member_names_            # 21 landmark names
    HAND_LIMBS  = [(int(a), int(b)) for a, b in mp_hands.HAND_CONNECTIONS]  # edges

    left = PoseHeaderComponent(
        name="LEFT_HAND_LANDMARKS",
        points=HAND_POINTS,                  # list[str], length 21
        limbs=HAND_LIMBS,                    # list[(u16,u16)]
        colors=hand_colors,                  # list[(r,g,b)] same length as limbs
        point_format="XYZC"                  # "XYZC" like pose-format holistic utils
    )
    right = PoseHeaderComponent(
        name="RIGHT_HAND_LANDMARKS",
        points=HAND_POINTS,
        limbs=HAND_LIMBS,
        colors=hand_colors,
        point_format="XYZC"
    )

    dims = PoseHeaderDimensions(width=frame_width, height=frame_height, depth=0)
    header = PoseHeader(version=0.1, dimensions=dims, components=[left, right])
    return header


def load_dataset(data_dir: Optional[str] = None):
    """
    :param data_dir:
    :return:
    """

    config = SignDatasetConfig(name="rwth_phoenix2014_t_with_video",
                               version="3.0.0",
                               include_video=True,
                               process_video=False)

    rwth_phoenix2014_t = tfds.load('rwth_phoenix2014_t', builder_kwargs=dict(config=config), data_dir=data_dir)

    return rwth_phoenix2014_t


Example = Dict[str, str]

def generate_examples(dataset: tf.data.Dataset,
                      split_name: str,
                      pose_header: PoseHeader,
                      pose_dir: str,
                      dry_run: bool = False) -> Iterator[Example]:
    """
    :param dataset:
    :param split_name: "train", "validation" or "test"
    :param pose_header:
    :param pose_dir:
    :param dry_run:
    :return:
    """

    if dry_run:
        print("\nDry run!\n")
        data_iterator = itertools.islice(dataset[split_name], 0, 10)
    else:
        data_iterator = dataset[split_name]

    for datum in data_iterator:

        datum_id = datum["id"].numpy().decode('utf-8')

        text = datum['text'].numpy().decode('utf-8')

        pose_data, pose_confidence, fps = generate_pose_data(datum)

        pose_body = NumPyPoseBody(fps=fps,
                            data=pose_data,
                            confidence=pose_confidence)

        # Construct Pose object and write to file
        pose = Pose(pose_header, pose_body)

        pose_filepath = os.path.join(pose_dir, f"{datum_id}.pose")

        if dry_run:
            logging.debug(f"Writing pose to: '{pose_filepath}'")

        with open(pose_filepath, "wb") as data_buffer:
            pose.write(data_buffer)

        example = {
            "datum_id": datum_id,
            "text": text,
            "pose_filepath": pose_filepath
        }

        yield example

def generate_pose_data(datum):
    pose_data = []
    pose_confidence = []
    fps = int(0)

    video = datum["video"].numpy()  # shape: (frames, height, width, 3)
    for frame in video: 
        pose, conf = mediapipe_hands_pose(frame)
        pose_data.append(pose)
        pose_confidence.append(conf)

    pose_data = np.stack(pose_data, axis=0).astype(np.float32)[:, np.newaxis, :, :]
    pose_confidence = np.stack(pose_confidence, axis=0).astype(np.float32)[:, np.newaxis, :]

    print("Completed processing poses for the whole video")
    return pose_data, pose_confidence, fps

def mediapipe_hands_pose(image):
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(image_rgb)

    # Initialize empty arrays
    left_hand = np.zeros((21, 3), dtype=np.float32)
    right_hand = np.zeros((21, 3), dtype=np.float32)
    left_conf = np.zeros(21, dtype=np.float32)
    right_conf = np.zeros(21, dtype=np.float32)

    if results.multi_hand_landmarks and results.multi_handedness:
        for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
            coords = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark], dtype=np.float32)
            conf = np.zeros(21, dtype=np.float32)  # Mediapipe doesn't provide confidence → assume 1.0

            label = handedness.classification[0].label  # 'Left' or 'Right'
            if label == "Left":
                left_hand, left_conf = coords, conf
            else:
                right_hand, right_conf = coords, conf

    # --- Combine both hands into single arrays ---
    pose_data = np.concatenate([left_hand, right_hand], axis=0)    
    confidence = np.concatenate([left_conf, right_conf], axis=0)   

    return pose_data, confidence

def write_examples_tsv(examples: List[Example],
                       output_dir: str,
                       encoder_prompt: str,
                       decoder_prompt: str,
                       split_name: str,):
    """
    If signal_start and signal_end are not required (when all the frames are used), must be set as 0.

    :param examples:
    :param output_dir:
    :param encoder_prompt:
    :param decoder_prompt:
    :param split_name:
    :return:
    """

    filepath = os.path.join(output_dir, f"rwth_phoenix2014_t.{split_name}.tsv")

    logging.debug("Writing generated examples to: '%s'" % filepath)

    fieldnames = ["signal", "signal_start", "signal_end", "encoder_prompt", "decoder_prompt", "output"]

    with open(filepath, "w", newline="") as outhandle:
        writer = csv.DictWriter(outhandle, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()

        for example in examples:
            row_dict = {
                "signal": example["pose_filepath"],
                "signal_start": 0,
                "signal_end": 0,
                "encoder_prompt": encoder_prompt,
                "decoder_prompt": decoder_prompt,
                "output": example["text"]
            }

            writer.writerow(row_dict)


def main():
    print("Beginning preprocessing with Mediapipe Hands")
    args = parse_arguments()
    print(f"\nARGS:\n{args}\n")

    logging.basicConfig(level=logging.DEBUG)
    logging.debug(args)

    print("Beginning load Phoenix dataset")
    phoenix_with_poses = load_dataset(data_dir=args.tfds_data_dir)
    print("Phoenix dataset load complete.")

    pose_header = load_pose_header(phoenix_with_poses) 

    stats = {}

    for split_name in ["train", "validation", "test"]:
        examples = list(generate_examples(dataset=phoenix_with_poses,
                                          split_name=split_name,
                                          pose_header=pose_header,
                                          pose_dir=args.pose_dir,
                                          dry_run=args.dry_run))

        stats[split_name] = len(examples)

        print("Beginning writing examples to .tsv")
        write_examples_tsv(examples=examples,
                           output_dir=args.output_dir,
                           encoder_prompt=args.encoder_prompt,
                           decoder_prompt=args.decoder_prompt,
                           split_name=split_name)

    logging.debug("Number of examples found:")
    logging.debug(stats)


if __name__ == "__main__":
    main()