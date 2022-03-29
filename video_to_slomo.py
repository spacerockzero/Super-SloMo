#!/usr/bin/env python3
import argparse
import os
import os.path
import ctypes
from posixpath import abspath
from shutil import rmtree, move, copytree
from PIL import Image
import torch
import torchvision.transforms as transforms
import model
import dataloader
import platform
from tqdm import tqdm
import math
import itertools
from more_itertools import consume

# For parsing commandline arguments
parser = argparse.ArgumentParser()
parser.add_argument("--ffmpeg_dir", type=str, default="", help='path to ffmpeg.exe')
parser.add_argument("--video", type=str, required=True, help='path of video to be converted')
parser.add_argument("--checkpoint", type=str, required=True, help='path of checkpoint for pretrained model')
parser.add_argument("--fps", type=float, default=30, help='specify fps of output video. Default: 30.')
parser.add_argument("--sf", type=int, required=True, help='specify the slomo factor N. This will increase the frames by Nx. Example sf=2 ==> 2x frames')
parser.add_argument("--batch_size", type=int, default=1, help='Specify batch size for faster conversion. This will depend on your cpu/gpu memory. Default: 1')
parser.add_argument("--output", type=str, default="output.mkv", help='Specify output file name. Default: output.mp4')
parser.add_argument("--copy_frames", type=bool, default=False, help='Should I copy processed frames to specified dir. Default: False')
parser.add_argument("--copy_frames_to", type=str, default="./", help='Copy processed frames to specified dir. Default: ./')
parser.add_argument("--resume", type=bool, default=False, help='Resume run from last saved interval. Default: False')
parser.add_argument("--save_interval", type=int, default=100, help='How often by input frame to save progress. Default: 100')
parser.add_argument("--img_dir", type=str, default=False, help='Where you get your raw frame images from, if we are doing an image morph instead of slowmo video. Default: False')
args = parser.parse_args()

def save_progress(frame, dir):
    # print('saving progress', frame)
    """
    Saves the progress of the conversion.

    Parameters
    ----------
        frame : int
            current frame number.
        dir : string
            path to directory to output the progress.

    Returns
    -------
        None
    """

    with open(os.path.join(dir, "progress.txt"), "w") as f:
        f.write("{}".format(frame))

def read_progress(dir):
    print('reading progress')
    """
    Reads the progress of the conversion.

    Parameters
    ----------
        dir : string
            path to directory to read the progress.

    Returns
    -------
        frame : int
            current frame number.
    """
    frame = 0
    filepath = os.path.join(dir, "progress.txt")
    if os.path.isfile(filepath):
        with open(filepath, "r") as f:
            frame = int(f.read())
    else:
        print("No progress file found. Creating one.")
        save_progress(frame, dir)
    return frame

def check():
    """
    Checks the validity of commandline arguments.

    Parameters
    ----------
        None

    Returns
    -------
        error : string
            Error message if error occurs otherwise blank string.
    """


    error = ""
    if (args.sf < 2):
        error = "Error: --sf/slomo factor has to be atleast 2"
    if (args.batch_size < 1):
        error = "Error: --batch_size has to be atleast 1"
    if (args.fps < 1):
        error = "Error: --fps has to be atleast 1"
    # if ".mkv" not in args.output:
    #     error = "output needs to have mkv container"
    return error

def extract_frames(video, outDir):
    """
    Converts the `video` to images.

    Parameters
    ----------
        video : string
            full path to the video file.
        outDir : string
            path to directory to output the extracted images.

    Returns
    -------
        error : string
            Error message if error occurs otherwise blank string.
    """


    error = ""
    IS_WINDOWS = 'Windows' == platform.system()

    if IS_WINDOWS:
        ffmpeg_path = os.path.join(args.ffmpeg_dir, "ffmpeg")
    else:
        ffmpeg_path = "ffmpeg"

    print('{} -i {} -vsync 0 {}/%06d.png'.format(ffmpeg_path, video, outDir))
    retn = os.system('{} -i "{}" -vsync 0 {}/%06d.png'.format(ffmpeg_path, video, outDir))
    if retn:
        error = "Error converting file:{}. Exiting.".format(video)
    return error

def copy_frames(src_dir, dst_dir):
    """
    Copies the frames from `src_dir` to `dst_dir`.

    Parameters
    ----------
        src_dir : string
            path to the source directory.
        dst_dir : string
            path to the destination directory.

    Returns
    -------
        None
    """

    print("Copying frames from {} to {}".format(src_dir, dst_dir))
    error = ""
    retn = copytree(src_dir, dst_dir)
    if retn:
        print("Error copying frames. Exiting.")
    return error

def create_video(dir):
    IS_WINDOWS = 'Windows' == platform.system()

    if IS_WINDOWS:
        ffmpeg_path = os.path.join(args.ffmpeg_dir, "ffmpeg")
    else:
        ffmpeg_path = "ffmpeg"

    error = ""
    # print('{} -r {} -i {}/%d.png -vcodec ffvhuff {}'.format(ffmpeg_path, args.fps, dir, args.output))
    # retn = os.system('{} -r {} -i {}/%d.png -vcodec ffvhuff "{}"'.format(ffmpeg_path, args.fps, dir, args.output))
    print('{} -r {} -i {}/%d.png -c:v libx264 -vf {}'.format(ffmpeg_path, args.fps, dir, args.output))
    retn = os.system('{} -r {} -i {}/%d.png -c:v libx264 -vf "{}"'.format(ffmpeg_path, args.fps, dir, args.output))
    if retn:
        error = "Error creating output video. Exiting."
    return error


def main():
    print('args', args)
    # Check if arguments are okay
    # error = check()
    # if error:
    #     print(error)
    #     exit(1)

    # Create extraction folder and extract frames
    IS_WINDOWS = 'Windows' == platform.system()
    extractionDir = "tmpSuperSloMo"
    if not IS_WINDOWS:
        # Assuming UNIX-like system where "." indicates hidden directories
        extractionDir = "." + extractionDir
    if os.path.isdir(extractionDir) and args.resume != True:
        rmtree(extractionDir)
        os.mkdir(extractionDir)
    if IS_WINDOWS:
        FILE_ATTRIBUTE_HIDDEN = 0x02
        # ctypes.windll only exists on Windows
        ctypes.windll.kernel32.SetFileAttributesW(extractionDir, FILE_ATTRIBUTE_HIDDEN)

    extractionPath = os.path.join(extractionDir, "input")
    outputPath     = os.path.join(extractionDir, "output")
    progress_path  = abspath(extractionDir)
    if args.resume != True:
        os.mkdir(extractionPath)
        os.mkdir(outputPath)
        if args.img_dir:
            copy_frames(args.img_dir, abspath(extractionPath))
        else:
            error = extract_frames(args.video, extractionPath)
        if error:
            print(error)
            exit(1)

    # Initialize transforms
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    mean = [0.429, 0.431, 0.397]
    std  = [1, 1, 1]
    normalize = transforms.Normalize(mean=mean,
                                     std=std)

    negmean = [x * -1 for x in mean]
    revNormalize = transforms.Normalize(mean=negmean, std=std)

    # Temporary fix for issue #7 https://github.com/avinashpaliwal/Super-SloMo/issues/7 -
    # - Removed per channel mean subtraction for CPU.
    if (device == "cpu"):
        transform = transforms.Compose([transforms.ToTensor()])
        TP = transforms.Compose([transforms.ToPILImage()])
    else:
        transform = transforms.Compose([transforms.ToTensor(), normalize])
        TP = transforms.Compose([revNormalize, transforms.ToPILImage()])

    # Load data
    frameCounter = 1
    input_frame = 0
    save_interval = args.save_interval if args.save_interval is not None else 100
    if args.resume == True:
        print('resume')
        # sync input progress
        input_frame = read_progress(progress_path)
        # sync output progress
        frameCounter = input_frame * args.sf
        print('frameCounter', frameCounter)

    videoFrames = dataloader.Video(root=extractionPath, transform=transform, resume=input_frame)
    videoFramesloader = torch.utils.data.DataLoader(videoFrames, batch_size=args.batch_size, shuffle=False)

    # Initialize model
    flowComp = model.UNet(6, 4)
    flowComp.to(device)
    for param in flowComp.parameters():
        param.requires_grad = False
    ArbTimeFlowIntrp = model.UNet(20, 5)
    ArbTimeFlowIntrp.to(device)
    for param in ArbTimeFlowIntrp.parameters():
        param.requires_grad = False

    flowBackWarp = model.backWarp(videoFrames.dim[0], videoFrames.dim[1], device)
    flowBackWarp = flowBackWarp.to(device)

    dict1 = torch.load(args.checkpoint, map_location='cpu')
    ArbTimeFlowIntrp.load_state_dict(dict1['state_dictAT'])
    flowComp.load_state_dict(dict1['state_dictFC'])

    with torch.no_grad():
        for _, (frame0, frame1) in enumerate(tqdm(videoFramesloader), 0):
            # print('output frame:', frameCounter)
            # print('iteration:', _)

            I0 = frame0.to(device)
            I1 = frame1.to(device)

            flowOut = flowComp(torch.cat((I0, I1), dim=1))
            F_0_1 = flowOut[:,:2,:,:]
            F_1_0 = flowOut[:,2:,:,:]

            # Save reference frames in output folder
            for batchIndex in range(args.batch_size):
                (TP(frame0[batchIndex].detach())).resize(videoFrames.origDim, Image.BILINEAR).save(os.path.join(outputPath, str(frameCounter + args.sf * batchIndex) + ".png"))
            frameCounter += 1

            # Generate intermediate frames
            for intermediateIndex in range(1, args.sf):
                t = float(intermediateIndex) / args.sf
                temp = -t * (1 - t)
                fCoeff = [temp, t * t, (1 - t) * (1 - t), temp]

                F_t_0 = fCoeff[0] * F_0_1 + fCoeff[1] * F_1_0
                F_t_1 = fCoeff[2] * F_0_1 + fCoeff[3] * F_1_0

                g_I0_F_t_0 = flowBackWarp(I0, F_t_0)
                g_I1_F_t_1 = flowBackWarp(I1, F_t_1)

                intrpOut = ArbTimeFlowIntrp(torch.cat((I0, I1, F_0_1, F_1_0, F_t_1, F_t_0, g_I1_F_t_1, g_I0_F_t_0), dim=1))

                F_t_0_f = intrpOut[:, :2, :, :] + F_t_0
                F_t_1_f = intrpOut[:, 2:4, :, :] + F_t_1
                V_t_0   = torch.sigmoid(intrpOut[:, 4:5, :, :])
                V_t_1   = 1 - V_t_0

                g_I0_F_t_0_f = flowBackWarp(I0, F_t_0_f)
                g_I1_F_t_1_f = flowBackWarp(I1, F_t_1_f)

                wCoeff = [1 - t, t]

                Ft_p = (wCoeff[0] * V_t_0 * g_I0_F_t_0_f + wCoeff[1] * V_t_1 * g_I1_F_t_1_f) / (wCoeff[0] * V_t_0 + wCoeff[1] * V_t_1)

                # Save intermediate frame
                for batchIndex in range(args.batch_size):
                    (TP(Ft_p[batchIndex].cpu().detach())).resize(videoFrames.origDim, Image.BILINEAR).save(os.path.join(outputPath, str(frameCounter + args.sf * batchIndex) + ".png"))

                frameCounter += 1

            # Set counter accounting for batching of frames
            frameCounter += args.sf * (args.batch_size - 1)
            progress = math.floor(frameCounter / args.sf)
            # print("input frame progress:", progress)
            if progress % save_interval == 0:
                # save the input frame number for resuming
                save_progress(progress, progress_path)

    # Generate video from interpolated frames
    create_video(outputPath)

    if args.copy_frames:
        print('args.copy_frames_to', args.copy_frames_to)
        copytree(outputPath, args.copy_frames_to)

    if args.resume != False:
        # Remove temporary files
        rmtree(extractionDir)

    exit(0)

main()
