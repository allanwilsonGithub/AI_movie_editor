import numpy as np
import os
import shutil
from distutils.dir_util import copy_tree
import cv2
import matplotlib.pyplot as plt
import insightface
from insightface.app import FaceAnalysis
from insightface.data import get_image as ins_get_image
import time 
import sys   
from collections import deque

with open("movies_list.txt") as file:
    movie_list = [line.rstrip() for line in file]

SHOW_STAGES = False
FACE_PIC_DIR_PIC = "PICS/source_images/face_source/"
FACE_PIC_DIR_MOVIES = "MOVIES/faces_source/"
CANVAS_DIR = 'PICS/source_images/canvas/'
OUTPUT_DIR = 'PICS/output_images/'
movie_source_dir = 'next/'

app_face = FaceAnalysis(name='buffalo_l')
app_face.prepare(ctx_id=0, det_size=(640,640))

def get_all_source_faces(FACE_PIC_DIR):
    ALL_FACES = []
    for individual_face in os.listdir(FACE_PIC_DIR):
        print(f"{FACE_PIC_DIR}{individual_face}")
        img = cv2.imread(f"{FACE_PIC_DIR}{individual_face}")
        plt.imshow(img[:,:,::-1])
        if SHOW_STAGES:
            plt.show()
        source_faces = app_face.get(img)
        source_face = source_faces[0]
        print(source_face.sex)
        bbox1 = source_face['bbox']
        bbox1 = [int(b) for b in bbox1]
        plt.imshow(img[bbox1[1]:bbox1[3],bbox1[0]:bbox1[2],::-1])
        if SHOW_STAGES:
            plt.show()
        ALL_FACES.append(source_face)
        print(f"Creating face source: {len(ALL_FACES)}")    
    print(f"There are {len(ALL_FACES)} faces...")
    return ALL_FACES

def do_the_swap_one_source_face(ALL_FACES, CANVAS_PIC, OUTPUT_DIR, filename,set_number):
    ## Get Target Faces
    img2 = cv2.imread(CANVAS_PIC)
    plt.imshow(img2[:,:,::-1])
    if SHOW_STAGES:
        plt.show()
    target_faces = app_face.get(img2)
    if len(target_faces) == 0:
        src = CANVAS_PIC
        dst = f"{OUTPUT_DIR}{set_number}/{filename}"
        shutil.copy(src, dst)

    else:
        print(f"there are {len(target_faces)} faces")
        target_face = target_faces[0]
        bbox2 = target_face['bbox']
        bbox2 = [int(b) for b in bbox2]
        try:
            plt.imshow(img2[bbox2[1]:bbox2[3],bbox2[0]:bbox2[2],::-1])
        except ValueError:
            pass
        
        if SHOW_STAGES:
            plt.show()

        swapper = insightface.model_zoo.get_model('inswapper_128.onnx',
                                        download=False,
                                        download_zip=False)


        res= cv2.imread(CANVAS_PIC)
        plt.imshow(res[:,:,::-1])
        if SHOW_STAGES:
            plt.show()

        for face in target_faces:
            res_new =swapper.get(res, target_face, ALL_FACES[set_number-1], paste_back=True)

        plt.imshow(res_new[:,:,::-1])
        if SHOW_STAGES:
            plt.show()

        epoch_time = int(time.time())
        if 'frame' in filename:
            if not os.path.exists(f"{OUTPUT_DIR}{set_number}/{filename}"):
                cv2.imwrite(f"{OUTPUT_DIR}{set_number}/{filename}", res_new)
        else:
            cv2.imwrite(f"{OUTPUT_DIR}{filename}_{epoch_time}.png", res_new)

def do_the_swap_multiple_sources_and_targets(ALL_ORIGINAL_FACES, CANVAS_PIC, OUTPUT_DIR, filename,set_number):
    ## Get target Faces (all the faces in the original picture)
    img2 = cv2.imread(CANVAS_PIC)
    plt.imshow(img2[:,:,::-1])
    if SHOW_STAGES:
        plt.show()
    target_faces = app_face.get(img2)

    #If there are no faces found just copy the frame without modifying
    if len(target_faces) == 0:
        src = CANVAS_PIC
        dst = f"{OUTPUT_DIR}{set_number}/{filename}"
        shutil.copy(src, dst)
        print(f"This frame had no face so skipped: {OUTPUT_DIR}{set_number}/{filename}")

    else:
        print(f"there are {len(ALL_ORIGINAL_FACES)} original faces")
        print(f"there are {len(target_faces)} target faces")

        res= cv2.imread(CANVAS_PIC)

        for orig_face_i in range(0,len(ALL_ORIGINAL_FACES)):
            if orig_face_i < len(target_faces):
                print(f"Swapping face {orig_face_i+1}/{len(target_faces)}")
                target_face = target_faces[orig_face_i]
                bbox2 = target_face['bbox']
                bbox2 = [int(b) for b in bbox2]
                try:
                    plt.imshow(img2[bbox2[1]:bbox2[3],bbox2[0]:bbox2[2],::-1])
                except ValueError:
                    pass
                
                if SHOW_STAGES:
                    plt.show()

                swapper = insightface.model_zoo.get_model('inswapper_128.onnx',
                                                download=False,
                                                download_zip=False)

                plt.imshow(res[:,:,::-1])
                if SHOW_STAGES:
                    plt.show()

                res_new = swapper.get(res, target_face, ALL_ORIGINAL_FACES[orig_face_i], paste_back=True)

                plt.imshow(res_new[:,:,::-1])
                if SHOW_STAGES:
                    plt.show()

                res = res_new

                epoch_time = int(time.time())
                if 'frame' in filename:
                    if not os.path.exists(f"{OUTPUT_DIR}{set_number}/{filename}"):
                        cv2.imwrite(f"{OUTPUT_DIR}{set_number}/{filename}", res_new)
                else:
                    if orig_face_i < len(target_faces)-1:
                        print(f".")
                    else:
                        cv2.imwrite(f"{OUTPUT_DIR}{filename}_{epoch_time}.png", res_new)


def generate_many_source_to_many_canvas():
    # Take all faces in source dir and add them to every pic in canvas dir
    ALL_FACES_FIRST_GET = get_all_source_faces(FACE_PIC_DIR_PIC)
    ALL_FACES = ALL_FACES_FIRST_GET

    index = 1
    for rotation in range(0,len(ALL_FACES_FIRST_GET)):
        print(f"Rotation: {rotation+1}")
        
        for filename in os.listdir(CANVAS_DIR):
            CANVAS_PIC = f"{CANVAS_DIR}/{filename}"
            print(f"Canvas: {index}:{len(os.listdir(CANVAS_DIR))} :::  {filename}")
            do_the_swap_multiple_sources_and_targets(ALL_FACES, CANVAS_PIC, OUTPUT_DIR, filename, set_number=1)
            index += 1

        ALL_FACES_ROTATED_ONCE = deque(ALL_FACES)
        ALL_FACES_ROTATED_ONCE.rotate()
        ALL_FACES = ALL_FACES_ROTATED_ONCE


def make_movie(movie_source_file):
    # Put all frames into MOVIES/FRAMES_SOURCE
    # Put at least one source pic in MOVIES/FACE_SOURCE
    # run script
    # Import all frames from MOVIE_FRAMES_OUTPUT back into Openshot editor to make movie
    FRAME_SOURCE_DIR = 'MOVIES/FRAMES_SOURCE/'
    FRAME_OUTPUT_DIR = 'MOVIES/MOVIE_FRAMES_OUTPUT/'
    # Get all source Frames
    capture = cv2.VideoCapture(f"MOVIES/movie_files/{movie_source_dir}{movie_source_file}.avi")
    frameNr = 0
    while (True):
        success, frame = capture.read()
        if success:
            if not os.path.exists(f'{FRAME_SOURCE_DIR}frame-{frameNr}.jpg'):
                cv2.imwrite(f'{FRAME_SOURCE_DIR}frame-{frameNr}.jpg', frame)
        else:
            break
    
        frameNr = frameNr+1
 
    capture.release()

    # Get all source faces
    ALL_FACES = get_all_source_faces(FACE_PIC_DIR_MOVIES)
    set_number = 1
    # Create output dirs for frames
    for face in ALL_FACES:
        if not os.path.exists(f"{FRAME_OUTPUT_DIR}{set_number}"):
            os.mkdir(f"{FRAME_OUTPUT_DIR}{set_number}")
        set_number += 1
    number_of_sets = set_number

    # Iterate over each source frame creating a modified version for each source face
    counter = 0
    total_frames = len(os.listdir(FRAME_SOURCE_DIR))
    for frame_pic in os.listdir(FRAME_SOURCE_DIR):
        print("########################################################################")
        print(f"{counter}/{total_frames}")
        print(frame_pic)
        print("########################################################################")
        for set_number in range(1,number_of_sets):
            if not os.path.exists(f"{FRAME_OUTPUT_DIR}{set_number}/{frame_pic}"):
                CANVAS_PIC = f"{FRAME_SOURCE_DIR}/{frame_pic}"
                do_the_swap_one_source_face(ALL_FACES, CANVAS_PIC, FRAME_OUTPUT_DIR, frame_pic, set_number)
            else:
                print(f"skipping {OUTPUT_DIR}{set_number}/{frame_pic} : Already exists")
        counter += 1
    
    print(f"Copying files to: {movie_source_file}")
    copy_tree(str(FRAME_OUTPUT_DIR), f"MOVIES/ready_image_sets/{str(movie_source_file)}")

    print(f"Cleaning working directories...")
    copy_tree(str(FRAME_OUTPUT_DIR), f"MOVIES/ready_image_sets/{str(movie_source_file)}")

    for folder in (FRAME_OUTPUT_DIR, FRAME_SOURCE_DIR):
        for filename in os.listdir(folder):
            file_path = os.path.join(folder, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)                
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                print('Failed to delete %s. Reason: %s' % (file_path, e))
            



if str(sys.argv[1]) == "pic":
    generate_many_source_to_many_canvas()
elif str(sys.argv[1]) == "vid":
    for movie_source_file in movie_list:
        if movie_source_file in (os.listdir(f"MOVIES/ready_image_sets/")):
            print(f"Directory exists, skipping: {movie_source_file}")
        else:
            make_movie(movie_source_file)
else:
    print(str(sys.argv[1]))
    print(f"Make a choice bro!")


# TODO
# when above clean up is done make the input files an input file that just checks what has been done and restarts from the next movie
# Male and female swap
# Ability to change order somehow
# Voice