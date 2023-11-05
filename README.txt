SHOULD BE EMPTY: FRAMES_SOURCE, MOVIE_FRAMES_OUTPUT

prep:
1. Put someone's facepic in 'faces_source'. Maybe 1-10 different friends' faces
2. copy a short AVI file into "movie_files"
3. in file "insightface_allu_swapper_v2.py" change 'movie_source_file'to the name of your movie
4. run ./insightface_allu_swapper_v2.py vid
5. When prompted check all the frames in FRAMES_SOURCE and make sure there is a full face in all of them
6. hit return
6. There should be some numbered dirs in MOVIE_FRAMES_OUTPUT

Copy all the frames from one dir into Openshot video editor and it will autostitch them.

HINT:
Put the original file in another OpenSHot layer and you'll get the original sound

To activate virtual python environment:
venv\Scripts\activate