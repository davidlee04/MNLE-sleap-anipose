import cv2
import skvideo.io
import matplotlib.pyplot as plt
import numpy as np

from tqdm import trange

from preprocessing import get_bodyparts, get_point_scores, get_2d_poses

def visualize_labels(config, poses_2d_orig, point_scores, bodyparts, vid_fname, outname):
	scheme = config['labeling']['scheme']
	cap = cv2.VideoCapture(vid_fname)
    # cap.set(1,0)

	fps = cap.get(cv2.CAP_PROP_FPS)
	writer = skvideo.io.FFmpegWriter(outname, inputdict={
	    # '-hwaccel': 'auto',
	    '-framerate': str(fps),
	}, outputdict={
	    '-vcodec': 'h264', '-qp': '28',
	    '-pix_fmt': 'yuv420p', # to support more players
	    '-vf': 'pad=ceil(iw/2)*2:ceil(ih/2)*2' # to handle width/height not divisible by 2
	})

	last = poses_2d_orig.shape[0]

	cmap = plt.get_cmap('tab10')

	pointsx = poses_2d_orig[:,:,0,0]
	pointsy = poses_2d_orig[:,:,1,0]
	pointsx = np.stack(pointsx, axis=-1)
	pointsy = np.stack(pointsy, axis=-1)

	points = [(pointsx[i,:], pointsy[i,:]) for i, bp in enumerate(bodyparts)]
	points = np.array(points)

	scores = [point_scores[:, i, 0] for i, bp in enumerate(bodyparts)]
	scores = np.array(scores)

	scores[np.isnan(scores)] = 0
	scores[np.isnan(points[:, 0])] = 0

	good = np.array(scores) > 0.1
	points[:, 0, :][~good] = np.nan
	points[:, 1, :][~good] = np.nan

	all_points = points

	for ix in trange(last, ncols=70):
	    ret, frame = cap.read()
	    if not ret:
	        break

	    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
	    points = all_points[:, :, ix]
	    img = label_frame(img, points, scheme, bodyparts)

	    writer.writeFrame(img)

	cap.release()
	writer.close()

def label_video(config, analysis_file, vid_fname, outname):
	bp = get_bodyparts(analysis_file)
	point_scores = get_point_scores(analysis_file)
	points_2d = get_2d_poses(analysis_file)
	visualize_labels(config, points_2d, point_scores, bp, vid_fname, outname)

def connect(img, points, bps, bodyparts, col=(0,255,0,255)):
    try:
        ixs = [bodyparts.index(bp) for bp in bps]
    except ValueError:
        return

    for a, b in zip(ixs, ixs[1:]):
        if np.any(np.isnan(points[[a,b]])):
            continue
        pa = tuple(np.int32(points[a]))
        pb = tuple(np.int32(points[b]))
        cv2.line(img, tuple(pa), tuple(pb), col, 4)

def connect_all(img, points, scheme, bodyparts):
    cmap = plt.get_cmap('tab10')
    for cnum, bps in enumerate(scheme):
        col = cmap(cnum % 10, bytes=True)
        col = [int(c) for c in col]
        connect(img, points, bps, bodyparts, col)

def label_frame(img, points, scheme, bodyparts, cmap='tab10'):
    n_joints, _ = points.shape

    cmap_c = plt.get_cmap(cmap)
    connect_all(img, points, scheme, bodyparts)

    for lnum, (x, y) in enumerate(points):
        if np.isnan(x) or np.isnan(y):
            continue
        x = np.clip(x, 1, img.shape[1]-1)
        y = np.clip(y, 1, img.shape[0]-1)
        x = int(round(x))
        y = int(round(y))
        col = cmap_c(lnum % 10, bytes=True)
        col = [int(c) for c in col]
#         col = (255, 255, 255)
        cv2.circle(img,(x,y), 5, col[:3], -1)

    return img