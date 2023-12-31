from tqdm import tqdm
import numpy as np
import statsmodels.api as sm
import sleap
import tensorflow as tf
from numpy import array as arr
from scipy import signal, stats
from scipy.interpolate import splev, splrep
from scipy.spatial.distance import cdist
from scipy.spatial import cKDTree
from scipy.special import logsumexp
from multiprocessing import cpu_count
from multiprocessing import Pool, get_context

def nan_helper(y):
    return np.isnan(y), lambda z: z.nonzero()[0]

def remove_dups(pts, thres=7):
    tindex = np.repeat(np.arange(pts.shape[0])[:, None], pts.shape[1], axis=1)*100
    pts_ix = np.dstack([pts, tindex])
    tree = cKDTree(pts_ix.reshape(-1, 3))

    shape = (pts.shape[0], pts.shape[1])
    pairs = tree.query_pairs(thres)
    indices = [b for a, b in pairs]

    if len(pairs) == 0:
        return pts

    i0, i1 = np.unravel_index(indices, shape)
    pts_out = np.copy(pts)
    pts_out[i0, i1] = np.nan

    return pts_out

def viterbi_path(points, scores, n_back=3, thres_dist=30):
    n_frames = points.shape[0]

    points_nans = remove_dups(points, thres=5)
    # points_nans[scores < 0.01] = np.nan

    num_points = np.sum(~np.isnan(points_nans[:, :, 0]), axis=1)
    num_max = np.max(num_points)

    particles = np.zeros((n_frames, num_max * n_back + 1, 3), dtype='float64')
    valid = np.zeros(n_frames, dtype='int64')
    for i in range(n_frames):
        s = 0
        for j in range(n_back):
            if i-j < 0:
                break
            ixs = np.where(~np.isnan(points_nans[i-j, :, 0]))[0]
            n_valid = len(ixs)
            particles[i, s:s+n_valid, :2] = points[i-j, ixs]
            particles[i, s:s+n_valid, 2] = scores[i-j, ixs] * np.power(2.0, -j)
            s += n_valid
        if s == 0:
            particles[i, 0] = [-1, -1, 0.001] # missing point
            s = 1
        valid[i] = s

    ## viterbi algorithm
    n_particles = np.max(valid)

    T_logprob = np.zeros((n_frames, n_particles), dtype='float64')
    T_logprob[:] = -np.inf
    T_back = np.zeros((n_frames, n_particles), dtype='int64')

    T_logprob[0, :valid[0]] = np.log(particles[0, :valid[0], 2])
    T_back[0, :] = -1

    for i in range(1, n_frames):
        va, vb = valid[i-1], valid[i]
        pa = particles[i-1, :va, :2]
        pb = particles[i, :vb, :2]

        dists = cdist(pa, pb)
        cdf_high = stats.norm.logcdf(dists + 2, scale=thres_dist)
        cdf_low = stats.norm.logcdf(dists - 2, scale=thres_dist)
        cdfs = np.array([cdf_high, cdf_low])
        P_trans = logsumexp(cdfs.T, b=[1,-1], axis=2)

        P_trans[P_trans < -100] = -100

        # take care of missing transitions
        P_trans[pb[:, 0] == -1, :] = np.log(0.001)
        P_trans[:, pa[:, 0] == -1] = np.log(0.001)

        pflat = particles[i, :vb, 2]
        possible = T_logprob[i-1, :va] + P_trans

        T_logprob[i, :vb] = np.max(possible, axis=1) + np.log(pflat)
        T_back[i, :vb] = np.argmax(possible, axis=1)

    out = np.zeros(n_frames, dtype='int')
    out[-1] = np.argmax(T_logprob[-1])

    for i in range(n_frames-1, 0, -1):
        out[i-1] = T_back[i, out[i]]

    trace = [particles[i, out[i]] for i in range(n_frames)]
    trace = np.array(trace)

    points_new = trace[:, :2]
    scores_new = trace[:, 2]
    # scores_new[out >= num_points] = 0

    return points_new, scores_new


def viterbi_path_wrapper(args):
    jix, pts, scs, max_offset, thres_dist = args
    pts_new, scs_new = viterbi_path(pts, scs, max_offset, thres_dist)
    return jix, pts_new, scs_new

def viterbi(config, all_points):
    n_frames, n_joints, n_possible, _ = all_points.shape

    points_full = all_points[:, :, :, :2]
    scores_full = all_points[:, :, :, 2]
    print(all_points.shape)
    print(all_points)

    points_full[scores_full < config['filter']['score_threshold']] = np.nan

    points = np.full((n_frames, n_joints, 2), np.nan, dtype='float64')
    scores = np.empty((n_frames, n_joints), dtype='float64')

    if config['filter']['multiprocessing']:
        n_proc_default = max(min(cpu_count() // 2, n_joints), 1)
        n_proc = config['filter'].get('n_proc', n_proc_default)
    else:
        n_proc = 1
    ctx = get_context('spawn')
    pool = ctx.Pool(n_proc)

    max_offset = config['filter']['n_back']
    thres_dist = config['filter']['offset_threshold']

    iterable = [ (jix, points_full[:, jix, :], scores_full[:, jix],
                  max_offset, thres_dist)
                 for jix in range(n_joints) ]

    results = pool.imap_unordered(viterbi_path_wrapper, iterable)

    for jix, pts_new, scs_new in tqdm(results, ncols=70):
        points[:, jix] = pts_new
        scores[:, jix] = scs_new

    pool.close()
    pool.join()

    return points, scores

def filter_pose_viterbi(config, points_full, scores_full):
    assert 'slp_model' in config, 'need slp_model key in config'
    predictor = sleap.load_model(config['slp_model'])
    predictor.inference_model.single_instance_layer.return_confmaps = True
    output_stride = predictor.confmap_config.model.heads.single_instance.output_stride

    video = sleap.load_video(R'D:\AN_local\AN664\041023\cam0\Cam00_230410_142351.avi')
    imgs = video[0:15000]

    predictions = predictor.inference_model.predict(imgs)
    confmaps = predictions['confmaps']
    print(confmaps.shape)

    all_res = get_topk(confmaps, config['filter']['nkey'], output_stride)
    print(all_res.shape) # frames x bodyparts x nkey x 3

    return viterbi(config, all_res)

def get_topk(confmaps, k, output_stride):
    all_res = []
    for i in range(confmaps.shape[0]):
        res_i = []
        for j in range(confmaps.shape[-1]):
            cm = confmaps[i,:,:,j]
            peaks = tf.math.top_k(tf.reshape(cm,[-1]),k=k)
            inds = peaks.indices
            vals = peaks.values.numpy()
            subs = tf.unravel_index(inds, cm.shape)
            subs = subs.numpy().astype('float32')
            x = subs[1,:] * output_stride
            y = subs[0,:] * output_stride
            res_i.append([x,y,vals])
        all_res.append(np.stack(res_i, axis=0))
    all_res = np.stack(all_res, axis=0)
    all_res = np.transpose(all_res, (0,1,3,2))
    return all_res

def filter_pose_medfilt(config, points_full, scores_full):
    n_frames, n_joints, n_possible, _ = points_full.shape

    points = np.full((n_frames, n_joints, 2), np.nan, dtype='float64')
    scores = np.empty((n_frames, n_joints), dtype='float64')

    for bp_ix in range(n_joints):
        x = points_full[:, bp_ix, 0, 0]
        y = points_full[:, bp_ix, 0, 1]

        score = scores_full[:, bp_ix, 0]

        xmed = signal.medfilt(x, kernel_size=config['filter']['medfilt'])
        ymed = signal.medfilt(y, kernel_size=config['filter']['medfilt'])

        errx = np.abs(x - xmed)
        erry = np.abs(y - ymed)
        err = errx + erry

        bad = np.zeros(len(x), dtype='bool')
        bad[err >= config['filter']['offset_threshold']] = True
        bad[score < config['filter']['score_threshold']] = True

        Xf = arr([xmed,ymed]).T
        Xf[bad] = np.nan

        Xfi = np.copy(Xf)

        for i in range(Xf.shape[1]):
            vals = Xfi[:, i]
            nans, ix = nan_helper(vals)
            # some data missing, but not too much
            if np.sum(nans) > 0 and np.mean(~nans) > 0.5 and np.sum(~nans) > 5:
                if config['filter']['spline']:
                    spline = splrep(ix(~nans), vals[~nans], k=3, s=0)
                    vals[nans]= splev(ix(nans), spline)
                else:
                    vals[nans] = np.interp(ix(nans), ix(~nans), vals[~nans])
            Xfi[:,i] = vals

        points[:, bp_ix, 0] = Xfi[:, 0]
        points[:, bp_ix, 1] = Xfi[:, 1]
        # dout[scorer, bp, 'interpolated'] = np.isnan(Xf[:, 0])

    scores = scores_full[:, :, 0]

    return points, scores

def convertparms2start(pn):
    """Creating a start value for sarimax in case of an value error
    See: https://groups.google.com/forum/#!topic/pystatsmodels/S_Fo53F25Rk"""
    if "ar." in pn:
        return 0
    elif "ma." in pn:
        return 0
    elif "sigma" in pn:
        return 1
    else:
        return 0

def FitSARIMAXModel(x, p, pcutoff, alpha, ARdegree, MAdegree, nforecast=0, disp=False):
    # Seasonal Autoregressive Integrated Moving-Average with eXogenous regressors (SARIMAX)
    # see http://www.statsmodels.org/stable/statespace.html#seasonal-autoregressive-integrated-moving-average-with-exogenous-regressors-sarimax
    Y = x.copy()
    Y[p < pcutoff] = np.nan  # Set uncertain estimates to nan (modeled as missing data)
    if np.sum(np.isfinite(Y)) > 10:
        # SARIMAX implementation has better prediction models than simple ARIMAX (however we do not use the seasonal etc. parameters!)
        mod = sm.tsa.statespace.SARIMAX(
            Y.flatten(),
            order=(ARdegree, 0, MAdegree),
            seasonal_order=(0, 0, 0, 0),
            simple_differencing=True,
        )
        # Autoregressive Moving Average ARMA(p,q) Model
        # mod = sm.tsa.ARIMA(Y, order=(ARdegree,0,MAdegree)) #order=(ARdegree,0,MAdegree)
        try:
            res = mod.fit(disp=disp)
        except (
            ValueError
        ):  # https://groups.google.com/forum/#!topic/pystatsmodels/S_Fo53F25Rk (let's update to statsmodels 0.10.0 soon...)
            print('value error')
            startvalues = np.array([convertparms2start(pn) for pn in mod.param_names])
            res = mod.fit(start_params=startvalues, disp=disp)
        except np.linalg.LinAlgError:
            print('linalg')
            # The process is not stationary, but the default SARIMAX model tries to solve for such a distribution...
            # Relaxing those constraints should do the job.
            mod = sm.tsa.statespace.SARIMAX(
                Y.flatten(),
                order=(ARdegree, 0, MAdegree),
                seasonal_order=(0, 0, 0, 0),
                simple_differencing=True,
                enforce_stationarity=False,
                enforce_invertibility=False,
                use_exact_diffuse=False,
            )
            res = mod.fit(disp=disp)

        predict = res.get_prediction(end=mod.nobs + nforecast - 1)
        return predict.predicted_mean, predict.conf_int(alpha=alpha)
    else:
        print('bad')
        return np.nan * np.zeros(len(Y)), np.nan * np.zeros((len(Y), 2))

def filter_pose_ar(config, points_full, scores_full):
    n_frames, n_joints, n_possible, _ = points_full.shape

    points = np.full((n_frames, n_joints, 2), np.nan, dtype='float64')
    scores = np.empty((n_frames, n_joints), dtype='float64')

    config_filter = config['filter']
    for bp_ix in range(n_joints):
        x = points_full[:, bp_ix, 0, 0]
        y = points_full[:, bp_ix, 0, 1]
        scores = scores_full[:, bp_ix, 0]

        meanx, x_conf_int = FitSARIMAXModel(
            x,
            scores,
            config_filter['score_threshold'],
            config_filter['alpha'],
            config_filter['ardegree'],
            config_filter['madegree']
        )

        meany, y_conf_int = FitSARIMAXModel(
            y,
            scores,
            config_filter['score_threshold'],
            config_filter['alpha'],
            config_filter['ardegree'],
            config_filter['madegree']
        )
        meanx[0] = x[0]
        meany[0] = y[0]
        points[:, bp_ix, 0] = meanx
        points[:, bp_ix, 1] = meany
        # scores[:, bp_ix] = 
    scores = scores_full[:, :, 0]
    return points, scores

def get_filter_poses(config, points_full, scores_full):
    filter_func = {
        'median': filter_pose_medfilt,
        'viterbi': filter_pose_viterbi,
        'ar': filter_pose_ar,
    }

    return filter_func[config['filter']['type']](config, points_full, scores_full)