import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import src.pathDefine

MAX_TIME_STEPS = 1299

audio_feature = src.pathDefine.features_slow_audio_file
# audio_feature = 'audio_Fast_feature.npy'
# audio_feature = src.pathDefine.features_audio_file
bvh_feature = src.pathDefine.features_bvh_file

audio = np.load(audio_feature) #(1299, 755)
audio = audio[: audio.shape[0], :]
df = pd.read_csv(bvh_feature)
bvh = df.iloc[: df.shape[0], 4:df.shape[1]].values #(1299, 103)

def get_batch():
    global BATCH_START, MAX_TIME_STEPS, audio, bvh
    frameNum_audio = audio.shape[0]
    if frameNum_audio > MAX_TIME_STEPS:
    #     multiple batches TBC
        seq = audio[:MAX_TIME_STEPS]
        res = bvh
        seqlen = MAX_TIME_STEPS
    else:
        dim_audio = audio.shape[1]
        rows = MAX_TIME_STEPS - frameNum_audio
        seq = np.pad(audio, ((0, rows),(0, 0)), 'constant', constant_values=0)
        res = bvh
        seqlen = audio.shape[0]
    print(seqlen,res.shape,seq.shape)
    return [seq[np.newaxis, :, :], res[np.newaxis, :, :], seqlen]

if __name__ == '__main__':
    sess = tf.Session()
    new_saver = tf.train.import_meta_graph('model/test2.meta')
    new_saver.restore(sess, tf.train.latest_checkpoint('model/'))
    graph = tf.get_default_graph()

    predict = tf.get_collection('predict')[0]
    xs = graph.get_operation_by_name("xs").outputs[0]
    seqlen = graph.get_operation_by_name("seqlen").outputs[0]
    ys = graph.get_operation_by_name("ys").outputs[0]
    average_cost = graph.get_tensor_by_name("average_cost:0")

    seq, res, seql = get_batch()
    sum_cost = graph.get_tensor_by_name("sum_cost:0")

    # fast
    predict_result = sess.run(predict, feed_dict={xs: seq,ys:res,seqlen:seql})
    predict_result = predict_result[0:seql]

    # normal
    predict_result,mse_cost,sum_cost = sess.run([predict,average_cost,sum_cost], feed_dict={xs: seq,ys:res,seqlen:seql})
    # print(res.shape)
    # print(ys.shape)
    # predict_result = predict_result[0:seql].flatten()
    # ys = ys[0][0:seql].flatten()
    #
    # div = ((predict_result - ys)**2).sum()
    # print(div)
    # name = 'output/motion_output'
    predict_result = predict_result[0:seql]
    np.save(src.pathDefine.motion_output, predict_result)
    print('==Test SLOW==')
    print('The shape of input audio is: ',audio.shape)
    print('The shape of output motion is: ',predict_result.shape)
    print('The SSE is: ',sum_cost)
    print('The MSE is: ',mse_cost)


