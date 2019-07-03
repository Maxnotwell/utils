import h5py

import numpy as np
import scipy.io as scio
import tensorflow as tf


def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _float_feature(value):
    """Returns a float_list from a float / double."""
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


def _int64_feature(value):
    """Returns an int64_list from a bool / enum / int / uint."""
    if type(value) is tuple:
        return tf.train.Feature(int64_list=tf.train.Int64List(value=value))
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


class CreateTFRecord(object):
    def __init__(self, h5f_path, idx_path, desc_path, embed_path):
        self.h5f_path = h5f_path
        self.idx_path = idx_path
        self.desc_path = desc_path
        self.embed_path = embed_path
        self.embedding = dict()
        self.description = []
        self.load_img()
        self.load_idx()
        self.load_embed()
        self.load_desc()
        self.file_maker("train")
        self.file_maker("test")

    def file_maker(self, mode):
        if mode == "train":
            idxes = self.train_idx
        else:
            idxes = self.test_idx

        counter = 0
        total_num = 0
        for i, idx in enumerate(idxes):
            if np.mod(i, 5000) == 0:
                writer = tf.python_io.TFRecordWriter(
                    "/home/cyx/AdaIN-Fashion/dataset/{}-{}.tfrecord".format(mode, counter))
                counter += 1

            idx -= 1
            if not self.is_vaild(self.description[idx]):
                continue
            img_raw = self.images_data[idx]
            img_shape = img_raw.shape
            img_raw = img_raw.tobytes()
            img_mean = np.array(self.image_mean).tobytes()
            desc = self.description[idx]
            desc_len = len(desc)
            embedding, embedding_shape= self.get_embedding(desc)
            embedding = embedding.tobytes()
            desc = str.encode(' '.join(desc))

            feature ={
                "img": _bytes_feature(img_raw),
                "img_mean": _bytes_feature(img_mean),
                "img_shape": _int64_feature(img_shape),
                "desc": _bytes_feature(desc),
                "desc_len": _int64_feature(desc_len),
                "embedding": _bytes_feature(embedding),
                "embedding_shape": _int64_feature(embedding_shape)
            }

            example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
            writer.write(example_proto.SerializeToString())
            total_num += 1
            if np.mod(i, 5000) == 4999:
                writer.close()

    def load_img(self):
        print('loading image data...')
        h5f = h5py.File(self.h5f_path, 'r')
        self.images_data = h5f['ih']
        self.images_data = np.transpose(self.images_data, [2, 1, 0])
        self.image_mean = h5f['ih_mean']
        print('complete.')

    def load_idx(self):
        print('loading index matrix...')
        ind = scio.loadmat(self.idx_path)
        self.train_idx = [item[0] for item in ind['train_ind']]
        self.test_idx = [s[0] for s in ind['test_ind']]
        self.pair_idx = ind['test_set_pair_ind']
        print('complete.')

    def load_desc(self):
        description_raw = scio.loadmat(self.desc_path)
        description_raw = description_raw['engJ']
        for i in range(len(description_raw)):
            description = description_raw[i][0][0].strip('.').split()
            items = []
            for item in description:
                if ',' in item:
                    item = item.strip(',')
                if '-' in item:
                    item = item.split('-')
                if '_' in item:
                    item = item.split('_')
                if type(item) is list:
                    items += item
                else:
                    items.append(item)
            self.description.append(items)

    def load_embed(self):
        with open(self.embed_path, 'rb') as f:
            for line in f:
                values = line.split()
                word = str(values[0], encoding='utf-8')
                coefs = np.asarray(values[1:], dtype='float32')
                self.embedding[word] = coefs

    def is_vaild(self, desc):
        for word in desc:
            if word not in self.embedding:
                return False
        return True

    def get_embedding(self, desc):
        embedding = np.zeros([18, 300])
        for i, item in enumerate(desc):
            embedding[i, :] = self.embedding[item]
        return embedding, embedding.shape


def main():
    h5f_path = '/home/cyx/AdaIN-Fashion/dataset/supervision_signals/G2.h5'
    idx_path = '/home/cyx/AdaIN-Fashion/dataset/benchmark/ind.mat'
    desc_path = '/home/cyx/AdaIN-Fashion/dataset/benchmark/language_original.mat'
    embed_path = '/home/cyx/AdaIN-Fashion/dataset/benchmark/wiki-news-300d-1M.vec'
    CreateTFRecord(h5f_path, idx_path, desc_path, embed_path)


if __name__ == '__main__':
    main()
