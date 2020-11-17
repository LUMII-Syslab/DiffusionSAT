from abc import abstractmethod
from pathlib import Path

from config import Config
import shutil


def elements_to_str(inputs: iter):
    return [str(x) for x in inputs]


class DIMACDataset:

    @abstractmethod
    def dimac_generator(self) -> tuple:
        """
        Generator function (instead of return use yield), that returns single instance to be writen in DIMACS file
        :return: tuple(variable_count: int, clauses: list of tuples)
        """
        pass

    @abstractmethod
    def dimac_to_data(self):
        pass

    @abstractmethod
    def write_dimacs_to_file(self):
        output_folder = Path(Config.data_dir) / self.__class__.__name__

        if Config.force_data_gen and output_folder.exists():
            shutil.rmtree(output_folder)

        if output_folder.exists():
            print("Not recreating data, as folder already exists")
            return
        else:
            output_folder.mkdir(parents=True)

        print(f"Generating data in '{output_folder}' directory!")
        for idx, (n_vars, clauses) in enumerate(self.dimac_generator()):
            clauses = [elements_to_str(c) for c in clauses]
            file = [f"p cnf {n_vars} {len(clauses)}"]
            file += [f"{' '.join(c)} 0" for c in clauses]

            out_filename = output_folder / f"example_{idx}.dimacs"
            with open(out_filename, 'w') as f:
                f.write('\n'.join(file))

#
# class GeneratorDataset(Dataset):
#     def train_dataset(self) -> list:
#         return self.create_file_based_dataset("train", self.train_output_shapes, self.train_size, training=True)
#
#     def create_file_based_dataset(self, file_prefix: str, output_shapes: list, dataset_size, training):
#         data_dir = Path(config.data_dir) / self.__class__.__name__.lower()
#
#         if not data_dir.exists():
#             data_dir.mkdir(parents=True)
#
#         datasets = []
#         for feature_sh, label_sh in output_shapes:
#             f_sh_str = "x".join([str(x) for x in feature_sh])
#             file_name = f"{file_prefix}_{f_sh_str}.tfrecord"
#             file_name = data_dir / file_name
#
#             data = self.dataset_from_file(file_name, feature_sh, label_sh, dataset_size, training)
#
#             datasets.append(data)
#         return datasets
#
#     def dataset_from_file(self, file_name: Path, feature_sh: tuple,
#                           label_sh: tuple, dataset_size, training: bool) -> tf.data.TFRecordDataset:
#
#         if file_name.exists() and config.force_file_generation:
#             file_name.unlink()
#
#         if not file_name.exists():
#             generator = self.generator_fn(feature_sh, label_sh, training=training)
#             generator = itertools.islice(generator(), dataset_size)
#             options = tf.io.TFRecordOptions(tf.compat.v1.io.TFRecordCompressionType.GZIP)
#
#             with tf.io.TFRecordWriter(str(file_name), options) as tfwriter:
#                 for feature, label in generator:
#                     example = self.create_example(feature, label)
#                     tfwriter.write(example.SerializeToString())
#
#         data = tf.data.TFRecordDataset([str(file_name)], "GZIP")
#         return data.map(lambda rec: self.extract(rec, feature_sh, label_sh), tf.data.experimental.AUTOTUNE)
#
#     @staticmethod
#     def extract(data_record, feature_shape, label_shape):
#         parsed = tf.io.parse_single_example(data_record, {
#             'feature': tf.io.VarLenFeature(tf.int64),
#             'label': tf.io.VarLenFeature(tf.int64),
#         })
#
#         feature = tf.reshape(parsed["feature"].values, feature_shape)  # type: tf.Tensor
#         label = tf.reshape(parsed["label"].values, label_shape)  # type: tf.Tensor
#
#         feature.set_shape(feature_shape)
#         label.set_shape(label_shape)
#
#         feature = tf.cast(feature, tf.int32)
#         label = tf.cast(label, tf.int32)
#
#         return feature, label
#
#     @staticmethod
#     def create_example(feature, label):
#         feature = tf.train.Int64List(value=feature.flatten())
#         label = tf.train.Int64List(value=label.flatten())
#
#         example_map = {
#             'feature': tf.train.Feature(int64_list=feature),
#             'label': tf.train.Feature(int64_list=label),
#         }
#
#         features = tf.train.Features(feature=example_map)
#         return tf.train.Example(features=features)
#
#     def eval_dataset(self) -> list:
#         return self.create_file_based_dataset("eval", self.eval_output_shapes, self.eval_size, training=False)
#
#     def create_dataset(self, feature_sh, label_sh):
#         return tf.data.Dataset.from_generator(
#             self.generator_fn(feature_sh, label_sh, training=False),
#             self.generator_output_types,
#             output_shapes=(tf.TensorShape(feature_sh), tf.TensorShape(label_sh))
#         )
