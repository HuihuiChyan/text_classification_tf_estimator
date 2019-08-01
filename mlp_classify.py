import os
import tensorflow as tf
import random
from collections import defaultdict
from collections import Counter
import pdb
import os
import numpy as np
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
FLAGS = tf.flags.FLAGS
tf.flags.DEFINE_string("data_dir", default="rt-polaritydata", help="Just for test.")
tf.flags.DEFINE_string("pos_file", default="rt-polaritydata/rt-polarity.pos", help="Just for test.")
tf.flags.DEFINE_string("neg_file", default="rt-polaritydata/rt-polarity.neg", help="Just for test.")
tf.flags.DEFINE_string("pos_train", default="rt-polaritydata/train.pos", help="Just for test.")
tf.flags.DEFINE_string("neg_train", default="rt-polaritydata/train.neg", help="Just for test.")
tf.flags.DEFINE_string("pos_dev", default="rt-polaritydata/dev.pos", help="Just for test.")
tf.flags.DEFINE_string("neg_dev", default="rt-polaritydata/dev.neg", help="Just for test.")
tf.flags.DEFINE_string("vocab_file", default="rt-polaritydata/vocab.txt", help="Just for test.")
tf.flags.DEFINE_integer("vocab_size", default=10000, help="Just for test.")
tf.flags.DEFINE_integer("batch_size", default=64, help="Just for test.")
tf.flags.DEFINE_string("model_dir", default="model_dir", help="Just for test.")
tf.flags.DEFINE_bool("is_training", default=True, help="Just for test.")
tf.flags.DEFINE_integer("max_train_step", default=10000, help="Just for test.")
tf.flags.DEFINE_integer("step_per_eval", default=500, help="Just for test.")
tf.flags.DEFINE_float("learning_rate", default=1e-3, help="Just for test.")
tf.flags.DEFINE_integer("max_len", default=50, help="Just for test.")
tf.flags.DEFINE_integer("embedding_size", default=512, help="The embedding size.")
tf.flags.DEFINE_integer("hidden_size", default=512, help="The hidden size for RNN cell.")
tf.flags.DEFINE_float("upper_grad", default=1.0, help="Upper bound for grad clip.")
tf.flags.DEFINE_float("lower_grad", default=1e-10, help="Lower bound for grad clip.")
tf.flags.DEFINE_integer("step_per_stat", default=100, help="Step number to print the loss value.")
tf.flags.DEFINE_integer("class_num", default=2, help="Number of classes.")


def train_dev_split(pos_file, neg_file, pos_train, neg_train, pos_dev, neg_dev):
	with open(pos_file, "r", encoding="utf-8") as fpos,\
	open(neg_file, "r", encoding="utf-8") as fneg,\
	open(pos_train, "w", encoding="utf-8") as fpostrain,\
	open(neg_train, "w", encoding="utf-8") as fnegtrain,\
	open(pos_dev, "w", encoding="utf-8") as fposdev,\
	open(neg_dev, "w", encoding="utf-8") as fnegdev:
		poslines = [line.strip() for line in fpos.readlines()]
		neglines = [line.strip() for line in fneg.readlines()]
		random.shuffle(poslines)
		random.shuffle(neglines)
		posdevlines = poslines[:500]
		negdevlines = neglines[:500]
		postrainlines = poslines[500:]
		negtrainlines = neglines[500:]
		for line in posdevlines:
			fposdev.write(line+"\n")
		for line in negdevlines:
			fnegdev.write(line+"\n")
		for line in postrainlines:
			fpostrain.write(line+"\n")
		for line in negtrainlines:
			fnegtrain.write(line+"\n")

class InputExample(object):
	def __init__(self, text, label):
		self.text = text
		self.label = label

class DataProcessor(object):
	def get_train_examples(self, data_dir):
		examples = []
		with open(os.path.join(data_dir, "train.pos"), "r", encoding="utf-8") as fpos,\
		open(os.path.join(data_dir, "train.neg"), "r", encoding="utf-8") as fneg:
			poslines = [line.strip() for line in fpos.readlines()]
			neglines = [line.strip() for line in fneg.readlines()]
			pos_num = 0
			neg_num = 0
			for line in poslines:
				example = InputExample(text=line,label="pos")
				examples.append(example)
				pos_num += 1
			for line in neglines:
				example = InputExample(text=line,label="neg")
				examples.append(example)
				neg_num += 1
			random.shuffle(examples)
		return examples
	def get_dev_examples(self, data_dir):
		examples = []
		with open(os.path.join(data_dir, "train.pos"), "r", encoding="utf-8") as fpos,\
		open(os.path.join(data_dir, "train.neg"), "r", encoding="utf-8") as fneg:
			poslines = [line.strip().split() for line in fpos.readlines()]
			neglines = [line.strip().split() for line in fneg.readlines()]
			for line in poslines:
				example = InputExample(text=line,label="pos")
				examples.append(example)
			for line in neglines:
				example = InputExample(text=line,label="neg")
				examples.append(example)
			random.shuffle(examples)
		return examples

class InputFeature(object):
	def __init__(self, input_ids, label_id, text_len):
		self.input_ids = input_ids
		self.label_id = label_id 
		self.text_len = text_len

class Tokenizer(object):
	def __init__(self, vocab_file):
		def return_zero():
			return 0
		def return_unk():
			return '<unk>'
		self.vocab = defaultdict(return_zero)
		self.idx_vocab = defaultdict(return_unk)
		with open(vocab_file, "r", encoding="utf-8") as fvocab:
			vocablines = [line.strip() for line in fvocab.readlines()]
			for i,line in enumerate(vocablines):
				self.vocab[line] = i
				self.idx_vocab[i] = line

	def tokenize(self, text_line):
		idx_line = []
		for word in text_line:
			idx_line.append(self.vocab[word])
		return idx_line

	def retokenize(self, idx_line):
		text_line = []
		for idx in idx_line:
			if idx != 1: # 1 for <pad>
				text_line.append(self.idx_vocab[idx])
		return text_line


def check_and_build_vocab(pos_train, neg_train, vocab_file):
	if tf.gfile.Exists(vocab_file):
		print("Vocabulary file already exist!")
	else:
		with open(pos_train, "r", encoding="utf-8") as fpos,\
		open(neg_train, "r", encoding="utf-8") as fneg,\
		open(vocab_file, "w", encoding="utf-8") as fvocab:
			poslines = [line.strip().split() for line in fpos.readlines()]
			neglines = [line.strip().split() for line in fneg.readlines()]
			lines = poslines + neglines
			words = []
			for line in lines:
				words.extend(line)
			counter = Counter(words)
			print("Totally %d words and we choose %d words." % (len(list(counter)), FLAGS.vocab_size))
			vocab_words = counter.most_common(FLAGS.vocab_size-2)
			fvocab.write("<unk>\n")
			fvocab.write("<pad>\n")
			for word in vocab_words:
				fvocab.write(word[0]+"\n")


def convert_single_example(example, tokenizer, max_len):
	text_len = len(example.text)
	idxes = tokenizer.tokenize(example.text)
	if len(idxes) < max_len:
		idxes = idxes + [1 for _ in range(max_len-len(idxes))]
	elif len(idxes) > max_len:
		idxes = idxes[:max_len]
	if example.label == 'pos':
		label_id = [0,1]
	else:
		label_id = [1,0]
	return InputFeature(input_ids=idxes, label_id=label_id, text_len=text_len)


def convert_examples_to_features(examples, tokenizer, max_len):
	features = []
	for example in examples:
		feature = convert_single_example(example, tokenizer, max_len)
		features.append(feature)
	return features

def convert_features_to_iterator(features, is_training=False, drop_remainder=True):
	num_examples = len(features)
	all_input_ids = []
	all_label_id = []
	all_text_len = []
	for feature in features:
		all_input_ids.append(feature.input_ids)
		all_label_id.append(feature.label_id)
		all_text_len.append(feature.text_len)
	batch_size = FLAGS.batch_size
	dataset = tf.data.Dataset.from_tensor_slices({
			"input_ids": tf.constant(all_input_ids, shape=[num_examples, FLAGS.max_len]),
			"input_label": tf.constant(all_label_id, shape=[num_examples, 2]),
			"text_len": tf.constant(all_text_len, shape=[num_examples])
		})	
	if is_training:
		dataset = dataset.repeat()
	dataset = dataset.batch(FLAGS.batch_size)
	iterator = dataset.make_initializable_iterator()
	return iterator

class Classifier(object):
	def __init__(self, iterator, reuse):
		self.features = iterator.get_next()
		with tf.variable_scope("my_variable_scope", reuse=reuse):
			embedding_table = tf.get_variable(name="embed_table", initializer=tf.contrib.layers.xavier_initializer(), shape=[FLAGS.vocab_size, FLAGS.embedding_size])
			input_embeddings = tf.nn.embedding_lookup(embedding_table, self.features["input_ids"])
			# fwLstmCell = tf.nn.rnn_cell.BasicLSTMCell(num_units=FLAGS.hidden_size, forget_bias=0.0, state_is_tuple=True, reuse=reuse)
			# fwLstmCell = tf.nn.rnn_cell.DropoutWrapper(cell=fwLstmCell, output_keep_prob=0.7)
			# bwLstmCell = tf.nn.rnn_cell.BasicLSTMCell(num_units=FLAGS.hidden_size, forget_bias=0.0, state_is_tuple=True, reuse=reuse)
			# bwLstmCell = tf.nn.rnn_cell.DropoutWrapper(cell=bwLstmCell, output_keep_prob=0.7)
			# bilstm_output, _ = tf.nn.bidirectional_dynamic_rnn(cell_fw=fwLstmCell,
			# 											cell_bw=bwLstmCell,
			# 											inputs=input_embeddings,
			# 											sequence_length=self.features["text_len"],
			# 											dtype=tf.float32)
			# fw_output, bw_output = bilstm_output
			# fw_output = tf.transpose(fw_output, [1,0,2])
			# bw_output = tf.transpose(bw_output, [1,0,2])
			# bidi_output = tf.concat([fw_output[-1], bw_output[-1]], axis=-1)
			input_reshape = tf.reshape(input_embeddings, [FLAGS.batch_size, FLAGS.embedding_size*FLAGS.max_len])
			middle_layer = tf.layers.dense(input_reshape, FLAGS.hidden_size, reuse=reuse, name="middle")
			self.output = tf.layers.dense(middle_layer, 2, reuse=reuse, name="output")
			self.prediction_result = tf.cast(tf.argmax(self.output, axis=-1), tf.int32)
			self.global_step = tf.train.get_or_create_global_step()

			per_example_loss = tf.nn.softmax_cross_entropy_with_logits(logits=self.output, labels=self.features["input_label"])
			self.loss = tf.reduce_mean(per_example_loss)
			optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate)
			# grads_and_vars = optimizer.compute_gradients(loss)
			# capped_grads = [(tf.clip_by_value(grad, FLAGS.lower_grad, FLAGS.upper_grad),var) for (grad, var) in grads_and_vars]
			self.train_op = optimizer.minimize(self.loss, global_step=self.global_step)

			self.label_logits = tf.argmax(self.features['input_label'], axis=-1)
			self.accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.cast(self.label_logits, tf.int32), self.prediction_result),tf.float32))
	
def main(_):
	vocab_path = os.path.join(FLAGS.vocab_file)
	if tf.gfile.Exists(FLAGS.pos_train):
		print("Train and eval file already splitted!")
	else:
		train_dev_split(FLAGS.pos_file, FLAGS.neg_file, FLAGS.pos_train, FLAGS.neg_train, FLAGS.pos_dev, FLAGS.neg_dev)
	dataProcessor = DataProcessor()
	train_examples = dataProcessor.get_train_examples(FLAGS.data_dir)
	dev_examples = dataProcessor.get_dev_examples(FLAGS.data_dir)
	check_and_build_vocab(FLAGS.pos_train, FLAGS.neg_train, FLAGS.vocab_file)
	tokenizer = Tokenizer(FLAGS.vocab_file)
	train_features = convert_examples_to_features(train_examples, tokenizer, FLAGS.max_len)
	dev_features = convert_examples_to_features(dev_examples, tokenizer, FLAGS.max_len)
	current_train_step = 0
	train_iterator = convert_features_to_iterator(train_features, is_training=True, drop_remainder=True)
	eval_iterator = convert_features_to_iterator(dev_features, drop_remainder=True)

	train_model = Classifier(train_iterator, reuse=False)
	eval_model = Classifier(eval_iterator, reuse=True)
	with tf.Session() as sess:
		sess.run(tf.initializers.global_variables())
		sess.run(tf.tables_initializer())
		sess.run(train_iterator.initializer)
		current_train_step = 1
		while current_train_step<=FLAGS.max_train_step:
			if current_train_step%FLAGS.step_per_eval==0:
				sess.run(eval_iterator.initializer)
				acc_list = []
				while(1):
					try:
						acc_single = sess.run(eval_model.accuracy)
						acc_list.append(acc_single)
					except:
						acc = np.mean(acc_list)
						break
				# pdb.set_trace()
				print("The following is an evaluation:")
				print("For train_step %d, the accuracy is %s" % (current_train_step, str(acc)))
				# features_value = sess.run(eval_model.features)
				# eval_text_id = features_value["input_ids"]
				# eval_label = features_value["input_label"]
				# eval_text = [tokenizer.retokenize(eval_text_index) for eval_text_index in eval_text_id]
				# eval_input = list(zip(eval_text, eval_label))
			_, loss_value = sess.run((train_model.train_op, train_model.loss))
			if current_train_step%FLAGS.step_per_stat==0:
				print("For train_step %d, the loss is %s" % (current_train_step, str(loss_value)))
			current_train_step += 1
if __name__=="__main__":
	tf.app.run()