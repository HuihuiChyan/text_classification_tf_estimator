import os
import tensorflow as tf
import random
from collections import defaultdict
from collections import Counter
import pdb
import os
from tensorflow.python import debug as tf_debug
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
FLAGS = tf.flags.FLAGS

tf.flags.DEFINE_string("vocab_file", default="rt-polaritydata/vocab.txt", help="Just for test.")
tf.flags.DEFINE_string("pos_file", default="rt-polaritydata/rt-polarity.pos", help="Just for test.")
tf.flags.DEFINE_string("neg_file", default="rt-polaritydata/rt-polarity.neg", help="Just for test.")
tf.flags.DEFINE_string("pos_train", default="rt-polaritydata/train.pos", help="Just for test.")
tf.flags.DEFINE_string("neg_train", default="rt-polaritydata/train.neg", help="Just for test.")
tf.flags.DEFINE_string("pos_dev", default="rt-polaritydata/dev.pos", help="Just for test.")
tf.flags.DEFINE_string("neg_dev", default="rt-polaritydata/dev.neg", help="Just for test.")
tf.flags.DEFINE_string("model_dir", default="model_dir", help="Just for test.")

tf.flags.DEFINE_integer("max_train_step", default=10000, help="Just for test.")
tf.flags.DEFINE_integer("step_per_eval", default=100, help="Just for test.")
tf.flags.DEFINE_integer("save_checkpoints_steps", default=100, help="Step number for save checkpoint.")
tf.flags.DEFINE_integer("step_per_stat", default=50, help="Step number for training state.")

tf.flags.DEFINE_bool("is_training", default=False, help="Just for test.")
tf.flags.DEFINE_integer("vocab_size", default=10000, help="Just for test.")
tf.flags.DEFINE_integer("batch_size", default=64, help="Just for test.")
tf.flags.DEFINE_float("learning_rate", default=1e-3, help="Just for test.")
tf.flags.DEFINE_integer("max_len", default=50, help="Just for test.")
tf.flags.DEFINE_integer("embedding_size", default=512, help="The embedding size.")
tf.flags.DEFINE_integer("hidden_size", default=512, help="The hidden size for RNN cell.")
tf.flags.DEFINE_float("upper_grad", default=1.0, help="Upper bound for grad clip.")
tf.flags.DEFINE_float("lower_grad", default=1e-10, help="Lower bound for grad clip.")


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

def input_fn_builder(features, is_training=False, drop_remainder=True):
	num_examples = len(features)
	all_input_ids = []
	all_label_id = []
	all_text_len = []
	for feature in features:
		all_input_ids.append(feature.input_ids)
		all_label_id.append(feature.label_id)
		all_text_len.append(feature.text_len)
	def input_fn(params):
		batch_size = FLAGS.batch_size
		feature_dataset = tf.data.Dataset.from_tensor_slices({
				"input_ids": tf.constant(all_input_ids, shape=[num_examples, FLAGS.max_len]),
				"text_len": tf.constant(all_text_len, shape=[num_examples])})	
		label_dataset = tf.data.Dataset.from_tensor_slices(tf.constant(all_label_id, shape=[num_examples, 2]))
		dataset = tf.data.Dataset.zip((feature_dataset, label_dataset))
		if is_training == True:
			dataset = dataset.shuffle(buffer_size=1000)
			dataset = dataset.repeat()
		dataset = dataset.batch(FLAGS.batch_size, drop_remainder=drop_remainder)
		# iterator = dataset.make_initializable_iterator()
		# return iterator.get_next()
		return dataset
	return input_fn

def predict_input_fn_builder(tokenizer):
	def feature_generator():
		while(1):
			input_line = input("Please input your text. If you want to stop, please input q or quit:")
			if input_line == "q" or input_line == "quit":
				break
			input_text = input_line.strip().split()
			input_len = len(input_text)
			input_id = tokenizer.tokenize(input_text)
			if input_len < FLAGS.max_len:
				new_input_id = input_id + [1 for _ in range(FLAGS.max_len-input_len)]
			elif input_len > FLAGS.max_len:
				new_input_id = input_id[:FLAGS.max_len]
			input_feature = (new_input_id, input_len)
			yield input_feature
	# generator = generator_creator()
	# pdb.set_trace()
	def label_generator():
		output = [0, 1]
		while(1):
			yield output

	def predict_input_fn(params):
		feature_dataset = tf.data.Dataset.from_generator(feature_generator, 
												(tf.int32, tf.int32), 
												output_shapes=(tf.TensorShape([FLAGS.max_len]), tf.TensorShape([])))
		feature_dataset = feature_dataset.map(lambda ele1, ele2:{"input_ids":ele1,"text_len":ele2})
		label_dataset = tf.data.Dataset.from_generator(label_generator, tf.int32, output_shapes=tf.TensorShape([2]))
		dataset = tf.data.Dataset.zip((feature_dataset, label_dataset))
		dataset = dataset.batch(1)
		# pdb.set_trace()
		# dataset.repeat()
		# dataset = dataset.batch(1)
		# iterator = dataset.make_one_shot_iterator()
		# iterator = iterator.get_next()
		# pdb.set_trace()
		return dataset
	return predict_input_fn

class InputFeature(object):
	def __init__(self, input_ids, label_id, text_len):
		self.input_ids = input_ids
		self.label_id = label_id 
		self.text_len = text_len

def convert_file_to_features(posfile, negfile, tokenizer, is_training=True):
	with open(posfile, "r", encoding="utf-8") as fpos,\
	open(negfile, "r", encoding="utf-8") as fneg:
		poslines = [line.strip().split() for line in fpos.readlines()]
		neglines = [line.strip().split() for line in fneg.readlines()]
		poslines = [(tokenizer.tokenize(line), len(line) if len(line)<=FLAGS.max_len else FLAGS.max_len) for line in poslines]
		neglines = [(tokenizer.tokenize(line), len(line) if len(line)<=FLAGS.max_len else FLAGS.max_len) for line in neglines]
		padded_poslines = []
		padded_neglines = []
		for line in poslines:
			if len(line[0]) < FLAGS.max_len:
				newline = line[0] + [1 for _ in range(FLAGS.max_len-len(line[0]))]
			elif len(line[0]) > FLAGS.max_len:
				newline = line[0][:FLAGS.max_len]
			padded_poslines.append((newline, [0,1], line[1]))
		for line in neglines:
			if len(line[0]) < FLAGS.max_len:
				newline = line[0] + [1 for _ in range(FLAGS.max_len-len(line[0]))]
			elif len(line[0]) > FLAGS.max_len:
				newline = line[0][:FLAGS.max_len]
			padded_poslines.append((newline, [1,0], line[1]))
		padded_lines = padded_poslines+padded_neglines
		random.shuffle(padded_lines)
		features = []
		for line in padded_lines:
			features.append(InputFeature(line[0], line[1], line[2]))
		return features

def model_fn_builder():
	def model_fn(features, labels, mode, params):
		embedding_table = tf.get_variable(name="embed_table", initializer=tf.contrib.layers.xavier_initializer(), shape=[FLAGS.vocab_size, FLAGS.embedding_size])
		input_embeddings = tf.nn.embedding_lookup(embedding_table, features["input_ids"])

		fwLstmCell = tf.nn.rnn_cell.BasicLSTMCell(num_units=FLAGS.hidden_size, forget_bias=0.7)
		fwLstmCell = tf.nn.rnn_cell.DropoutWrapper(cell=fwLstmCell, output_keep_prob=0.7)
		bwLstmCell = tf.nn.rnn_cell.BasicLSTMCell(num_units=FLAGS.hidden_size, forget_bias=0.7)
		bwLstmCell = tf.nn.rnn_cell.DropoutWrapper(cell=bwLstmCell, output_keep_prob=0.7)
		_, bilstm_state = tf.nn.bidirectional_dynamic_rnn(cell_fw=fwLstmCell,
													cell_bw=bwLstmCell,
													inputs=input_embeddings,
													sequence_length=features["text_len"], 
													dtype=tf.float32)
		
		fw_state = bilstm_state[0][1]
		bw_state = bilstm_state[1][1]
		bidi_output = tf.concat([fw_state, bw_state], axis=-1)

		output = tf.layers.dense(bidi_output, 2)
		prediction_result = tf.cast(tf.argmax(output, axis=-1), tf.int32)
		global_step = tf.train.get_or_create_global_step()

		if mode == tf.estimator.ModeKeys.TRAIN:
			
			
			loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=output, labels=labels))
			optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate)
			# grads_and_vars = optimizer.compute_gradients(loss)
			# capped_grads = [(tf.clip_by_value(grad, FLAGS.lower_grad, FLAGS.upper_grad),var) for (grad, var) in grads_and_vars]
			train_op = optimizer.minimize(loss, global_step=global_step)
			output_spec = tf.contrib.tpu.TPUEstimatorSpec(
				mode=mode,
				loss=loss,
				train_op=train_op)

		elif mode == tf.estimator.ModeKeys.EVAL:
			loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=output, labels=labels))
			def metric_fn(prediction_result, label_logits):				
				accuracy = tf.metrics.accuracy(prediction_result, label_logits)
				return {
					"eval_accuracy": accuracy
				}
			label_logits = tf.argmax(labels, axis=-1)
			eval_metrics = (metric_fn, [prediction_result, label_logits])
			output_spec = tf.contrib.tpu.TPUEstimatorSpec(
				mode=mode,
				loss=loss,
				eval_metrics=eval_metrics)

		elif mode == tf.estimator.ModeKeys.PREDICT:
			output_spec = tf.contrib.tpu.TPUEstimatorSpec(
				mode=mode,
				predictions={"classification_result":prediction_result})
		return output_spec
	return model_fn

def main(_):
	vocab_path = os.path.join(FLAGS.vocab_file)
	if tf.gfile.Exists(FLAGS.pos_train):
		print("Train and eval file already splitted!")
	else:
		train_dev_split(FLAGS.pos_file, FLAGS.neg_file, FLAGS.pos_train, FLAGS.neg_train, FLAGS.pos_dev, FLAGS.neg_dev)
	check_and_build_vocab(FLAGS.pos_train, FLAGS.neg_train, FLAGS.vocab_file)
	tokenizer = Tokenizer(FLAGS.vocab_file)
	train_features = convert_file_to_features(FLAGS.pos_train, FLAGS.neg_train, tokenizer, is_training=True)
	dev_features = convert_file_to_features(FLAGS.pos_dev, FLAGS.neg_dev, tokenizer, is_training=True)

	model_fn = model_fn_builder()
	run_config = tf.contrib.tpu.RunConfig(
		model_dir=FLAGS.model_dir,
		save_checkpoints_steps=FLAGS.save_checkpoints_steps,
		log_step_count_steps=FLAGS.step_per_stat)

	estimator = tf.contrib.tpu.TPUEstimator(use_tpu=False,
											model_fn=model_fn,
											config=run_config,
											train_batch_size=FLAGS.batch_size,
											eval_batch_size=FLAGS.batch_size,
											predict_batch_size=1)
	current_train_step = 0
	if FLAGS.is_training == True:
		train_input_fn = input_fn_builder(train_features, is_training=True, drop_remainder=True)
		eval_input_fn = input_fn_builder(dev_features, is_training=False, drop_remainder=False)
		estimator.evaluate(input_fn=eval_input_fn)
		# print("Step %d, the eval result is:" % (current_train_step))
		# print(result)
		while current_train_step <= FLAGS.max_train_step:
			current_train_step += FLAGS.step_per_eval
			estimator.train(input_fn=train_input_fn, steps=FLAGS.step_per_eval)	
			estimator.evaluate(input_fn=eval_input_fn)
			# print("Step %d, the eval result is:" % (current_train_step))
			# print(result)
	else:
		predict_input_fn = predict_input_fn_builder(tokenizer)
		# hooks = [tf_debug.LocalCLIDebugHook()]
		predict_result = estimator.predict(input_fn=predict_input_fn)

		while(1):
			result = next(predict_result)["classification_result"]
			if result == 1:
				print("Positive")
			else:
				print("Negative")

		# for prediction in enumerate(predict_result):
		# 	print(prediction)		

if __name__=="__main__":
	tf.logging.set_verbosity(tf.logging.DEBUG)
	tf.app.run()
