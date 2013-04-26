package yuncong;

import java.io.IOException;

import org.apache.hadoop.mapred.lib.MultipleOutputs;

static class MultipleOutputsReducer extends
		Reducer<Text, Text, NullWritable, Text> {
	private MultipleOutputs<NullWritable, Text> multipleOutputs;

	@Override
	protected void setup(Context context) throws IOException,
			InterruptedException {
		multipleOutputs = new MultipleOutputs<NullWritable, Text>(context);
	}

	@Override
	protected void reduce(Text key, Iterable<Text> values, Context context)
			throws IOException, InterruptedException {
		for (Text value : values) {
			multipleOutputs.write(NullWritable.get(), value, key.toString());
		}
	}

	@Override
	protected void cleanup(Context context) throws IOException,
			InterruptedException {
		multipleOutputs.close();
	}
}