package yuncong;

import java.io.IOException;
import java.util.Iterator;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.conf.Configured;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.BytesWritable;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapred.FileInputFormat;
import org.apache.hadoop.mapred.FileOutputFormat;
import org.apache.hadoop.mapred.JobClient;
import org.apache.hadoop.mapred.JobConf;
import org.apache.hadoop.mapred.MapReduceBase;
import org.apache.hadoop.mapred.Mapper;
import org.apache.hadoop.mapred.OutputCollector;
import org.apache.hadoop.mapred.Reducer;
import org.apache.hadoop.mapred.Reporter;
import org.apache.hadoop.mapred.WordCount;
import org.apache.hadoop.util.Tool;
import org.apache.hadoop.util.ToolRunner;

public class WholeFileOutputter extends Configured implements Tool {
	public static class MapClass extends MapReduceBase implements
			Mapper<Text, BytesWritable, Text, IntWritable> {

		private final static IntWritable one = new IntWritable(1);
		private Text word = new Text();

		public void map(Text key, BytesWritable value,
				OutputCollector<Text, IntWritable> output, Reporter reporter)
				throws IOException {
			String key_str = key.toString();
			word.set(key_str);
			output.collect(word, one);
		}
	}

	/**
	 * A reducer class that just emits the sum of the input values.
	 */
	public static class ReduceClass extends MapReduceBase implements
			Reducer<Text, IntWritable, Text, IntWritable> {

		public void reduce(Text key, Iterator<IntWritable> values,
				OutputCollector<Text, IntWritable> output, Reporter reporter)
				throws IOException {
			int sum = 0;
			while (values.hasNext()) {
				sum += values.next().get();
			}
			output.collect(key, new IntWritable(sum));
		}
	}

	@Override
	public int run(String[] args) throws Exception {
		JobClient client = new JobClient();
		JobConf conf = new JobConf(WholeFileOutputter.class);

		conf.setJobName("WholeFileOutputter");
		
		conf.setInputFormat(WholeFileInputFormatOld.class);
	
		conf.setOutputKeyClass(Text.class);
		conf.setOutputValueClass(IntWritable.class);

		FileInputFormat.addInputPath(conf, new Path(args[0]));
		FileOutputFormat.setOutputPath(conf, new Path(args[1]));

		conf.setMapperClass(MapClass.class);
		conf.setReducerClass(ReduceClass.class);

		client.setConf(conf);

		try {
			JobClient.runJob(conf);
		} catch (Exception e) {
			e.printStackTrace();
		}
		return 0;

	}

	public static void main(String[] args) throws Exception {
		int exitCode = ToolRunner.run(new WholeFileOutputter(), args);
		System.exit(exitCode);
	}
}
