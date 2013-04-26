package yuncong;

import java.io.IOException;

import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.BytesWritable;
import org.apache.hadoop.io.NullWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapred.FileInputFormat;
import org.apache.hadoop.mapred.FileSplit;
//import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapred.InputSplit;
import org.apache.hadoop.mapred.JobConf;
import org.apache.hadoop.mapred.JobContext;
import org.apache.hadoop.mapred.RecordReader;
import org.apache.hadoop.mapred.Reporter;

public class WholeFileInputFormatOld extends FileInputFormat<Text, Text> {
	@Override
	protected boolean isSplitable(FileSystem fs, Path file) {
		return false;
	}

	@Override
	public RecordReader<Text, Text> getRecordReader(
			InputSplit split, JobConf job, Reporter reporter) throws IOException {
	    reporter.setStatus(split.toString());
	    return new WholeFileRecordReaderOld(job, (FileSplit)split);

//		WholeFileRecordReader reader = new WholeFileRecordReader(job, null);
//		reader.initialize(split, context);
//		return reader;
	}

}