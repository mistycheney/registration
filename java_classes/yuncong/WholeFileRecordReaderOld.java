package yuncong;

import java.io.IOException;

import java.io.InputStream;
import java.math.BigInteger;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FSDataInputStream;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.ByteWritable;
import org.apache.hadoop.io.BytesWritable;
import org.apache.hadoop.io.IOUtils;
import org.apache.hadoop.io.NullWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.io.compress.CompressionCodec;
import org.apache.hadoop.io.compress.CompressionCodecFactory;
import org.apache.hadoop.mapred.FileSplit;
import org.apache.hadoop.mapred.InputSplit;
import org.apache.hadoop.mapred.JobConf;
import org.apache.hadoop.mapred.RecordReader;
import org.apache.hadoop.mapred.Reporter;

import org.apache.commons.codec.binary.Base64;

class WholeFileRecordReaderOld implements RecordReader<Text, Text> {	
	private FileSplit fileSplit;
	private JobConf conf;
	private boolean processed = false;
	private long pos = 0;

	public WholeFileRecordReaderOld(JobConf job, FileSplit split) throws IOException {
	    conf = job;
	    fileSplit = split;
	}
	
//	public String toHex(String arg) {
//		  return String.format("%x", new BigInteger(1, arg.getBytes(/*YOUR_CHARSET?*/)));
//		}
	
	@Override
	public boolean next(Text key, Text value) throws IOException {
		if (!processed) {
			byte[] contents = new byte[(int) fileSplit.getLength()];
			Path file = fileSplit.getPath();
			key.set(file.toString());

			FileSystem fs = file.getFileSystem(conf);
			FSDataInputStream in = null;
			try {
				in = fs.open(file);
				IOUtils.readFully(in, contents, 0, contents.length);
				byte[] contents64 = Base64.encodeBase64(contents);
				value.set(contents64, 0, contents64.length);
			} finally {
				IOUtils.closeStream(in);
			}
			processed = true;
			pos = contents.length;
			return true;
		}
		return false;
	}

	@Override
	public float getProgress() throws IOException {
		return processed ? 1.0f : 0.0f;
	}

	@Override
	public void close() throws IOException {
		// do nothing
	}

	@Override
	public Text createKey() {
		return new Text("");
	}

	@Override
	public Text createValue() {
		return new Text("");
	}

	@Override
	public long getPos() throws IOException {
		return pos;
	}

}