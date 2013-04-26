package yuncong;

import java.io.ByteArrayInputStream;
import java.io.DataOutputStream;
import java.io.DataInputStream;
import java.io.IOException;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.fs.FSDataOutputStream;

import org.apache.hadoop.mapred.FileOutputFormat;
import org.apache.hadoop.mapred.JobConf;
import org.apache.hadoop.mapred.RecordWriter;
import org.apache.hadoop.mapred.Reporter;
import org.apache.hadoop.io.NullWritable;
import org.apache.hadoop.io.BytesWritable;
import org.apache.hadoop.util.Progressable;

public class RawFileOutputFormat<K, V> extends FileOutputFormat<K, V> {
	protected static class RawFileRecordWriter<K, V> implements
			RecordWriter<K, V> {
		protected DataOutputStream out;

		public RawFileRecordWriter(DataOutputStream out) {
			this.out = out;
		}

		private byte[] readRawBytes(BytesWritable bv) throws IOException {
			byte[] typedbytesContent = bv.getBytes();
			ByteArrayInputStream bais = new ByteArrayInputStream(
					typedbytesContent);
			DataInputStream in = new DataInputStream(bais);
			try {
				in.readUnsignedByte();// read the code and discard it
			} catch (Exception eof) {
				return null;
			}
			int length = in.readInt();
			byte[] bytes = new byte[length];
			in.readFully(bytes);
			return bytes;
		}

		public synchronized void write(K key, V value) throws IOException {
			boolean nullValue = (value == null)
					|| value instanceof NullWritable;
			if (!nullValue) {
				BytesWritable bv = (BytesWritable) value;
				out.write(readRawBytes(bv));
			}
		}

		public synchronized void close(Reporter reporter) throws IOException {
			out.close();
		}
	}

	@Override
	public RecordWriter<K, V> getRecordWriter(FileSystem ignored, JobConf job,
			String name, Progressable progress) throws IOException {
		Path file = FileOutputFormat.getTaskOutputPath(job, name);
		FileSystem fs = file.getFileSystem(job);
		FSDataOutputStream fileOut = fs.create(file, progress);
		return new RawFileRecordWriter<K, V>(fileOut);
	}
}