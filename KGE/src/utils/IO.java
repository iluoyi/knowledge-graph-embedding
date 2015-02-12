package utils;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;

public class IO {

	BufferedReader br;
	BufferedWriter bw;

	public IO(String filePath, String mode) throws Exception {
		if (mode.equals("r")) {
			br = new BufferedReader(new FileReader(filePath));
		} else {
			bw = new BufferedWriter(new FileWriter(filePath));
		}
	}

	public boolean readReady() throws Exception {
		return br.ready();
	}

	public String readLine() throws Exception {
		return br.readLine();
	}

	public void writeLine(String line) throws IOException {
		bw.write(line + "\n");
	}

	public void readClose() throws Exception {
		br.close();
	}

	public void writeClose() throws Exception {
		bw.close();
	}

	public static void assignTrainOrTest(IO train, IO test, String line,
			float prop, float thrd) throws IOException {
		if (train != null && test != null) {
			if (prop < thrd)
				train.writeLine(line);
			else
				test.writeLine(line);
		}
	}
}
