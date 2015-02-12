package onto;

import java.util.ArrayList;
import java.util.HashSet;
import java.util.Random;

import utils.IO;

public class PostProcessing {
	public static final String TRAIN_FILE = "datasets/Pizza8/train.txt";
	public static final String VALID_TEST_FILE = "datasets/Pizza8/valid_test.txt";
	public static final String TEST_FILE = "datasets/Pizza8/test.txt";
	public static final String OTHER_FILE = "datasets/Pizza8/other.txt";
	public static final String VALID_FILE = "datasets/Pizza8/valid.txt";
	public static final String TC_VALID_FILE = "datasets/Pizza8/tc-valid.txt";
	
	public static HashSet<String> getTrainingEntities() throws Exception {
		IO ioTrain = new IO(TRAIN_FILE, "r");		
		HashSet<String> entities = new HashSet<String>();
		
		while(ioTrain.readReady()) {
			String line = ioTrain.readLine();
			String[] triplet = line.split("\t");
			entities.add(triplet[0]);
			entities.add(triplet[2]);
		}
		ioTrain.readClose();
		return entities;
	}
	
	public static void filterQualifiedTriples() throws Exception {
		HashSet<String> trainEntities = getTrainingEntities();

		IO io = new IO(VALID_TEST_FILE, "r");
		IO ioTest = new IO(TEST_FILE, "w"), ioOther = new IO(OTHER_FILE, "w");

		while (io.readReady()) {
			String line = io.readLine();
			String[] triplet = line.split("\t");
			if (trainEntities.contains(triplet[0])
					&& trainEntities.contains(triplet[2])) {
				ioTest.writeLine(line);
			} else {
				ioOther.writeLine(line);
			}
		}

		io.readClose();
		ioTest.writeClose();
		ioOther.writeClose();
	}
	
	public static void splitValidTest() throws Exception {
		ArrayList<String> triples = new ArrayList<String>();		
		IO io = new IO(OTHER_FILE, "r");
		while (io.readReady()) {
			String line = io.readLine();
			triples.add(line);
		}
		io.readClose();
		int size = triples.size() / 5;
		
		Random rand = new Random();
		IO ioTest = new IO(TEST_FILE, "w"), ioValid = new IO(VALID_FILE, "w");
		
		for (int i = 0; i < size; i ++) {
			int index = rand.nextInt(triples.size());
			ioValid.writeLine(triples.get(index));
			triples.remove(index);
		}
		ioValid.writeClose();
		
		for (int i = 0; i < triples.size(); i ++) {
			ioTest.writeLine(triples.get(i));
		}
		ioTest.writeClose();
	}
	
	public static void getTCValid() throws Exception {
		ArrayList<String> trainEntities = new ArrayList<String>(getTrainingEntities());
		
		IO io = new IO(VALID_FILE, "r"), ioTCValid = new IO(TC_VALID_FILE, "w");
		Random rand = new Random();
		while (io.readReady()) {
			String line = io.readLine();
			String[] triplet = line.split("\t");
			ioTCValid.writeLine(line + "\t1");
			// build a negative triple
			if (rand.nextBoolean()) {
				ioTCValid.writeLine(trainEntities.get(rand.nextInt(trainEntities.size())) + "\t" + triplet[1] + "\t" + triplet[2] + "\t-1");
			} else {
				ioTCValid.writeLine(triplet[0] + "\t" + triplet[1] + "\t" + trainEntities.get(rand.nextInt(trainEntities.size())) + "\t-1");
			}
		}
		io.readClose();
		ioTCValid.writeClose();
	}

	public static void main(String args[]) throws Exception {
		getTCValid();
	}
}
