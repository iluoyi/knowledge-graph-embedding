package pretest;

import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Properties;
import java.util.Set;

import utils.IO;
import utils.StringCount;

public class BasicStats {
	public String CONFIG_PATH;
	
	public String TRAIN_FILE_PATH;
	public String DEV_FILE_PATH;
	public String TEST_FILE_PATH;
	public String TC_DEV_FILE_PATH;
	public String TC_TEST_FILE_PATH;
		
	public int DIMENSION;
	public float MARGIN;
	public float STEP_SIZE;
	public int EPOCHES;
	public String NORM;
	
	public HashSet<String> goldTriplets;
	
	public BasicStats() throws IOException {
//		this("FB15K_Config.properties");
		this("Pizza8_Config.properties");
//		this("WN18_Config.properties");
	}
	
	
	public BasicStats(String configStr) throws IOException {
		this.CONFIG_PATH = configStr;
		
		InputStream in = new FileInputStream(this.CONFIG_PATH);
		Properties prop = new Properties();
		prop.load(in);
		
		this.TRAIN_FILE_PATH = prop.getProperty("TRAIN_FILE_PATH");
		this.DEV_FILE_PATH = prop.getProperty("DEV_FILE_PATH");
		this.TEST_FILE_PATH = prop.getProperty("TEST_FILE_PATH");
		
		this.TC_DEV_FILE_PATH = prop.getProperty("TC_DEV_FILE_PATH");
		
		this.DIMENSION = Integer.parseInt(prop.getProperty("DIMENSION"));
		this.EPOCHES = Integer.parseInt(prop.getProperty("EPOCHES"));
		this.MARGIN = Float.parseFloat(prop.getProperty("MARGIN"));
		this.STEP_SIZE = Float.parseFloat(prop.getProperty("STEP_SIZE"));
		this.NORM = prop.getProperty("NORM");
		
		goldTriplets = new HashSet<String>();
	}

	public HashSet<String> getGoldTriples() {
		return this.goldTriplets;
	}
	
	public ArrayList<StringCount> getAllRelations() throws Exception {
		IO io = new IO(this.TRAIN_FILE_PATH, "r");
		HashMap<String, Integer> relationCount = new HashMap<String, Integer>();
		
		while(io.readReady()) {
			String line = io.readLine();
			String[] triplet = line.split("\t");
			String relation = triplet[1];
			
			goldTriplets.add(line);
			
			if (!relationCount.containsKey(relation)) {
				relationCount.put(relation, 1);
			} else {
				relationCount.put(relation, relationCount.get(relation) + 1);
			}
		}
		
		ArrayList<StringCount> relationList = new ArrayList<StringCount>();
		Set<String> relations = relationCount.keySet();
		
		for (String relation : relations)
			relationList.add(new StringCount(relation, relationCount.get(relation)));
		
		Collections.sort(relationList);
		
		return relationList;
	}
	
	
	public ArrayList<String> readAllRelations(String fileName) throws Exception {
		ArrayList<String> relationList = new ArrayList<String>();
		
		IO io = new IO(fileName, "r");
		while(io.readReady()) {
			relationList.add(io.readLine());
		}
		
		return relationList;
	}
	
	
	public void manyTOMany() throws Exception {
		IO io = new IO(this.TRAIN_FILE_PATH, "r");
		HashMap<String, HashMap<String, Integer>> headCount = new HashMap<String, HashMap<String, Integer>>();
		HashMap<String, HashMap<String, Integer>> tailCount = new HashMap<String, HashMap<String, Integer>>();
		
		while(io.readReady()) {
			String line = io.readLine();
			String[] triplet = line.split("\t");
			String head = triplet[0];
			String relation = triplet[1];
			String tail = triplet[2];
			
			if (!headCount.containsKey(relation)) {
				headCount.put(relation, new HashMap<String, Integer>());
			}
			HashMap<String, Integer> relationTail = headCount.get(relation);
			String key = relation + "\t" + tail;
			if (relationTail.containsKey(key)) {
				relationTail.put(key, relationTail.get(key) + 1);
			} else {
				relationTail.put(key, 1);
			}
			
			if (!tailCount.containsKey(relation)) {
				tailCount.put(relation, new HashMap<String, Integer>());
			}
			HashMap<String, Integer> relationHead = tailCount.get(relation);
			key = head + "\t" + relation;
			if (relationHead.containsKey(key)) {
				relationHead.put(key, relationHead.get(key) + 1);
			} else {
				relationHead.put(key, 1);
			}
		}
		
		Set<String> allRelations = headCount.keySet();
		for (String relation : allRelations) {
			System.out.print(relation + ", h = ");
			
			HashMap<String, Integer> relationTail = headCount.get(relation);
			Set<String> keySet = relationTail.keySet(); 
			int sum = 0, count = 0;
			if (!keySet.isEmpty()) {
				for (String key : keySet) {
					sum += relationTail.get(key);
					count ++;
				}
			}
			System.out.print(sum / count + ", t = ");
			
			HashMap<String, Integer> relationHead = tailCount.get(relation);
			keySet = relationHead.keySet(); 
			sum = 0; count = 0;
			if (!keySet.isEmpty()) {
				for (String key : keySet) {
					sum += relationHead.get(key);
					count ++;
				}
				System.out.print(sum / count);
			}
			System.out.println();
		}
		
		io.readClose();
	}
	
	public static void main(String args[]) throws Exception {
		BasicStats stats = new BasicStats();
//		ArrayList<StringCount> allRelations = stats.getAllRelations();
//		
//		//ArrayList<String> all = new ArrayList<String>();
//		for (StringCount strCount : allRelations) {
//			if (strCount.entity.indexOf('.') == -1)
//				System.out.println(strCount.entity + ", " + strCount.count);
//			//all.add(strCount.entity);
//		}
		
		stats.manyTOMany();
		
//		ArrayList<String> top100 = new ArrayList<String>();
//		for (int i = 0; i < 100; i ++) {
//			//System.out.println(relationList.get(relationList.size() - i - 1).entity + ", " + relationList.get(relationList.size() - i - 1).count);
//			top100.add(allRelations.get(allRelations.size() - i - 1).entity);
//		}
		
		//System.out.println(top100);
		//System.out.println(all);
	}
	
}
