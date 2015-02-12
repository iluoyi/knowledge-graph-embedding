package test;

import java.io.FileInputStream;
import java.io.InputStream;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Iterator;
import java.util.List;
import java.util.Properties;
import java.util.concurrent.ForkJoinPool;
import java.util.concurrent.RecursiveTask;

import utils.AvgRankHitAtTenBean;
import utils.EmbeddingCalculations;
import utils.IO;
import utils.TestPair;

public class TestTransEinParallel {
	private static int TRIPLE_NO = 0;
	
	public static void main(String args[]) throws Exception {
		String CONFIG_PATH = "Pizza8_Config.properties";
		InputStream in = new FileInputStream(CONFIG_PATH);
		Properties prop = new Properties();
		prop.load(in);
		
		String TRAIN_FILE_PATH = prop.getProperty("TRAIN_FILE_PATH");
//		String DEV_FILE_PATH = prop.getProperty("DEV_FILE_PATH");
		String TEST_FILE_PATH = prop.getProperty("TEST_FILE_PATH");
		String EN_EMBD_FILE_PATH = prop.getProperty("EN_EMBD_FILE_PATH");
		String RE_EMBD_FILE_PATH = prop.getProperty("RE_EMBD_FILE_PATH");
		int DIMENSION = Integer.parseInt(prop.getProperty("DIMENSION"));
		String NORM = prop.getProperty("NORM");
		
		HashSet<String> entitySet = new HashSet<String>();
		HashSet<String> goldTriplets = new HashSet<String>();
		HashMap<String, float[]> bestEntityEmbeddings = new HashMap<String, float[]>();
		HashMap<String, float[]> bestRelationEmbeddings = new HashMap<String, float[]>();
		ArrayList<String[]> testExamples = new ArrayList<String[]>();
		
		System.out.println("We have " + Runtime.getRuntime().availableProcessors() + " processors.");
		System.out.println("Start loading...");
		load(TRAIN_FILE_PATH, TEST_FILE_PATH, testExamples, goldTriplets, entitySet);
		readEmbeddings(EN_EMBD_FILE_PATH, RE_EMBD_FILE_PATH, DIMENSION, bestEntityEmbeddings, bestRelationEmbeddings);
		System.out.println("Done loading...");
		
		System.out.println("Start testing...");
		RecursiveTask<AvgRankHitAtTenBean> task = new AvgRankHitAtTenTask(entitySet, goldTriplets, 
				bestEntityEmbeddings, bestRelationEmbeddings, NORM, testExamples, 0, testExamples.size() - 1);
		ForkJoinPool pool = new ForkJoinPool(10);
		AvgRankHitAtTenBean result = pool.invoke(task);
		System.out.println("Done testing...");
		
		System.out.println("****************************");
		System.out.println("RAW_RANK: " + (result.totalRawHeadRank + result.totalRawTailRank) * 1.0 / (2 * testExamples.size()));
		System.out.println("FILTER_RANK: " + (result.totalFilterHeadRank + result.totalFilterTailRank) * 1.0 / (2 * testExamples.size()));
		System.out.println("RAW_HIT@10: " + (result.totalRawHeadHit10 + result.totalRawTailHit10) * 1.0 / (2 * testExamples.size()));
		System.out.println("FILTER_HIT@10: " + (result.totalFilterHeadHit10 + result.totalFilterTailHit10) * 1.0 / (2 * testExamples.size()));
	}
	
	/**
	 * To load testExamples, goldTriplets and entitySet
	 */
	public static void load(String trainFile, String testFile,
			ArrayList<String[]> testExamples, HashSet<String> goldTriplets,
			HashSet<String> entitySet) throws Exception {
		IO io = new IO(trainFile, "r");
		while (io.readReady()) {
			String line = io.readLine();
			goldTriplets.add(line);
			String[] triplet = line.split("\t");
			String headEntity = triplet[0];
			String tailEntity = triplet[2];
			entitySet.add(headEntity);
			entitySet.add(tailEntity);

		}
		io.readClose();
		io = new IO(testFile, "r");
		while (io.readReady()) {
			String line = io.readLine();
			String[] triplet = line.split("\t");
			testExamples.add(triplet);
		}
		io.readClose();
	}
	
	/**
	 * To read embeddings from the file
	 */
	public static void readEmbeddings(String enetityCSV, String relationCSV, int DIMENSION,
			HashMap<String, float[]> bestEntityEmbeddings, HashMap<String, float[]> bestRelationEmbeddings) throws Exception {
		// read embeddings for entities
		IO io = new IO(enetityCSV, "r");
		String line, entity, relation;
		String[] parts, floats;
				
		while((line = io.readLine()) != null) {
			parts = line.split("\t");			
			entity = parts[0];
			floats = parts[1].split(", ");
			float[] embeddings = new float[DIMENSION];
			for (int i = 0; i < floats.length; i ++)
				embeddings[i] = Float.parseFloat(floats[i]);
			bestEntityEmbeddings.put(entity, embeddings); // float[] embeddings is a reference
		}
		io.readClose();
		
		// read embeddings for relations
		io = new IO(relationCSV, "r");

		while ((line = io.readLine()) != null) {
			parts = line.split("\t");
			relation = parts[0];
			floats = parts[1].split(", ");
			float[] embeddings = new float[DIMENSION];
			for (int i = 0; i < floats.length; i++)
				embeddings[i] = Float.parseFloat(floats[i]);
			bestRelationEmbeddings.put(relation, embeddings);
		}
		io.readClose();
	}
	
	/**
	 * Concrete ForkJoinTask
	 */
	private static class AvgRankHitAtTenTask extends RecursiveTask<AvgRankHitAtTenBean> {
		private final static int THRESHOLD = 1000;
		private HashSet<String> entitySet;
		private HashSet<String> goldTriplets;
		private HashMap<String, float[]> bestEntityEmbeddings;
		private HashMap<String, float[]> bestRelationEmbeddings;
		private ArrayList<String[]> testExamples;
		private String norm;
		private int start;
		private int end;
		
		public AvgRankHitAtTenTask(HashSet<String> entitySet, HashSet<String> goldTriplets,
				HashMap<String, float[]> bestEntityEmbeddings, HashMap<String, float[]> bestRelationEmbeddings, 
				String norm, ArrayList<String[]> testExamples, int start, int end) {
			this.entitySet = entitySet;
			this.goldTriplets = goldTriplets;
			this.bestEntityEmbeddings = bestEntityEmbeddings;
			this.bestRelationEmbeddings = bestRelationEmbeddings;
			this.testExamples = testExamples;
			this.norm = norm;
			this.start = start;
			this.end = end;
		}
		
		@Override
		protected AvgRankHitAtTenBean compute() {
			if (end - start < THRESHOLD) {
				long totalRawHeadRank = 0L;
				long totalRawTailRank = 0L;
				long totalFilterHeadRank = 0L;
				long totalFilterTailRank = 0L;
				
				long totalRawHeadHit10 = 0L;
				long totalRawTailHit10 = 0L;
				long totalFilterHeadHit10 = 0L;
				long totalFilterTailHit10 = 0L;
				
				List<TestPair> rawTailList = new ArrayList<TestPair>();
				List<TestPair> rawHeadList = new ArrayList<TestPair>();
				List<TestPair> filterTailList = new ArrayList<TestPair>();
				List<TestPair> filterHeadList = new ArrayList<TestPair>();
				
				for (int i = start; i <= end; i++) {
					System.out.println("Processing " + (++TRIPLE_NO) + "-th of " + testExamples.size() + " triples.");
					String headEntity = testExamples.get(i)[0];
					String relation = testExamples.get(i)[1];
					String tailEntity = testExamples.get(i)[2];

					float[] bestHeadEntityEmb = bestEntityEmbeddings.get(headEntity);
					float[] bestRelationEmb = bestRelationEmbeddings.get(relation);
					float[] bestTailEntityEmb = bestEntityEmbeddings.get(tailEntity);

					rawTailList.clear();
					filterTailList.clear();

					/* Replace Tail Entity */
					Iterator<String> entityIt = entitySet.iterator();
					while (entityIt.hasNext()) {
						String corruptedTailEntity = entityIt.next();
						float[] corruptedTailEntityEmb = bestEntityEmbeddings.get(corruptedTailEntity);
//						if (bestHeadEntityEmb == null || bestRelationEmb == null || corruptedTailEntityEmb == null) 
//							System.out.println(headEntity + "=" + bestHeadEntityEmb +", " + 
//											relation + "=" + bestRelationEmb + ", " +
//											tailEntity + "=" + corruptedTailEntityEmb);
						float distance = EmbeddingCalculations.norm(EmbeddingCalculations.getDistanceEmb(bestHeadEntityEmb, bestRelationEmb, corruptedTailEntityEmb), norm);
						rawTailList.add(new TestPair(corruptedTailEntity, distance));
						if (!goldTriplets.contains(headEntity + "\t" + relation + "\t" + corruptedTailEntity)) {
							filterTailList.add(new TestPair(corruptedTailEntity, distance));
						}
					}
					Collections.sort(rawTailList);
					Collections.sort(filterTailList);

					for (int j = 1; j <= rawTailList.size(); j++) {
						if (rawTailList.get(j - 1).entity.equals(tailEntity)) {
							totalRawTailRank += j;
							if (j <= 10)
								totalRawTailHit10++;
							break;
						}
					}

					for (int j = 1; j <= filterTailList.size(); j++) {
						if (filterTailList.get(j - 1).entity.equals(tailEntity)) {
							totalFilterTailRank += j;
							if (j <= 10)
								totalFilterTailHit10++;
							break;
						}
					}

					rawHeadList.clear();
					filterHeadList.clear();

					/* Replace Head Entity */
					entityIt = entitySet.iterator();
					while (entityIt.hasNext()) {
						String corruptedHeadEntity = entityIt.next();
						float[] corruptedHeadEntityEmb = bestEntityEmbeddings.get(corruptedHeadEntity);
						float distance = EmbeddingCalculations.norm(EmbeddingCalculations.getDistanceEmb(corruptedHeadEntityEmb, bestRelationEmb, bestTailEntityEmb), norm);
						rawHeadList.add(new TestPair(corruptedHeadEntity, distance));
						if (!goldTriplets.contains(corruptedHeadEntity + "\t" + relation + "\t" + tailEntity)) {
							filterHeadList.add(new TestPair(corruptedHeadEntity, distance));
						}
					}
					Collections.sort(rawHeadList);
					Collections.sort(filterHeadList);

					for (int j = 1; j <= rawHeadList.size(); j++) {
						if (rawHeadList.get(j - 1).entity.equals(headEntity)) {
							totalRawHeadRank += j;
							if (j <= 10)
								totalRawHeadHit10++;
							break;
						}
					}
					for (int j = 1; j <= filterHeadList.size(); j++) {
						if (filterHeadList.get(j - 1).entity.equals(headEntity)) {
							totalFilterHeadRank += j;
							if (j <= 10)
								totalFilterHeadHit10++;
							break;
						}
					}
				}
				
				AvgRankHitAtTenBean result = new AvgRankHitAtTenBean();
				result.totalRawHeadRank = totalRawHeadRank;
				result.totalRawTailRank = totalRawTailRank;
				result.totalFilterHeadRank = totalFilterHeadRank;
				result.totalFilterTailRank = totalFilterTailRank;
				
				result.totalRawHeadHit10 = totalRawHeadHit10;
				result.totalRawTailHit10 = totalRawTailHit10;
				result.totalFilterHeadHit10 = totalFilterHeadHit10;
				result.totalFilterTailHit10 = totalFilterTailHit10;
				return result;
			} else {
				int mid = (start + end) / 2;
				RecursiveTask<AvgRankHitAtTenBean> left = new AvgRankHitAtTenTask(entitySet, goldTriplets, 
																	bestEntityEmbeddings, bestRelationEmbeddings, norm, testExamples, start, mid - 1);
				RecursiveTask<AvgRankHitAtTenBean> right = new AvgRankHitAtTenTask(entitySet, goldTriplets, 
																	bestEntityEmbeddings, bestRelationEmbeddings, norm, testExamples, mid, end);
				left.fork();
				right.fork();
				
				AvgRankHitAtTenBean result1 = left.join();
				AvgRankHitAtTenBean result2 = right.join();
				
				return AvgRankHitAtTenBean.merge(result1, result2);
			}
		}
	}
}
