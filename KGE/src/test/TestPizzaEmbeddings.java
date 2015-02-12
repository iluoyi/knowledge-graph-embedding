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

import utils.AvgRankHitAtTenBean;
import utils.EmbeddingCalculations;
import utils.IO;
import utils.TestPair;

public class TestPizzaEmbeddings {
	public static String[] pizzaRelations = {"hasCountryOrigin", "hasIngredient", "isIngredientOf", 
										"hasTopping", "isToppingOf", "hasBase", "isBaseOf", "hasSpiciness"};
	
	public static void main(String args[]) throws Exception {
		if (args.length != 2) {
			System.out.println("Usage: TestPizzaEmbeddings configFile relationNo.");
		} else {
			int rIndex = Integer.parseInt(args[1]); // which relation?
			int rSize = pizzaRelations.length;
			if (rIndex >= rSize || rIndex < 0) {
				System.out.println("Desired relation index is out of the relation size.");
			} else {
				String aimRelation = pizzaRelations[rIndex];
				String CONFIG_PATH = args[0];			
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
				
				System.out.println("Start loading...");
				readEmbeddings(EN_EMBD_FILE_PATH, RE_EMBD_FILE_PATH, DIMENSION, bestEntityEmbeddings, bestRelationEmbeddings);
				load(TRAIN_FILE_PATH, TEST_FILE_PATH, aimRelation, testExamples, goldTriplets, entitySet);
				System.out.println("Loaded " + testExamples.size() + " triples for relation = " + aimRelation);
				System.out.println("Done loading...");
				System.out.println();
				System.out.println("Start testing...");
				AvgRankHitAtTenBean result = test(entitySet, goldTriplets, bestEntityEmbeddings, bestRelationEmbeddings, NORM, testExamples);
				System.out.println("Done testing...");
				
				System.out.println("****************************");
				System.out.println("RAW_RANK: " + (result.totalRawHeadRank + result.totalRawTailRank) * 1.0 / (2 * testExamples.size()));
				System.out.println("FILTER_RANK: " + (result.totalFilterHeadRank + result.totalFilterTailRank) * 1.0 / (2 * testExamples.size()));
				System.out.println("RAW_HIT@10: " + (result.totalRawHeadHit10 + result.totalRawTailHit10) * 1.0 / (2 * testExamples.size()));
				System.out.println("FILTER_HIT@10: " + (result.totalFilterHeadHit10 + result.totalFilterTailHit10) * 1.0 / (2 * testExamples.size()));
			}
		}
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
	 * To load testExamples, goldTriplets and entitySet
	 */
	public static void load(String trainFile, String testFile, String aimRelation,
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
			if (aimRelation.equals(triplet[1]))
				testExamples.add(triplet); // only process target relations
		}
		io.readClose();
	}
	
	public static AvgRankHitAtTenBean test(HashSet<String> entitySet, HashSet<String> goldTriplets,
			HashMap<String, float[]> bestEntityEmbeddings, HashMap<String, float[]> bestRelationEmbeddings, 
			String norm, ArrayList<String[]> testExamples) {
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
		
		for (int i = 0; i < testExamples.size(); i++) {
			System.out.println("Processing " + i + "-th of " + testExamples.size() + " triples.");
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
	}
	
}
