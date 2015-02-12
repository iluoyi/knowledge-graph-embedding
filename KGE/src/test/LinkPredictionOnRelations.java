package test;

import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Iterator;
import java.util.List;
import java.util.Properties;

import pretest.BasicStats;
import utils.IO;
import utils.StringCount;
import utils.TestPair;

/**
 * 
 * Select top 100 frequent relations and test the Link Prediction performance 
 * upon these relations.
 * 
 * @author Yi
 *
 */
public class LinkPredictionOnRelations {
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
	
	public HashMap<String, float[]> bestEntityEmbeddings;
	public HashMap<String, float[]> bestRelationEmbeddings;
	public List<String[]> testExamples;
	
	public HashSet<String> goldTriplets;
	
	public LinkPredictionOnRelations() throws IOException {
		bestEntityEmbeddings = new HashMap<String, float[]>();
		bestRelationEmbeddings = new HashMap<String, float[]>();
		testExamples = new ArrayList<String[]>(); // test dataset
			
		this.CONFIG_PATH = "FB15K_Config.properties";
		
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
	}
	
	public void readEmbeddings(String enetityCSV, String relationCSV) throws Exception {
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
	
	public void loadTest() throws Exception
	{
		IO io = new IO(this.TEST_FILE_PATH, "r");

		while(io.readReady())
		{
			String line = io.readLine();
			String[] triplet = line.split("\t");			
			testExamples.add(triplet);
		}
	}
	public float[] getDistanceEmb(float[] headEmb, float[] relationEmb, float[] tailEmb)
	{
		return embCalculator(embCalculator(headEmb, "+", relationEmb), "-", tailEmb);
	}
	public float[] embCalculator(float[] firstEmb, String operator, float[] secondEmb)
	{
		float[] resultEmb = new float[DIMENSION];
		
		if(operator.equals("+"))
		{
			for(int i = 0; i < DIMENSION; i++)
			{
				resultEmb[i] = firstEmb[i] + secondEmb[i];
			}
		}
		
		else if (operator.equals("-"))
		{
			for(int i = 0; i < DIMENSION; i++)
			{
				resultEmb[i] = firstEmb[i] - secondEmb[i];
			}
		}
		else
		{
			
		}
		return resultEmb;
			
	}
	public float norm(float[] embedding)
	{
		float mode = 0.0f;
		
		if(NORM.equals("L1"))
		{
			for(int i = 0; i < DIMENSION; i++)
			{
				mode += Math.abs(embedding[i]);
			}
			
		}
		else if(NORM.equals("L2"))
		{
			for(int i = 0; i < DIMENSION; i++)
			{
				mode += embedding[i] * embedding[i];
			}
			//mode = (float)Math.sqrt(mode);
		}
		else {
			
		}
	
		return mode;
	}
	// 1. try top 100 relations
	public void test() throws Exception {
		long totalRawHeadRank = 0L;
		long totalRawTailRank = 0L;
		long totalRawHeadHit10 = 0L;
		long totalRawTailHit10 = 0L;
		
		long totalFilterHeadRank = 0L;
		long totalFilterTailRank = 0L;
		long totalFilterHeadHit10 = 0L;
		long totalFilterTailHit10 = 0L;
		
		int total = 0;
		
		List<TestPair> rawTailList = new ArrayList<TestPair>();
		List<TestPair> rawHeadList = new ArrayList<TestPair>();
		List<TestPair> filterTailList = new ArrayList<TestPair>();
		List<TestPair> filterHeadList = new ArrayList<TestPair>();
		
		BasicStats stats = new BasicStats(this.CONFIG_PATH);
	
		// get top 100 relations
		ArrayList<String> relations = stats.readAllRelations("results/FB15k/top_100_nodot_relation.txt");
		stats.getAllRelations(); // initialize goldTriples as well
		
		goldTriplets = stats.getGoldTriples();
		
		IO io = new IO("results/FB15k/TransM/top_100_nodot_relation_stats.csv", "w");
		for (int k = 0; k < 100; k ++) {
			total = 0;
			totalRawHeadRank = 0L;
			totalRawTailRank = 0L;
			totalRawHeadHit10 = 0L;
			totalRawTailHit10 = 0L;
			
			totalFilterHeadRank = 0L;
			totalFilterTailRank = 0L;
			totalFilterHeadHit10 = 0L;
			totalFilterTailHit10 = 0L;
			
			System.out.println("processing " + k + "th relation: " + relations.get(k));
					
			for (int i = 0; i < testExamples.size(); i++) {
				String headEntity = testExamples.get(i)[0];
				String relation = testExamples.get(i)[1];
				String tailEntity = testExamples.get(i)[2];
						
				if (relation.equals(relations.get(k))) {
					total ++;
					float[] bestHeadEntityEmb = bestEntityEmbeddings.get(headEntity);
					float[] bestRelationEmb = bestRelationEmbeddings.get(relation);
					float[] bestTailEntityEmb = bestEntityEmbeddings.get(tailEntity);
		
					rawTailList.clear();
					filterTailList.clear();
					//System.out.println(i + "th tail entity of " + headEntity + "\t" + relation + "\t" + tailEntity);
					/* Replace Tail Entity */
					Iterator<String> entityIt = bestEntityEmbeddings.keySet().iterator();
					while (entityIt.hasNext()) {
						String corruptedTailEntity = entityIt.next();
						float[] corruptedTailEntityEmb = bestEntityEmbeddings.get(corruptedTailEntity);
						float distance = norm(getDistanceEmb(bestHeadEntityEmb, bestRelationEmb, corruptedTailEntityEmb));
						rawTailList.add(new TestPair(corruptedTailEntity, distance));
						
						if(!goldTriplets.contains(headEntity + "\t" + relation + "\t" + corruptedTailEntity))
						{
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
					
					for(int j = 1; j <= filterTailList.size(); j++)
					{
						if(filterTailList.get(j - 1).entity.equals(tailEntity))
						{
							totalFilterTailRank += j;
							if(j <= 10)
								totalFilterTailHit10++;
							break;
						}
					}
		
					rawHeadList.clear();
					filterHeadList.clear();
					//System.out.println(i + "th head entity of " + headEntity + "\t" + relation + "\t" + tailEntity);
					/* Replace Head Entity */
					entityIt = bestEntityEmbeddings.keySet().iterator();
					while (entityIt.hasNext()) {
						String corruptedHeadEntity = entityIt.next();
						float[] corruptedHeadEntityEmb = bestEntityEmbeddings
								.get(corruptedHeadEntity);
						float distance = norm(getDistanceEmb(corruptedHeadEntityEmb, bestRelationEmb, bestTailEntityEmb));
						rawHeadList.add(new TestPair(corruptedHeadEntity, distance));
						
						if(!goldTriplets.contains(corruptedHeadEntity + "\t" + relation + "\t" + tailEntity))
						{
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
					
					for(int j = 1; j <= filterHeadList.size(); j++)
					{
						if(filterHeadList.get(j - 1).entity.equals(headEntity))
						{
							totalFilterHeadRank += j;
							if(j <= 10)
								totalFilterHeadHit10++;
							break;
						}
					}
				}
			}
			
			String oneRelation = relations.get(k);
			double raw_rank = (totalRawHeadRank + totalRawTailRank) * 1.0 / (2 * total);
			double raw_hit = (totalRawHeadHit10 + totalRawTailHit10) * 1.0 / (2 * total);
			double filter_rank = (totalFilterHeadRank + totalFilterTailRank) * 1.0 / (2 * total);
			double filter_hit = (totalFilterHeadHit10 + totalFilterTailHit10) * 1.0 / (2 * total);
			
			System.out.println("****************************");
			System.out.println("Relation: " +  oneRelation);
			System.out.println("RAW_RANK: " + raw_rank);
			System.out.println("RAW_HIT@10: " + raw_hit);
			System.out.println("FILTER_RANK: " + filter_rank);
			System.out.println("FILTER_HIT@10: " + filter_hit);
			io.writeLine(String.format("%s, %.3f, %.3f, %.3f, %.3f", oneRelation, raw_rank, raw_hit, filter_rank, filter_hit));
		}
		io.writeClose();
		System.out.println("Done!");
	}
	
	public static void main(String args[]) throws Exception {
		LinkPredictionOnRelations experiment = new LinkPredictionOnRelations();
		experiment.readEmbeddings("results/FB15k/TransM/EntityEmbeddings.csv", "results/FB15k/TransM/RelationEmbeddings.csv");
		
		experiment.loadTest();
		experiment.test();	
		
//		Set<String> entities = experiment.bestEntityEmbeddings.keySet();
//		for (String entity : entities) {
//			float[] embedding = experiment.bestEntityEmbeddings.get(entity);
//			StringBuilder str = new StringBuilder(entity + "\t");
//			for (int i = 0; i < embedding.length; i++) 
//				str.append(embedding[i] + ", ");
//			System.out.println(str.toString());
//		}
	}
}
