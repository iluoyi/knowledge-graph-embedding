package train;
import java.io.*;
import java.util.*;

import utils.IO;
import utils.TestPair;

public class TranS {
	
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
	
	public List<String[]> trainExamples;
	public List<String[]> devExamples;
	public List<String[]> testExamples;
	
	public List<String[]> tcDevExamples;
	public List<String[]> tcTestExamples;
	
	public List<String> entityList;
	
	public HashMap<String, float[]> bestEntityEmbeddings;
	public HashMap<String, float[]> bestRelationEmbeddings;
	
	public float minLoss;
	
	public HashMap<String, float[]> entityEmbeddings;
	public HashMap<String, float[]> relationEmbeddings;
	
	public HashSet<String> entitySet;
	public HashSet<String> relationSet;
	
	public Random rand;
	
	public TranS() throws Exception
	{	
		/*FB15K Config File*/
		//this.CONFIG_PATH = "FB15K_Config.properties";
		/*WN18 Config File*/
		this.CONFIG_PATH = "WN18_Config.properties";
		
		rand = new Random();
		
		trainExamples = new ArrayList<String[]>();
		devExamples = new ArrayList<String[]>();
		testExamples = new ArrayList<String[]>();
		
		tcDevExamples = new ArrayList<String[]>();
		tcTestExamples = new ArrayList<String[]>();
		
		entityEmbeddings = new HashMap<String, float[]>();
		relationEmbeddings = new HashMap<String, float[]>();
		
		bestEntityEmbeddings = new HashMap<String, float[]>();
		bestRelationEmbeddings = new HashMap<String, float[]>();
		
		minLoss = Float.MAX_VALUE;
		
		entitySet = new HashSet<String>();
		relationSet = new HashSet<String>();
		
		entityList = new ArrayList<String>();		
		
		InputStream in = new FileInputStream(this.CONFIG_PATH);
		Properties prop = new Properties();
		prop.load(in);
		
		this.TRAIN_FILE_PATH = prop.getProperty("TRAIN_FILE_PATH");
		this.DEV_FILE_PATH = prop.getProperty("DEV_FILE_PATH");
		this.TEST_FILE_PATH = prop.getProperty("TEST_FILE_PATH");
		
		this.DIMENSION = Integer.parseInt(prop.getProperty("DIMENSION"));
		this.EPOCHES = Integer.parseInt(prop.getProperty("EPOCHES"));
		this.MARGIN = Float.parseFloat(prop.getProperty("MARGIN"));
		this.STEP_SIZE = Float.parseFloat(prop.getProperty("STEP_SIZE"));
		this.NORM = prop.getProperty("NORM");
			
	}
	
	public void loadTrain() throws Exception
	{
		IO io = new IO(this.TRAIN_FILE_PATH, "r");

		while(io.readReady())
		{
			String line = io.readLine();
			String[] triplet = line.split("\t");
			
			String headEntity = triplet[0];
			String relation = triplet[1];
			String tailEntity = triplet[2];
			
			entitySet.add(headEntity);
			entitySet.add(tailEntity);
					
			relationSet.add(relation);
			
			trainExamples.add(triplet);
		}
		//entityList.addAll(entitySet);		
	}
	
	public void loadDev() throws Exception
	{
		IO io = new IO(this.DEV_FILE_PATH, "r");

		while(io.readReady())
		{
			String line = io.readLine();
			String[] triplet = line.split("\t");			
			devExamples.add(triplet);
		}
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
	
	public void init()
	{
		Iterator<String> entityIt = entitySet.iterator();
		while(entityIt.hasNext())
		{
			String entityKey = entityIt.next();
			
			float[] entityEmb = initEmb(); 
			entityEmbeddings.put(entityKey, entityEmb);

		}
		
		Iterator<String> relationIt = relationSet.iterator();
		
		while(relationIt.hasNext())
		{
			String relationKey = relationIt.next();
			
			float[] relationEmb = initEmb();
			normEmb(relationEmb);
			relationEmbeddings.put(relationKey, relationEmb);
		}
		
		bestEntityEmbeddings.clear();
		bestEntityEmbeddings.putAll(entityEmbeddings);
		bestRelationEmbeddings.clear();
		bestRelationEmbeddings.putAll(relationEmbeddings);
	}
	
	public float[] initEmb()
	{
		float[] embedding = new float[DIMENSION];
		for(int i = 0; i < DIMENSION; i++)
		{
			embedding[i] = (float) ((rand.nextFloat() - 0.5) / 0.5 * 6.0 / Math.sqrt(DIMENSION * 1.0));
		}
		return embedding;
	}
	
	public void normEmb(float[] emb)
	{
		float mode = 0.0f;
		for(int i = 0; i < DIMENSION; i++)
		{
			mode += emb[i] * emb[i];
		}
		mode = (float)Math.sqrt(mode);
		for(int i = 0; i < DIMENSION; i++)
			emb[i] /= mode;
	}
	public void train()
	{
		for(int i = 0; i < EPOCHES; i++)
		{
			//float totalLoss = 0.0f;
			Iterator<String> entityIt = entitySet.iterator();
			while(entityIt.hasNext())
			{
				String entityKey = entityIt.next();

				float[] entityEmb = entityEmbeddings.get(entityKey);
				normEmb(entityEmb);

			}
			Collections.shuffle(trainExamples);
			for(int j = 0; j < trainExamples.size(); j++)
			{
				String headEntity = trainExamples.get(j)[0];
				String relation = trainExamples.get(j)[1];
				String tailEntity = trainExamples.get(j)[2];
				
				float[] headEmb = entityEmbeddings.get(headEntity);
				float[] relationEmb = relationEmbeddings.get(relation);
				float[] tailEmb = entityEmbeddings.get(tailEntity);
				
				float[] distanceEmb = getDistanceEmb(headEmb, relationEmb, tailEmb);
				float[] gradientEmb = getGradientEmb(distanceEmb);
				
				//totalLoss += norm(distanceEmb);
				
				/*Updating*/
				float[] updatedGradientEmb = embCalculator(gradientEmb, "*", STEP_SIZE);
				
				float[] tmpEmb = entityEmbeddings.get(headEntity);
				entityEmbeddings.put(headEntity, embCalculator(tmpEmb, "-", updatedGradientEmb));
				tmpEmb = entityEmbeddings.get(tailEntity);
				entityEmbeddings.put(tailEntity, embCalculator(tmpEmb, "+", updatedGradientEmb));
				tmpEmb = relationEmbeddings.get(relation);
				relationEmbeddings.put(relation, embCalculator(tmpEmb, "-", updatedGradientEmb));			
			}
			float devAvgLoss = dev();
			if(devAvgLoss < minLoss)
			{
				minLoss = devAvgLoss;
				bestEntityEmbeddings.clear();
				bestEntityEmbeddings.putAll(entityEmbeddings);
				bestRelationEmbeddings.clear();
				bestRelationEmbeddings.putAll(relationEmbeddings);
				
				System.out.println("MIN AVG LOSS@" + i + ": " + minLoss);
				
			}
		}
	}
	
	public float dev()
	{
		float totalLoss = 0.0f;
		for(int i = 0; i < devExamples.size(); i++)
		{
			String headEntity = devExamples.get(i)[0];
			String relation = devExamples.get(i)[1];
			String tailEntity = devExamples.get(i)[2];
			
			float[] distanceEmb = getDistanceEmb(headEntity, relation, tailEntity);
			float loss = norm(distanceEmb);
			
			totalLoss += loss;
		}
		return totalLoss / devExamples.size();
	}
	
	public void test()
	{
		long totalHeadRank = 0L;
		long totalTailRank = 0L;
		List<TestPair> tailList = new ArrayList<TestPair>();
		List<TestPair> headList = new ArrayList<TestPair>();
		
		for(int i = 0; i < testExamples.size(); i++)
		{			
			String headEntity = testExamples.get(i)[0];
			String relation = testExamples.get(i)[1];
			String tailEntity = testExamples.get(i)[2];
			
			float[] bestHeadEntityEmb = bestEntityEmbeddings.get(headEntity);
			float[] bestRelationEmb = bestRelationEmbeddings.get(relation);
			float[] bestTailEntityEmb = bestEntityEmbeddings.get(tailEntity);
			
			tailList.clear();
			/*Replace Tail Entity*/
			Iterator<String> entityIt = entitySet.iterator();
			while(entityIt.hasNext())
			{
				String corruptedTailEntity = entityIt.next();
				float[] corruptedTailEntityEmb = bestEntityEmbeddings.get(corruptedTailEntity);
				float distance = norm(getDistanceEmb(bestHeadEntityEmb, bestRelationEmb, corruptedTailEntityEmb));
				tailList.add(new TestPair(corruptedTailEntity, distance));
			}
			Collections.sort(tailList);
			
			for(int j = 1; j <= tailList.size(); j++)
			{
				if(tailList.get(j - 1).entity.equals(tailEntity))
				{
					totalTailRank += j;
					break;
				}
			}
			
			headList.clear();
			/*Replace Tail Entity*/
			entityIt = entitySet.iterator();
			while(entityIt.hasNext())
			{
				String corruptedHeadEntity = entityIt.next();	
				float[] corruptedHeadEntityEmb = bestEntityEmbeddings.get(corruptedHeadEntity);
				float distance = norm(getDistanceEmb(corruptedHeadEntityEmb, bestRelationEmb, bestTailEntityEmb));
				headList.add(new TestPair(corruptedHeadEntity, distance));
			}
			Collections.sort(headList);
			
			for(int j = 1; j <= headList.size(); j++)
			{
				if(headList.get(j - 1).entity.equals(headEntity))
				{
					totalHeadRank += j;
					break;
				}
			}					
		}
		
		System.out.println((totalHeadRank + totalTailRank) * 1.0 / (2 * testExamples.size()));
	}
	
	public void displayTuple(String[] tuple)
	{
		for(int i = 0; i < tuple.length; i++)
		{
			System.out.print(tuple[i] + ",");
		}
		System.out.println();
	}
	
	public void displayEmb(float[] embedding)
	{
		for(int i = 0; i < DIMENSION; i++)
		{
			System.out.print(embedding[i] + ",");
		}
		System.out.println();
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
			mode = (float)Math.sqrt(mode);
		}
		else {
			
		}
	
		return mode;
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
	
	public float[] embCalculator(float[] firstEmb, String operator, float second)
	{
		float[] resultEmb = new float[DIMENSION];
		
		if(operator.equals("*"))
		{
			for(int i = 0; i < DIMENSION; i++)
			{
				resultEmb[i] = firstEmb[i] * second;
			}
		}
		
		else if (operator.equals("/"))
		{
			for(int i = 0; i < DIMENSION; i++)
			{
				resultEmb[i] = firstEmb[i] / second;
			}
		}
		else
		{
			
		}
		return resultEmb;
	}
	
	public float[] getDistanceEmb(String headEntity, String relation, String tailEntity)
	{
		float[] headEmb = entityEmbeddings.get(headEntity);
		float[] relationEmb = relationEmbeddings.get(relation);
		float[] tailEmb = entityEmbeddings.get(tailEntity);
		
		return embCalculator(embCalculator(headEmb, "+", relationEmb), "-", tailEmb);
	}
	
	public float[] getGradientEmb(float[] distanceEmb)
	{
		float[] gradientEmb = new float[DIMENSION];
		if(NORM.equals("L1"))
		{
			for(int i = 0; i < DIMENSION; i++)
			{
				if(distanceEmb[i] > 0.0f)
					gradientEmb[i] = 1.0f;
				else 
					gradientEmb[i] = -1.0f;
			}
		}
		else if(NORM.equals("L2"))
		{
			for(int i = 0; i < DIMENSION; i++)
			{
				gradientEmb[i] = 2.0f * distanceEmb[i];
			}
		}
		return gradientEmb;
	}
	
	public float[] getDistanceEmb(float[] headEmb, float[] relationEmb, float[] tailEmb)
	{
		return embCalculator(embCalculator(headEmb, "+", relationEmb), "-", tailEmb);
	}
	
	
	public static void main(String[] args) throws Exception
	{
		TranS trans = new TranS();
		
		/*Train and Dev*/
		trans.loadTrain();	
		trans.loadDev();
		trans.init();
		trans.train();
		
		/*Test*/
		trans.loadTest();
		trans.test();		
		/*TC Test*/
	}
	
}


