package train;
import java.io.*;
import java.util.*;

import utils.IO;
import utils.TestPair;

public class TrainTransE {
	
	public String CONFIG_PATH;
		
	public String TRAIN_FILE_PATH;
	public String DEV_FILE_PATH;
	public String TEST_FILE_PATH;
	public String EN_EMBD_FILE_PATH;
	public String RE_EMBD_FILE_PATH;
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
	
	public HashSet<String> goldTriplets;
	
	public Random rand;
	
	public TrainTransE() throws Exception
	{	
		/*FB15K Config File*/
		this.CONFIG_PATH = "FB15K_Config.properties";
		/*WN18 Config File*/
		//this.CONFIG_PATH = "WN18_Config.properties";
		/*Pizza8 Config File*/
		//this.CONFIG_PATH = "Pizza8_Config.properties";
		
		rand = new Random();
		
		trainExamples = new ArrayList<String[]>();
		devExamples = new ArrayList<String[]>(); // validation dataset
		testExamples = new ArrayList<String[]>(); // test dataset
		
		tcDevExamples = new ArrayList<String[]>();
		tcTestExamples = new ArrayList<String[]>();
		
		entityEmbeddings = new HashMap<String, float[]>();
		relationEmbeddings = new HashMap<String, float[]>();
		
		bestEntityEmbeddings = new HashMap<String, float[]>();
		bestRelationEmbeddings = new HashMap<String, float[]>();
		
		minLoss = Float.MAX_VALUE;
		
		entitySet = new HashSet<String>();
		relationSet = new HashSet<String>();
		
		goldTriplets = new HashSet<String>();
		
		entityList = new ArrayList<String>();		
		
		InputStream in = new FileInputStream(this.CONFIG_PATH);
		Properties prop = new Properties();
		prop.load(in);
		
		this.TRAIN_FILE_PATH = prop.getProperty("TRAIN_FILE_PATH");
		this.DEV_FILE_PATH = prop.getProperty("DEV_FILE_PATH");
		this.TEST_FILE_PATH = prop.getProperty("TEST_FILE_PATH");
		
		this.EN_EMBD_FILE_PATH = prop.getProperty("EN_EMBD_FILE_PATH");
		this.RE_EMBD_FILE_PATH = prop.getProperty("RE_EMBD_FILE_PATH");
		
		this.TC_DEV_FILE_PATH = prop.getProperty("TC_DEV_FILE_PATH");
		
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
			goldTriplets.add(line);
			String[] triplet = line.split("\t");
			
			String headEntity = triplet[0];
			String relation = triplet[1];
			String tailEntity = triplet[2];
			
			entitySet.add(headEntity);
			entitySet.add(tailEntity);
					
			relationSet.add(relation);
			
			trainExamples.add(triplet);
		}
		entityList.addAll(entitySet);		
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
	
	public void loadTCDev() throws Exception
	{
		IO io = new IO(this.TC_DEV_FILE_PATH, "r");

		while(io.readReady())
		{
			String line = io.readLine();
			String[] tuple = line.split("\t");
			if(tuple[3].equals("1"))
				goldTriplets.add(tuple[0] + "\t" + tuple[1] + "\t" + tuple[2]);
			tcDevExamples.add(tuple); //both pos and neg triples
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
			entityEmbeddings.put(entityKey, entityEmb); // uniform distribution
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
			Iterator<String> entityIt = entitySet.iterator();
//			System.out.println(i + " = " + i + ": loop start with entity size = " + entitySet.size());
			while(entityIt.hasNext())
			{
				String entityKey = entityIt.next();

				float[] entityEmb = entityEmbeddings.get(entityKey);
				normEmb(entityEmb);
			} // to normalize every entity embedding
			
			Collections.shuffle(trainExamples);
//			System.out.println(i + " = " + i + ": shuffle done, start loop of " + trainExamples.size());
			
			for(int j = 0; j < trainExamples.size(); j++)
			{
				String[] posTriplet = trainExamples.get(j);
				String[] negTriplet = getNegTriplet(posTriplet);
				
				String posHeadEntity = posTriplet[0];
				String posTailEntity = posTriplet[2];
				String negHeadEntity = negTriplet[0];
				String negTailEntity = negTriplet[2];
				String relation = posTriplet[1];
				
				float[] posHeadEmb = entityEmbeddings.get(posHeadEntity);
				float[] posTailEmb = entityEmbeddings.get(posTailEntity);
				
				float[] negHeadEmb = entityEmbeddings.get(negHeadEntity);
				float[] negTailEmb = entityEmbeddings.get(negTailEntity);
				
				float[] relationEmb = relationEmbeddings.get(relation);
				
				float[] posDistanceEmb = getDistanceEmb(posHeadEmb, relationEmb, posTailEmb);
				float[] negDistanceEmb = getDistanceEmb(negHeadEmb, relationEmb, negTailEmb);
				
				float posDistance = norm(posDistanceEmb);
				float negDistance = norm(negDistanceEmb);
				
				if(MARGIN + posDistance - negDistance > 0.0f) // gradient descent
				{
					float[] posGradientEmb = getGradientEmb(posDistanceEmb);
					float[] negGradientEmb = getGradientEmb(negDistanceEmb);
					
					float[] posUpdatedGradientEmb = embCalculator(posGradientEmb, "*", STEP_SIZE);
					float[] negUpdatedGradientEmb = embCalculator(negGradientEmb, "*", STEP_SIZE);
					
					float[] tmpEmb = entityEmbeddings.get(posHeadEntity);
					entityEmbeddings.put(posHeadEntity, embCalculator(tmpEmb, "-", posUpdatedGradientEmb));
					
					tmpEmb = entityEmbeddings.get(posTailEntity);
					entityEmbeddings.put(posTailEntity, embCalculator(tmpEmb, "+", posUpdatedGradientEmb));
					
					tmpEmb = entityEmbeddings.get(negHeadEntity);
					entityEmbeddings.put(negHeadEntity, embCalculator(tmpEmb, "+", negUpdatedGradientEmb));
					
					tmpEmb = entityEmbeddings.get(negTailEntity);
					entityEmbeddings.put(negTailEntity, embCalculator(tmpEmb, "-", negUpdatedGradientEmb));
					
					tmpEmb = relationEmbeddings.get(relation);
					relationEmbeddings.put(relation, embCalculator(tmpEmb, "-", embCalculator(posUpdatedGradientEmb, "-", negUpdatedGradientEmb)));
				}
			}
			
			float devAvgLoss = dev();
//			System.out.println(i + " = " + i + ": dev avg loos done.");
			if(devAvgLoss < minLoss)
			{
				minLoss = devAvgLoss;
				bestEntityEmbeddings.clear();
				bestEntityEmbeddings.putAll(entityEmbeddings);
				bestRelationEmbeddings.clear();
				bestRelationEmbeddings.putAll(relationEmbeddings);
				System.out.println(i + " = " + i + ": MIN AVG LOSS - " + minLoss);
			}
			System.out.println(i + " = " + i + ": whole i-th loop done.");
		}
	}
	
	/**
	 * To replace the head or the tail randomly
	 */
	public String[] getNegTriplet(String[] posTriplet)
	{
		String[] negTriplet = new String[3];
		
		if(rand.nextFloat() < 0.5)
		{
			/*Replacing Head*/
			String corruptedHeadEntity = posTriplet[0];
			while(corruptedHeadEntity.equals(posTriplet[0]))
			{
				int corruptedHeadIdx = rand.nextInt(entityList.size());
				corruptedHeadEntity = entityList.get(corruptedHeadIdx);
			}
			negTriplet[0] = corruptedHeadEntity;
			negTriplet[1] = posTriplet[1];
			negTriplet[2] = posTriplet[2];
		}
		else {
			/*Replacing Tail*/
			String corruptedTailEntity = posTriplet[2];
			while(corruptedTailEntity.equals(posTriplet[2]))
			{
				int corruptedTailIdx = rand.nextInt(entityList.size());
				corruptedTailEntity = entityList.get(corruptedTailIdx);
			}
			negTriplet[0] = posTriplet[0];
			negTriplet[1] = posTriplet[1];
			negTriplet[2] = corruptedTailEntity;
		}
		return negTriplet;
	}
	
	public float dev()
	{
		float totalLoss = 0.0f;
		for(int i = 0; i < tcDevExamples.size(); i+=2)
		{
			String[] posTuple = tcDevExamples.get(i);
			String[] negTuple = tcDevExamples.get(i + 1);

			float[] posDistanceEmb = getDistanceEmb(posTuple[0], posTuple[1], posTuple[2]);
			float posDistance = norm(posDistanceEmb);
			
			float[] negDistanceEmb = getDistanceEmb(negTuple[0], negTuple[1], negTuple[2]);
			float negDistance = norm(negDistanceEmb);
			
			float loss = MARGIN + posDistance - negDistance;
			if(loss > 0.0f)
			{
				totalLoss += loss;
			}	
		}
		return totalLoss / tcDevExamples.size();
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
			//mode = (float)Math.sqrt(mode);
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
	
	/**
	 * To store embeddings into the disk
	 * @throws Exception 
	 */
	public void storeEmbeddings() throws Exception {
		IO io = new IO(EN_EMBD_FILE_PATH, "w");
		
		Set<String> entities = this.bestEntityEmbeddings.keySet();
		for (String entity : entities) {
			float[] embedding = bestEntityEmbeddings.get(entity);
			StringBuilder str = new StringBuilder(entity + "\t");
			for (int i = 0; i < embedding.length; i++) 
				str.append(embedding[i] + ", ");
			
			io.writeLine(str.toString());
		}
		
		io.writeClose();
		
		io = new IO(RE_EMBD_FILE_PATH, "w");

		Set<String> relations = this.bestRelationEmbeddings.keySet();
		for (String relation : relations) {
			float[] embedding = bestRelationEmbeddings.get(relation);
			StringBuilder str = new StringBuilder(relation + "\t");
			for (int i = 0; i < embedding.length; i++) 
				str.append(embedding[i] + ", ");
			
			io.writeLine(str.toString());
		}
		
		io.writeClose();
		System.out.println("Embeddings stored!");
	}
	
	public static void main(String[] args) throws Exception
	{
		TrainTransE transe = new TrainTransE();
		/*Train and Dev*/
		transe.loadTrain();	
			//transe.loadDev();
		transe.loadTCDev();
		transe.init();
		transe.train();
		
		/*To store best embbedings*/
		transe.storeEmbeddings();
	}
	
}

