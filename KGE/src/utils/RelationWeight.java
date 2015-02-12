package utils;

import java.util.HashSet;

public class RelationWeight {
	public int count;
	
	public HashSet<String> headEntities;
	public HashSet<String> tailEntities;
	
	public float weight;
	
	public RelationWeight()
	{
		this.count = 0;
		this.weight = 0.0f;
		this.headEntities = new HashSet<String>();
		this.tailEntities = new HashSet<String>();
	}
	
	public void weight()
	{
		int headCount = headEntities.size();
		int tailCount = tailEntities.size();
		
		this.weight = (float) (1.0 / (Math.log((count * 1.0) / headCount + (count * 1.0) / tailCount)));			
	}
}
