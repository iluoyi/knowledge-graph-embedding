package utils;

import java.util.Arrays;
import java.util.Random;

public class RandomWithBlackList {
	private Random rand;
	
	
	public RandomWithBlackList() {
		rand = new Random();
	}
	
	public int nextInt(int bound) {
		return rand.nextInt(bound);
	}
	
	public boolean nextBoolean() {
		return rand.nextBoolean();
	}
			
	public int nextInt(int bound, Object[] blackList) {
		if (blackList == null) {
			return nextInt(bound);
		} else {
			int len = bound - blackList.length;
			int n = rand.nextInt(len);
			Arrays.sort(blackList);
			
			for (int i = 0; i < blackList.length; i ++) {
				if (n >= i) // jump the hole
					n ++;
				else
					break;
			}
			
			return n;
		}
	}
}
