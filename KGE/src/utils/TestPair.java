package utils;

public class TestPair implements Comparable<TestPair>
{
	public String entity;
	public float distance;
	
	public TestPair(String entity, float distance) {
		this.entity = entity;
		this.distance = distance;
	}
	
	public int compareTo(TestPair o) {
		if (this.distance < o.distance) {
			return -1;
		} else if (this.distance > o.distance) {
			return 1;
		} else {
			return 0;
		}
	}
	
}
