package utils;

public class StringCount implements Comparable<StringCount> {
	public String entity;
	public int count;
	
	public StringCount(String entity, int count) {
		this.entity = entity;
		this.count = count;
	}
	
	public int compareTo(StringCount o) {
		if (this.count < o.count) {
			return -1;
		} else if (this.count > o.count)
		{
			return 1;
		} else {
			return 0;
		}
	}
}
